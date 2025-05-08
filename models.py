"""
Model definitions for the vision-text integration pipeline.
Includes MedViT for vision processing and VisionTextModel for combining vision and text embeddings.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import math
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NORM_EPS = 1e-5

# Helper classes for MedViT
class ConvBNReLU(nn.Module):
    """Convolution-BatchNorm-ReLU block for efficient feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

def _make_divisible(v, divisor, min_value=None):
    """Ensure channel counts are divisible by a specified divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class PatchEmbed(nn.Module):
    """Patch embedding layer with optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))

class MHCA(nn.Module):
    """Multi-Head Channel Attention for local feature aggregation."""
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out

class h_sigmoid(nn.Module):
    """Hard sigmoid activation function."""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    """Hard swish activation function."""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ECALayer(nn.Module):
    """Efficient Channel Attention layer."""
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid() if sigmoid else h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LocalityFeedForward(nn.Module):
    """Locality-aware feed-forward network with depth-wise convolutions."""
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3
        layers = [
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
        ]
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)
        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError(f'Activation type {act} is not implemented')
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x

class ECB(nn.Module):
    """Efficient Convolution Block for MedViT."""
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0, drop=0, head_dim=32, mlp_ratio=3):
        super(ECB, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)
        self.conv = LocalityFeedForward(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        x = self.norm(x)
        x = x + self.conv(x)
        return x

class E_MHSA(nn.Module):
    """Efficient Multi-Head Self-Attention for MedViT."""
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=sr_ratio ** 2, stride=sr_ratio ** 2)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LTB(nn.Module):
    """Local-Transformer Block for MedViT."""
    def __init__(self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
                 mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0):
        super(LTB, self).__init__()
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels
        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout)
        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))
        self.norm2 = norm_func(out_channels)
        self.conv = LocalityFeedForward(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        out = self.norm1(x)
        out = rearrange(out, "b c h w -> b (h w) c")
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + rearrange(out, "b (h w) c -> b c h w", h=H)
        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = torch.cat([x, out], dim=1)
        out = self.norm2(x)
        x = x + self.conv(out)
        return x

class MedViT(nn.Module):
    """
    MedViT: A vision transformer model designed for medical image processing.
    Reference: https://github.com/Omid-Nejati/MedViT
    """
    def __init__(self, stem_chs, depths, path_dropout=0.2, num_classes=None,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75):
        super(MedViT, self).__init__()
        self.stage_out_channels = [[96] * depths[0],
                                   [192] * (depths[1] - 1) + [256],
                                   [384, 384, 384, 384, 512] * (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [1024]]
        self.stage_block_types = [[ECB] * depths[0],
                                  [ECB] * (depths[1] - 1) + [LTB],
                                  [ECB, ECB, ECB, ECB, LTB] * (depths[2] // 5),
                                  [ECB] * (depths[3] - 1) + [LTB]]
        
        # Initialize stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        
        # Initialize stages
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                stride = strides[stage_id] if block_id == 0 else 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is ECB:
                    layer = ECB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id])
                elif block_type is LTB:
                    layer = LTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio)
                features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)
        
        # Final norm and avgpool
        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear projection head if num_classes is provided
        if num_classes is not None:
            self.proj_head = nn.Linear(output_channel, num_classes)
        else:
            self.proj_head = None
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming and truncated normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through MedViT, producing a flattened feature vector."""
        x = self.stem(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.proj_head is not None:
            x = self.proj_head(x)
        return x

    def load_pretrained_weights(self, weight_path):
        """Load pretrained weights for MedViT, ignoring classification head if num_classes=None."""
        state_dict = torch.load(weight_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if self.proj_head is None:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('proj_head')}
        self.load_state_dict(state_dict, strict=False)

class VisionTextModel(nn.Module):
    """
    VisionTextModel: Integrates MedViT vision features with BioGpt text embeddings for multimodal captioning.
    Concatenates vision and text embeddings, adjusting labels to match sequence length.
    """
    def __init__(self, vision_model, text_model, vision_dim, text_dim):
        super(VisionTextModel, self).__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        logger.info(f"VisionTextModel initialized with vision_dim={vision_dim}, text_dim={text_dim}")

    def forward(self, pixel_values, input_ids, attention_mask):
        """Forward pass combining vision and text embeddings, with adjusted labels."""
        try:
            logger.info("Running vision model")
            vision_features = self.vision_model(pixel_values)  # [batch, vision_dim=1024]
            logger.info(f"Vision features shape: {vision_features.shape}")

            # Reshape vision features
            batch_size = pixel_values.size(0)  # Explicitly get batch size
            vision_embed = vision_features.reshape(batch_size, 1, vision_features.size(-1))  # [batch, 1, 1024]
            logger.info(f"Reshaped vision embed shape: {vision_embed.shape}")

            logger.info("Getting text embeddings")
            text_embed = self.text_model.get_input_embeddings()(input_ids)  # [batch, seq_len, text_dim=1024]
            logger.info(f"Text embed shape: {text_embed.shape}")

            # Verify dimensions match for concatenation
            if vision_embed.size(-1) != text_embed.size(-1):
                logger.error(f"Vision embed dimension {vision_embed.size(-1)} does not match text embed dimension {text_embed.size(-1)}")
                raise ValueError(f"Vision embed dimension {vision_embed.size(-1)} does not match text embed dimension {text_embed.size(-1)}")

            logger.info("Concatenating embeddings")
            combined_embed = torch.cat([vision_embed, text_embed], dim=1)  # [batch, 1+seq_len, 1024]
            logger.info(f"Combined embed shape: {combined_embed.shape}")

            # Create combined attention mask
            combined_mask = torch.cat([torch.ones(batch_size, 1, device=attention_mask.device), attention_mask], dim=1)  # [batch, 1+seq_len]
            logger.info(f"Combined mask shape: {combined_mask.shape}")

            # Adjust labels to match combined_embed sequence length
            logger.info("Adjusting labels")
            pad_token_id = self.text_model.config.pad_token_id or 0  # Use BioGpt's pad token
            pad_tokens = torch.full((batch_size, 1), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            adjusted_labels = torch.cat([pad_tokens, input_ids], dim=1)  # [batch, 1+seq_len]
            logger.info(f"Adjusted labels shape: {adjusted_labels.shape}")

            # Verify shapes
            if combined_embed.size(0) != adjusted_labels.size(0):
                logger.error(f"Batch size mismatch: combined_embed {combined_embed.size(0)} vs adjusted_labels {adjusted_labels.size(0)}")
                raise ValueError(f"Batch size mismatch: combined_embed {combined_embed.size(0)} vs adjusted_labels {adjusted_labels.size(0)}")
            if combined_embed.size(1) != adjusted_labels.size(1):
                logger.error(f"Sequence length mismatch: combined_embed {combined_embed.size(1)} vs adjusted_labels {adjusted_labels.size(1)}")
                raise ValueError(f"Sequence length mismatch: combined_embed {combined_embed.size(1)} vs adjusted_labels {adjusted_labels.size(1)}")

            logger.info("Running text model")
            outputs = self.text_model(
                inputs_embeds=combined_embed,
                attention_mask=combined_mask,
                labels=adjusted_labels
            )
            logger.info("Text model forward pass completed")
            return outputs
        except Exception as e:
            logger.error(f"Error in VisionTextModel.forward: {str(e)}")
            raise
