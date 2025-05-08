"""
Dataset utilities for the DeepEyeNet dataset.
Handles image loading, caption preprocessing, and dataset creation.
"""

import os
import re
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_caption(keywords, clinical_desc):
    """
    Preprocess captions by normalizing text and expanding medical abbreviations.
    
    Args:
        keywords (str): Keywords associated with the clinical description.
        clinical_desc (str): Clinical description of the medical image.
    
    Returns:
        str: Processed caption combining description and keywords.
    """
    # Normalize slashes and spaces
    clinical_desc = re.sub(r'(\d+)\\(\d+)', r'\1/\2', clinical_desc)
    keywords = re.sub(r'(\d+)\\(\d+)', r'\1/\2', keywords)
    keywords = re.sub(r'\s*\\\s*', ', ', keywords)
    clinical_desc = re.sub(r'\s+', ' ', clinical_desc.strip().lower())
    keywords = re.sub(r'\s+', ' ', keywords.strip().lower())
    
    # Combine description and keywords
    if keywords:
        combined = f"{clinical_desc} Keywords: {keywords}"
    else:
        combined = clinical_desc
    
    # Expand medical abbreviations
    abbreviation_map = {
        'fa': 'fluorescein angiography',
        'oct': 'optical coherence tomography',
        'bdr': 'background diabetic retinopathy',
        'srnv': 'subretinal neovascularization',
        're': 'right eye',
        'le': 'left eye',
        'ou': 'both eyes'
    }
    for abbr, full in abbreviation_map.items():
        combined = re.sub(r'\b' + abbr + r'\b', full, combined)
    return combined

def load_and_process_json(json_file):
    """
    Load and process JSON file containing image paths and captions.
    
    Args:
        json_file (str): Path to the JSON file.
    
    Returns:
        tuple: Lists of valid image paths and corresponding captions.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_paths, captions = [], []
    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            logger.warning(f"Invalid item format: {item}")
            continue
        image_path, value = next(iter(item.items()))
        if not isinstance(value, dict) or "keywords" not in value or "clinical-description" not in value:
            logger.warning(f"Invalid value format for {image_path}: {value}")
            continue
        full_path = os.path.normpath(os.path.join(image_path))
        if os.path.exists(full_path):
            image_paths.append(full_path)
            caption = preprocess_caption(value["keywords"], value["clinical-description"])
            captions.append(caption)
        else:
            logger.warning(f"Image not found at {full_path}")
    return image_paths, captions

class EyeNetDataset(Dataset):
    """
    Custom dataset for the DeepEyeNet dataset, pairing medical images with captions.
    """
    def __init__(self, image_paths, captions, transform, tokenizer, max_length):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of image file paths.
            captions (list): List of corresponding captions.
            transform: Image transformation pipeline.
            tokenizer: Text tokenizer for captions.
            max_length (int): Maximum length for tokenized captions.
        """
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Load and preprocess a single sample.
        
        Args:
            index (int): Index of the sample.
        
        Returns:
            dict: Dictionary containing pixel values, input IDs, and attention mask.
        """
        try:
            logger.info(f"Loading image {index}: {self.image_paths[index]}")
            image = Image.open(self.image_paths[index]).convert("RGB")
            logger.info(f"Image {index} loaded")
            
            logger.info(f"Applying transforms to image {index}")
            image = self.transform(image)
            logger.info(f"Transforms applied to image {index}")
            
            logger.info(f"Tokenizing caption {index}")
            caption = self.captions[index]
            inputs = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            logger.info(f"Caption {index} tokenized")
            
            return {
                "pixel_values": image,
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze()
            }
        except Exception as e:
            logger.error(f"Error in EyeNetDataset.__getitem__ for index {index}: {str(e)}")
            raise
