"""
Main execution script for training a vision-text model on the DeepEyeNet dataset.
This script orchestrates the pipeline, defines global configurations, and integrates
the MedViT vision model with BioGpt for multimodal medical image captioning.
"""

import os
import logging
import torch
import torchvision.transforms as transforms
from models import MedViT, VisionTextModel
from dataset import EyeNetDataset, load_and_process_json
from train import train_model_test
from transformers import BioGptForCausalLM, BioGptTokenizer

# Configure logging for detailed debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Clear CUDA cache and optimize memory usage
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Configuration class for global variables
class Config:
    """Configuration class holding global hyperparameters and paths."""
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model hyperparameters
    VISION_DIM = 1024  # Output dimension of MedViT
    TEXT_DIM = 1024    # Embedding dimension of BioGpt
    STEM_CHS = [64, 32, 64]  # Channels in MedViT stem
    DEPTHS = [3, 4, 20, 3]   # Depths of MedViT stages
    BATCH_SIZE = 2            # Batch size for training
    ACCUM_STEPS = 4           # Gradient accumulation steps
    NUM_EPOCHS = 5            # Number of training epochs
    LR = 2e-5                 # Learning rate
    WEIGHT_DECAY = 0.01       # Weight decay for AdamW
    WARMUP_RATIO = 0.05       # Warmup steps ratio
    MAX_LENGTH = 256          # Maximum caption length
    
    ################# BEAM SEARCH
    EARLY_STOPPING = False # make it true to use beam_search
    BEAM_SEARCH = 1 # Assign it to 3 or 5, 1 is greedy search
    
    ################# ACTIVATE THE CONTRASTIVE LOSS
    CONS_LOSS = False
    
    # Paths
    PRETRAINED_PATH = 'MedViT_base_im1k.pth'  # Path to pretrained MedViT weights
    OUTPUT_DIR = 'MedBio'                     # Directory for model outputs
    JSON_PATHS = {
        'train': 'eyenet0420/DeepEyeNet_train.json',
        'val': 'eyenet0420/DeepEyeNet_valid.json',
        'test': 'eyenet0420/DeepEyeNet_test.json'
    }
    
    # Environment settings for CUDA debugging
    CUDA_ENV = {
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_USE_CUDA_DSA": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", 
        "TF_ENABLE_ONEDNN_OPTS": "0"  
    }

def setup_environment():
    """Set up environment variables for CUDA debugging and memory optimization."""
    for key, value in Config.CUDA_ENV.items():
        os.environ[key] = value
    if Config.DEVICE.type == "cpu":
        logger.warning("CUDA not available, falling back to CPU. Training may be slow.")
    logger.info(f"Using device: {Config.DEVICE}")

def define_transforms():
    """Define data augmentation and normalization transforms for training and evaluation."""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, eval_transforms

def main():
    """Main pipeline for training and evaluating the vision-text model."""
    try:
        logger.info("Starting training pipeline")
        
        # Set up environment and random seed
        setup_environment()
        logger.info("Setting random seed for reproducibility")
        torch.manual_seed(42)

        # Initialize vision model (MedViT)
        logger.info("Initializing MedViT model")
        vision_model = MedViT(stem_chs=Config.STEM_CHS, depths=Config.DEPTHS, num_classes=None)
        logger.info(f"Loading pretrained weights from {Config.PRETRAINED_PATH}")
        if not os.path.exists(Config.PRETRAINED_PATH):
            logger.error(f"Pretrained weights file not found: {Config.PRETRAINED_PATH}")
            raise FileNotFoundError(f"Pretrained weights file not found: {Config.PRETRAINED_PATH}")
        vision_model.load_pretrained_weights(Config.PRETRAINED_PATH)

        # Initialize text model and tokenizer (BioGpt)
        logger.info("Initializing BioGPT model and tokenizer")
        text_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

        # Initialize combined vision-text model
        logger.info("Initializing VisionTextModel")
        model = VisionTextModel(
            vision_model=vision_model,
            text_model=text_model,
            vision_dim=Config.VISION_DIM,
            text_dim=Config.TEXT_DIM
        )

        # Define transforms
        logger.info("Defining transforms")
        train_transforms, eval_transforms = define_transforms()

        # Load datasets
        logger.info("Loading datasets")
        for split, path in Config.JSON_PATHS.items():
            if not os.path.exists(path):
                logger.error(f"JSON file not found: {path}")
                raise FileNotFoundError(f"JSON file not found: {path}")

        logger.info("Processing train dataset")
        train_paths, train_captions = load_and_process_json(Config.JSON_PATHS['train'])
        logger.info("Processing validation dataset")
        val_paths, val_captions = load_and_process_json(Config.JSON_PATHS['val'])
        logger.info("Processing test dataset")
        test_paths, test_captions = load_and_process_json(Config.JSON_PATHS['test'])
        logger.info(f"Loaded {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test samples")

        # Verify datasets are not empty
        if not train_paths or not val_paths or not test_paths:
            logger.error("One or more datasets are empty")
            raise ValueError("One or more datasets are empty")

        # Check sample image
        logger.info(f"Checking sample image: {train_paths[0]}")
        if not os.path.exists(train_paths[0]):
            logger.error(f"Sample image not found: {train_paths[0]}")
            raise FileNotFoundError(f"Sample image not found: {train_paths[0]}")

        # Initialize datasets
        logger.info("Initializing datasets")
        train_dataset = EyeNetDataset(train_paths, train_captions, train_transforms, tokenizer, Config.MAX_LENGTH)
        val_dataset = EyeNetDataset(val_paths, val_captions, eval_transforms, tokenizer, Config.MAX_LENGTH)
        test_dataset = EyeNetDataset(test_paths, test_captions, eval_transforms, tokenizer, Config.MAX_LENGTH)
        logger.info("Datasets initialized")

        # Create output directory
        logger.info(f"Creating output directory: {Config.OUTPUT_DIR}")
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Run training
        logger.info("Running train_model_test")
        model, tokenizer = train_model_test(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            output_dir=Config.OUTPUT_DIR,
            config=Config, 
            beam_search= Config.BEAM_SEARCH,
            early_stopping= Config.EARLY_STOPPING
        )
        logger.info("Training pipeline completed")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()