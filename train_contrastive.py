"""
Training and evaluation functions for the vision-text model.
Implements training loop, validation, and testing with comprehensive metrics.
"""

import torch
import os
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import corpus_bleu
from evaluate import load
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model_test(model, tokenizer, train_dataset, val_dataset, test_dataset, output_dir, config, beam_search = 1, early_stopping = False):
    """
    Test training pipeline for a single epoch with enhanced logging and evaluation.
    
    Args:
        model: VisionTextModel instance.
        tokenizer: BioGpt tokenizer.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        output_dir (str): Directory to save outputs.
        config: Configuration object with hyperparameters.
        beam_search: Number of beam search decoding. **********
        early_stopping: **********
    
    Returns:
        tuple: Trained model and tokenizer.
    """
    try:
        logger.info("Starting train_model_test")
        device = config.DEVICE
        logger.info(f"Using device: {device}")
        model.to(device)

        logger.info("Initializing optimizer")
        optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

        logger.info("Creating DataLoaders")
        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        logger.info(f"DataLoaders created: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")

        logger.info("Initializing scheduler")
        total_steps = len(train_dataloader) // config.ACCUM_STEPS
        warmup_steps = int(config.WARMUP_RATIO * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        logger.info("Loading evaluation metrics")
        rouge = load("rouge")
        meteor = load("meteor")
        bertscore = load("bertscore")
        logger.info("Evaluation metrics loaded")

        logger.info("Starting training loop")
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Testing Epoch")):
            try:
                logger.info(f"Batch {batch_idx} loaded from DataLoader")
                
                logger.info(f"Moving batch {batch_idx} to device")
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logger.info(f"Batch {batch_idx} moved to device")

                logger.info(f"Performing forward pass for batch {batch_idx}")
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                logger.info(f"Forward pass completed for batch {batch_idx}")

                lm_loss = outputs['lm_loss']
                cont_loss = outputs['cont_loss']
                loss = outputs['total_loss'] / config.ACCUM_STEPS
                logger.info(f"Batch {batch_idx} - LM Loss: {lm_loss.item():.4f}, Contrastive Loss: {cont_loss.item():.4f}, Total Loss: {loss.item() * config.ACCUM_STEPS:.4f}")
                logger.info(f"Performing backward pass for batch {batch_idx}")
                
                loss.backward()
                logger.info(f"Backward pass completed for batch {batch_idx}")

                if (batch_idx + 1) % config.ACCUM_STEPS == 0:
                    logger.info(f"Performing optimizer step for batch {batch_idx}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    logger.info(f"Optimizer step completed for batch {batch_idx}")

                train_loss += loss.item() * config.ACCUM_STEPS
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                raise

        train_loss /= len(train_dataloader)
        logger.info(f"Training completed, train_loss: {train_loss:.4f}")

        # Validation
        logger.info("Starting validation loop")
        model.eval()
        val_loss = 0
        generated_texts = []
        reference_texts = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validating")):
                logger.info(f"Processing validation batch {batch_idx}")
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                val_loss += outputs['total_loss'].item()
                logger.info(f"Validation batch {batch_idx} - LM Loss: {outputs['lm_loss'].item():.4f}, Contrastive Loss: {outputs['cont_loss'].item():.4f}")

                logger.info(f"Generating captions for validation batch {batch_idx}")
                vision_features = model.vision_model(pixel_values)
                batch_size = pixel_values.size(0)
                vision_embed = vision_features.reshape(batch_size, 1, vision_features.size(-1))  # [batch, 1, 1024]
                generated_ids = model.text_model.generate(
                    inputs_embeds=vision_embed,
                    attention_mask=torch.ones(batch_size, 1, device=device),
                    max_length=config.MAX_LENGTH, 
                    num_beams = beam_search, 
                    early_stopping = early_stopping
                )
                gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                ref_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                generated_texts.extend(gen_texts)
                reference_texts.extend(ref_texts)

        val_loss /= len(val_dataloader)
        logger.info(f"Validation completed, val_loss: {val_loss:.4f}")

        # Compute metrics
        logger.info("Computing metrics")
        bleu_score = corpus_bleu([[ref.split()] for ref in reference_texts], [gen.split() for gen in generated_texts])
        rouge_scores = rouge.compute(predictions=generated_texts, references=reference_texts)
        meteor_score = meteor.compute(predictions=generated_texts, references=reference_texts)
        bertscore_results = bertscore.compute(predictions=generated_texts, references=reference_texts, lang="en")

        # Log metrics
        logger.info(f"Test Epoch - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"BLEU: {bleu_score:.4f}, ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}, "
                    f"METEOR: {meteor_score['meteor']:.4f}, BERTScore: {sum(bertscore_results['f1'])/len(bertscore_results['f1']):.4f}")

        # Save sample captions for inspection
        logger.info("Saving sample captions")
        with open(os.path.join(output_dir, "sample_captions.txt"), "w") as f:
            for gen, ref in zip(generated_texts[:5], reference_texts[:5]):
                f.write(f"Generated: {gen}\nReference: {ref}\n\n")

        # Clinical relevance
        logger.info("Computing clinical relevance")
        clinical_terms = ["diabetic retinopathy", "microaneurysms", "hard exudates", "right eye", "left eye"]
        clinical_matches = sum(1 for gen in generated_texts if any(term in gen.lower() for term in clinical_terms))
        logger.info(f"Clinical relevance: {clinical_matches}/{len(generated_texts)} captions contain clinical terms")

        logger.info("train_model_test completed")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error in train_model_test: {str(e)}")
        raise