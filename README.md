# Medical Image Captioning with MedViT and BioGpt

This repository contains the implementation of a multimodal medical image captioning model combining MedViT (a vision transformer for medical imaging) and BioGpt (a biomedical language model). The model is designed to generate clinically relevant captions for retinal images from the DeepEyeNet dataset, supporting ophthalmology diagnostics.

Note: This project is a work in progress. I am actively developing and experimenting with new features to enhance performance and clinical utility.

## Project Overview
The goal is to develop an automated system for generating accurate and clinically meaningful captions for retinal images, aiding in the diagnosis of conditions like diabetic retinopathy. The model integrates:

  MedViT: Extracts visual features from retinal images, leveraging pretrained weights from ImageNet.
  BioGpt: Generates natural language captions, fine-tuned for biomedical text.
  DeepEyeNet Dataset: A specialized dataset of retinal images with clinical descriptions and keywords.


Current Features

  Beam Search Decoding: Generates coherent captions using beam search (configurable: greedy or beam size 3/5).
  Contrastive Loss: Aligns vision and text embeddings for better multimodal understanding.
  Visual and Semantic Topic Attention: Emphasizes clinically relevant tokens in captions.
  LoRA (Low-Rank Adaptation): Enables parameter-efficient fine-tuning, reducing memory usage on resource-constrained hardware.
  Comprehensive Evaluation: Metrics include BLEU, ROUGE, METEOR, BERTScore, and clinical relevance (based on medical term presence).

## Future Work
I am exploring the following procedures to enhance the model:

  Reinforcement Learning for Caption Generation:
    Apply reinforcement learning (e.g., REINFORCE or Proximal Policy Optimization) to optimize caption generation.
    Use reward functions based on clinical relevance (e.g., presence of medical terms like "diabetic retinopathy") and linguistic coherence to improve caption quality.

  Perceiver Architecture:
    Integrate the Perceiver model to efficiently process multimodal inputs (retinal images and text) using a compact latent space.
    Aims to reduce computational complexity and enhance scalability for large-scale medical datasets.

## Project Structure

main.py: Orchestrates training pipeline and configuration.
models.py: Defines MedViT and VisionTextModel (with contrastive loss and topic attention).
dataset.py: Handles DeepEyeNet dataset loading and preprocessing.
train.py: Implements training, validation, and evaluation loops.

Dataset used: DeepEyeNet https://github.com/Jhhuangkay/DeepOpht-Medical-Report-Generation-for-Retinal-Images-via-Deep-Models-and-Visual-Explanation
Downloading MedViT pre-trained weights on ImageNet1k: https://drive.google.com/file/d/1Lrfzjf3CK7YOztKa8D6lTUZjYJIiT7_s/view?usp=sharing

Last updated: May 9, 2025
