#!/usr/bin/env python3
"""
Test script for the ECG Vision Transformer model.

This script demonstrates how to use the model to predict an ECG image classification.
"""

import os
import sys
import argparse
import logging

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vit_model import ECGVisionTransformer
from src.utils.helpers import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the ECG Vision Transformer model.
    """
    parser = argparse.ArgumentParser(description="Test the ECG Vision Transformer model.")
    parser.add_argument("--image", required=True, help="Path to the ECG image to classify.")
    parser.add_argument("--model", default="model.h5", help="Path to the model weights file.")
    parser.add_argument("--config", default="config.json", help="Path to the model configuration file.")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Check if config exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Initialize the model
    logger.info("Initializing the ECG Vision Transformer model...")
    model = ECGVisionTransformer(model_path=args.model, config_path=args.config)
    
    # Make prediction
    logger.info(f"Analyzing ECG image: {args.image}")
    predicted_label, img = model.predict(args.image)
    
    # Print the prediction
    logger.info(f"Prediction: {predicted_label}")
    print(f"\nECG Classification:\n------------------\n{predicted_label}\n")

if __name__ == "__main__":
    main() 