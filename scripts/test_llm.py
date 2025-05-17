#!/usr/bin/env python3
"""
Test script for the ECG LLM Analyzer.

This script demonstrates how to use the LLM analyzer to provide detailed justification for an ECG classification.
"""

import os
import sys
import argparse
import logging

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vit_model import ECGVisionTransformer
from src.models.llm_model import ECGLLMAnalyzer
from src.utils.helpers import setup_logging, image_to_base64

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the ECG LLM Analyzer.
    """
    parser = argparse.ArgumentParser(description="Test the ECG LLM Analyzer.")
    parser.add_argument("--image", required=True, help="Path to the ECG image to analyze.")
    parser.add_argument("--model", default="model.h5", help="Path to the model weights file.")
    parser.add_argument("--config", default="config.json", help="Path to the model configuration file.")
    parser.add_argument("--label", help="Optional label to override ViT prediction.")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)
    
    # Load the image and convert to base64
    image_base64 = image_to_base64(args.image)
    
    # If a label is provided, use it; otherwise, use the ViT model to get a prediction
    label = args.label
    if not label:
        # Check if model exists
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            sys.exit(1)
        
        # Check if config exists
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        
        # Initialize the ViT model
        logger.info("Initializing the ECG Vision Transformer model...")
        vit_model = ECGVisionTransformer(model_path=args.model, config_path=args.config)
        
        # Make prediction
        logger.info(f"Getting ViT prediction for: {args.image}")
        label, _ = vit_model.predict(args.image)
    
    # Initialize the LLM analyzer
    logger.info("Initializing the ECG LLM Analyzer...")
    llm_analyzer = ECGLLMAnalyzer()
    
    # Get LLM analysis
    logger.info("Getting LLM analysis...")
    llm_response = llm_analyzer.get_analysis(image_base64, label)
    
    # Print the results
    print("\nECG Analysis Results")
    print("===================")
    print(f"Decision: {llm_response['decision']}")
    print("\nJustification:")
    print("--------------")
    print(llm_response['justification'])

if __name__ == "__main__":
    main() 