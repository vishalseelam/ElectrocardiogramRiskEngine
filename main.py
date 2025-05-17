#!/usr/bin/env python3
"""
Main entry point for the Electrocardiogram Risk Engine.

This script runs the FastAPI server to provide ECG analysis services.
"""

import os
import uvicorn
from src.utils.helpers import setup_logging, clean_temp_files
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the ECG Risk Engine API.
    """
    logger.info("Starting ECG Risk Engine API...")
    
    # Clean any temporary files from previous runs
    clean_temp_files()
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8005))
    
    # Start the FastAPI server
    uvicorn.run(
        "src.api.ecg_api:app", 
        host="0.0.0.0", 
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main() 