#!/usr/bin/env python3
"""
API client script for the ECG Risk Engine.

This script demonstrates how to call the ECG Risk Engine API.
"""

import os
import sys
import argparse
import requests
import json
import logging
from pprint import pprint

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """
    Main function to demonstrate how to call the ECG Risk Engine API.
    """
    parser = argparse.ArgumentParser(description="Call the ECG Risk Engine API.")
    parser.add_argument("--image", required=True, help="Path to the ECG image to analyze.")
    parser.add_argument("--host", default="localhost", help="API host address.")
    parser.add_argument("--port", default="8005", help="API port.")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)
    
    # Build API URL
    api_url = f"http://{args.host}:{args.port}/api/analyze"
    
    # Prepare the file to upload
    files = {
        'image': (os.path.basename(args.image), open(args.image, 'rb'), 'image/jpeg')
    }
    
    # Make the API request
    logger.info(f"Sending request to: {api_url}")
    try:
        response = requests.post(api_url, files=files)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Parse the response
        result = response.json()
        
        # Print the results
        print("\nECG Analysis Results")
        print("===================")
        print(f"Decision: {result['response']['decision']}")
        print("\nJustification:")
        print("--------------")
        print(result['response']['justification'])
        print("\nAPI Response Details:")
        print("-------------------")
        print(f"Status: {result['status']}")
        print(f"Status Code: {result['statusCode']}")
        print(f"Time Taken: {result['timeTaken']} seconds")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        sys.exit(1)

if __name__ == "__main__":
    main() 