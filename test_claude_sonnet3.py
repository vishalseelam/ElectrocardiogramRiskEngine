import boto3
import json
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# Uncomment if you are using a session token
# AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# Claude Sonnet 3 Model ID (adjust if needed)
model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'

def test_claude_sonnet3():
    try:
        # Initialize a boto3 client for AWS Bedrock
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',  # Replace with your region if different
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            # aws_session_token=AWS_SESSION_TOKEN,  # Uncomment if you are using a session token
        )

        # Define a simple prompt
        system_prompt = "You are an AI assistant."
        user_prompt = "What is the capital of France?"

        # Prepare the request payload
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,  # Adjust token limit as needed
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.7
        })

        # Invoke the Claude Sonnet 3 model
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id
        )

        # Parse and print the response
        response_body = json.loads(response.get('body').read())
        print("Claude Sonnet 3 API Response:")
        print(json.dumps(response_body, indent=2))

    except NoCredentialsError:
        print("No AWS credentials found. Please check if your credentials are correctly set up.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials. Please ensure both access key ID and secret access key are set.")
    except ClientError as e:
        print(f"An error occurred with AWS services: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_claude_sonnet3()
