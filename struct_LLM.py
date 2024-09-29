import os
import boto3
import json
import logging
from botocore.exceptions import ClientError
from botocore.config import Config
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

config = Config(read_timeout=1000)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'

def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        "temperature": 0.7
    })
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
    return response_body

def prompt(image_base64, predicted_label):
    prompt_message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64
                }
            },
            {
                "type": "text",
                "text": f"""
                    INPUT:
                    The label from a VIT model and an Image.

                    Label: The provided ECG is classified as {predicted_label}.
                    your task is to justify that why it is classified as {predicted_label}

                    OUTPUT EXAMPLE FORMAT (very Important):
                    CLASS LABEL:
                    {{
                      decision: 
                      Justification: 
                      Remarks:
                    }}
                    The above example is overall decision, justification, and Remarks.
                """
            }
        ]
    }
    return prompt_message

def get_llm_response(api1_response, prompt_type=1):
    image_base64 = api1_response["image"]
    label = api1_response.get("label", "")

    if prompt_type == 1:
        prompt_message = prompt(image_base64, label)
    else:
        prompt_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": "Analyze the ECG image and provide a decision and justification."
                }
            ]
        }

    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        # aws_session_token=AWS_SESSION_TOKEN,
        config=config
    )
    model_id = modelId
    system_prompt = "You are a Cardiologist and your task is to analyze an ECG image and provide a detailed report."
    max_tokens = 4096

    try:
        response = generate_message(bedrock_runtime, model_id, system_prompt, [prompt_message], max_tokens)
        logger.info(f"LLM API Response: {response}")

        # Log the entire response for debugging
        logger.info(f"Full LLM Response: {json.dumps(response, indent=2)}")

        # Check if the response contains the expected fields
        if 'content' in response and len(response['content']) > 0:
        # if len(response['content']) > 0:

            response_text = response['content'][0]['text']

            print(response_text)
            logger.info(f"Response Text: {response_text}")

            if 'decision:' in response_text and 'Justification:' in response_text:
                # Extract decision and justification from the response text
                decision_start = response_text.find('decision:') + len('decision:')
                justification_start = response_text.find('Justification:') + len('Justification:')
                decision_end = response_text.find('Justification:')
                decision = response_text[decision_start:justification_start].strip()
                justification = response_text[justification_start:].strip()

                return {
                    "decision": api1_response["label"],
                    "justification": justification
                }
            else:
                logger.error("Response text does not contain 'decision:' or 'Justification:' fields.")
                raise ValueError("Unexpected response format: Missing 'decision:' or 'Justification:' fields.")
        else:
            logger.error(f"Unexpected LLM API response format: {response}")
            raise ValueError("Unexpected LLM API response format")

    except ClientError as e:
        logger.error(f"A client error occurred: {str(e)}")
        raise
