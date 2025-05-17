import os
import boto3
import json
import logging
from botocore.exceptions import ClientError
from botocore.config import Config
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

class ECGLLMAnalyzer:
    def __init__(self):
        """
        Initialize the LLM analyzer for ECG interpretations using Anthropic Claude on AWS Bedrock.
        """
        # Load environment variables
        load_dotenv()
        
        # Load AWS credentials
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.getenv("AWS_SESSION_TOKEN", None)
        
        # Configure the AWS session
        self.config = Config(read_timeout=1000)
        self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        
        # Initialize the Bedrock client
        self._init_bedrock_client()
    
    def _init_bedrock_client(self):
        """
        Initialize the AWS Bedrock client.
        """
        client_params = {
            'service_name': 'bedrock-runtime',
            'region_name': 'us-east-1',
            'aws_access_key_id': self.aws_access_key_id,
            'aws_secret_access_key': self.aws_secret_access_key,
            'config': self.config
        }
        
        # Add session token if available
        if self.aws_session_token:
            client_params['aws_session_token'] = self.aws_session_token
            
        self.bedrock_runtime = boto3.client(**client_params)
    
    def _generate_message(self, system_prompt, messages, max_tokens=4096):
        """
        Generate a message using the Anthropic Claude model.
        
        Args:
            system_prompt: The system prompt to set context for the model
            messages: List of message objects to send to the model
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Model response
        """
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": 0.7
        })
        
        response = self.bedrock_runtime.invoke_model(body=body, modelId=self.model_id)
        response_body = json.loads(response.get('body').read())
        return response_body
    
    def _create_prompt(self, image_base64, predicted_label=None):
        """
        Create a prompt message for the LLM based on the image and optional prediction.
        
        Args:
            image_base64: Base64 encoded image
            predicted_label: Optional predicted label from the ViT model
            
        Returns:
            Formatted prompt message
        """
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64
                }
            }
        ]
        
        if predicted_label:
            content.append({
                "type": "text",
                "text": f"""
                    INPUT:
                    The label from a VIT model and an Image.

                    Label: The provided ECG is classified as {predicted_label}.
                    Your task is to justify that why it is classified as {predicted_label}

                    OUTPUT EXAMPLE FORMAT (very Important):
                    CLASS LABEL:
                    {{
                      decision: 
                      Justification: 
                      Remarks:
                    }}
                    The above example is overall decision, justification, and Remarks.
                """
            })
        else:
            content.append({
                "type": "text",
                "text": "Analyze the ECG image and provide a decision and justification."
            })
        
        return {"role": "user", "content": content}
    
    def get_analysis(self, image_base64, predicted_label=None):
        """
        Get an analysis of an ECG image from the LLM.
        
        Args:
            image_base64: Base64 encoded image
            predicted_label: Optional label from the ViT model
            
        Returns:
            Dictionary containing the decision and justification
        """
        try:
            # Create the prompt message
            prompt_message = self._create_prompt(image_base64, predicted_label)
            
            # Set system prompt
            system_prompt = "You are a Cardiologist and your task is to analyze an ECG image and provide a detailed report."
            
            # Generate response from model
            response = self._generate_message(system_prompt, [prompt_message], max_tokens=4096)
            logger.info(f"LLM API Response received")
            
            # Extract text from response
            if 'content' in response and len(response['content']) > 0:
                response_text = response['content'][0]['text']
                
                # Parse the response to extract decision and justification
                if 'decision:' in response_text and 'Justification:' in response_text:
                    decision_start = response_text.find('decision:') + len('decision:')
                    justification_start = response_text.find('Justification:') + len('Justification:')
                    decision_end = response_text.find('Justification:')
                    
                    decision = response_text[decision_start:decision_end].strip()
                    justification = response_text[justification_start:].strip()
                    
                    return {
                        "decision": predicted_label or decision,
                        "justification": justification
                    }
                else:
                    logger.warning("Response format does not contain expected fields")
                    return {
                        "decision": predicted_label or "Unknown",
                        "justification": response_text
                    }
            else:
                logger.error(f"Unexpected LLM API response format")
                raise ValueError("Unexpected LLM API response format")
                
        except ClientError as e:
            logger.error(f"AWS client error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            raise 