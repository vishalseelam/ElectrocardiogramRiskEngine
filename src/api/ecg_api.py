from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
import os
import sys

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_model import ECGVisionTransformer
from models.llm_model import ECGLLMAnalyzer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize the FastAPI app
app = FastAPI(
    title="ECG Risk Engine API",
    description="API for analyzing ECG images using Vision Transformer and LLM technologies",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
vit_model = None
llm_analyzer = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize models when the API starts up.
    """
    global vit_model, llm_analyzer
    try:
        # Initialize the Vision Transformer model
        vit_model = ECGVisionTransformer()
        logger.info("Vision Transformer model initialized successfully")
        
        # Initialize the LLM model
        llm_analyzer = ECGLLMAnalyzer()
        logger.info("LLM Analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise RuntimeError(f"Failed to initialize models: {str(e)}")

def generate_response(response_data, status_code, status_message, start):
    """
    Generate a standardized API response.
    
    Args:
        response_data: The data to include in the response
        status_code: HTTP status code
        status_message: Status message
        start: Start time for measuring execution time
        
    Returns:
        Standardized response dictionary and HTTP code
    """
    output_response = {
        "response": response_data,
        "status": status_message,
        "statusCode": status_code,
        "timeTaken": round(time.time() - start, 3)
    }
    http_code = int(status_code)
    return output_response, http_code

@app.post("/api/analyze")
async def analyze_ecg(image: UploadFile = File(...)):
    """
    Analyze an ECG image using the Vision Transformer model and LLM.
    
    Args:
        image: The uploaded ECG image file
        
    Returns:
        Analysis results including decision and justification
    """
    try:
        logger.info(f"Received file: {image.filename}")
        start = time.time()
        
        # Save the uploaded image temporarily
        image_path = f"temp_{image.filename}"
        with open(image_path, "wb") as f:
            f.write(image.file.read())
        logger.info(f"Image saved to: {image_path}")
        
        try:
            # Get prediction from ViT model
            predicted_label, img = vit_model.predict(image_path)
            image_base64 = vit_model.image_to_base64(image_path)
            logger.info(f"Prediction completed: {predicted_label}")
            
            # Get LLM justification
            llm_response = llm_analyzer.get_analysis(image_base64, predicted_label)
            logger.info("LLM response received")
            
            # Prepare response
            response_data = {
                "decision": llm_response["decision"],
                "justification": llm_response["justification"]
            }
            
            status_code = "200"
            status_message = "Success"
            output_response, http_code = generate_response(
                response_data, status_code, status_message, start
            )
            logger.info("API Execution completed")
            
            # Clean up the temporary file
            os.remove(image_path)
            
            return JSONResponse(output_response, status_code=http_code)
            
        except Exception as e:
            logger.error(f"Error processing image with ViT model: {str(e)}")
            
            # Fallback to LLM-only analysis
            image_base64 = ECGVisionTransformer.image_to_base64(image_path)
            
            # Get LLM analysis without prediction
            llm_response = llm_analyzer.get_analysis(image_base64)
            
            response_data = {
                "decision": llm_response.get("decision", "Unknown"),
                "justification": llm_response.get("justification", "Unable to provide justification")
            }
            
            status_code = "500"
            status_message = "Fallback LLM response"
            logger.info("Fallback LLM response received")
            
            output_response, http_code = generate_response(
                response_data, status_code, status_message, start
            )
            
            # Clean up the temporary file
            os.remove(image_path)
            
            return JSONResponse(output_response, status_code=http_code)
            
    except Exception as e:
        logger.error(f"Failed to process the image: {str(e)}")
        
        # Try to clean up temporary file if it exists
        try:
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
        except:
            pass
            
        raise HTTPException(status_code=500, detail=f"Failed to process the image: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        Health status of the API
    """
    return {"status": "healthy", "models": {"vit": vit_model is not None, "llm": llm_analyzer is not None}}

# Run the API server if this module is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ecg_api:app", host="0.0.0.0", port=8005, reload=True) 