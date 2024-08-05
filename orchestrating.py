from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time
import struct_LLM
import struct_VIT
import logging
import time
start = time.time()
 

 


 




app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_response(response_data, status_code, status_message, start):
    output_response = {
        "response": response_data,
        "status": status_message,
        "statusCode": status_code,
        "timeTaken": round(time.time() - start, 3)
    }
    http_code = int(status_code)
    return output_response, http_code

@app.post("/api1")
async def api1(image: UploadFile = File(...), label: str = None):
    try:
        start = time.time()
        image_path = f"temp_{image.filename}"
        print("image loaded successfully")
        with open(image_path, "wb") as f:
            f.write(image.file.read())
        logger.info(f"Image saved to {image_path}")
        api1_response = {}

        try:
            predicted_label, img = struct_VIT.get_vit_prediction(image_path)
            image_base64 = struct_VIT.image_to_base64(image_path)
            logger.info(f"Prediction completed: {predicted_label}")

            api1_response = {
                "label": predicted_label,
                "image": image_base64
            }

            #print(api1_response)
            print(predicted_label)
            end = time.time()
            print("The time of execution of above program is :",(end-start) * 10**3, "ms")
            
            start2 = time.time()
            
            llm_response = struct_LLM.get_llm_response(api1_response)
            logger.info("LLM response received")

            print(llm_response)
            
            response_data = {
                "decision": llm_response["decision"],
                "justification": llm_response["justification"]
            }
            status_code = "200"
            status_message = "Success"
            output_response, http_code = generate_response(response_data, status_code, status_message, start)
            logger.info("API Execution completed")
            return JSONResponse(output_response, status_code=http_code)

        except Exception as e:
            logger.error(f"Error processing image with VIT model: {str(e)}")
            image_base64 = struct_VIT.image_to_base64(image_path)
            fallback_response = {
                "label": None,
                #"image": image_base64
            }
            llm_response = struct_LLM.get_llm_response(fallback_response, prompt_type=2)
            response_data = {
                "decision": llm_response["decision"],
                "justification": llm_response["justification"]
            }
            status_code = "500"
            status_message = "Fallback LLM response received"
            logger.info("Fallback LLM response received")
            output_response, http_code = generate_response(response_data, status_code, status_message, start)
            logger.info("API Execution completed")
            return JSONResponse(output_response, status_code=http_code)

    except Exception as e:
        logger.error(f"Failed to process the image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process the image: {str(e)}")
    end2 = time.time()
    print("The time of execution of above program is :",(end2-start2) * 10**3, "ms")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrating:app", host="0.0.0.0", port=8005, reload=True)

