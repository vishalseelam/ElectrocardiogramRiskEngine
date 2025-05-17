import os
import base64
import logging
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def image_to_base64(image_path):
    """
    Convert an image file to base64 encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string representation of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return encoded_image
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise

def base64_to_image(base64_string, output_path=None):
    """
    Convert a base64 string to an image file.
    
    Args:
        base64_string: Base64 encoded image string
        output_path: Path to save the decoded image (optional)
        
    Returns:
        If output_path is provided, saves the image and returns the path.
        Otherwise, returns a PIL Image object.
    """
    try:
        image_data = base64.b64decode(base64_string)
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(image_data)
            return output_path
        else:
            import io
            image = Image.open(io.BytesIO(image_data))
            return image
    except Exception as e:
        logger.error(f"Error converting base64 to image: {str(e)}")
        raise

def save_temp_file(file_content, prefix="temp_", suffix=".jpg"):
    """
    Save content to a temporary file.
    
    Args:
        file_content: Content to save
        prefix: Prefix for the temporary file
        suffix: Suffix for the temporary file
        
    Returns:
        Path to the temporary file
    """
    try:
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as temp_file:
            temp_file.write(file_content)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error saving temporary file: {str(e)}")
        raise

def clean_temp_files(directory=None, prefix="temp_"):
    """
    Clean up temporary files in a directory.
    
    Args:
        directory: Directory to clean (default: current directory)
        prefix: Prefix of files to clean up
    """
    try:
        target_dir = directory or os.getcwd()
        for filename in os.listdir(target_dir):
            if filename.startswith(prefix):
                file_path = os.path.join(target_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning temporary files: {str(e)}") 