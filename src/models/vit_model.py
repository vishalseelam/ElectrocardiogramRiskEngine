from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
import torch
from PIL import Image
import base64
import h5py
import os

class ECGVisionTransformer:
    def __init__(self, model_path='model.h5', config_path='config.json'):
        """
        Initialize the Vision Transformer model for ECG classification.
        
        Args:
            model_path: Path to the model weights file
            config_path: Path to the model configuration file
        """
        self.num_classes = 5
        self.config = ViTConfig.from_pretrained(config_path)
        self.config.num_labels = self.num_classes
        self.model = ViTForImageClassification(self.config)
        
        # Load the model weights
        self.load_model_from_h5(self.model, model_path)
        
        # Initialize the feature extractor
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        # Define the mapping from class index to label
        self.id_to_label = {
            0: 'Myocardial Infarction', 
            1: 'Abnormal Heartbeats', 
            2: 'Normal Heartbeats', 
            3: 'History of MI', 
            4: 'Covid_19'
        }

    def load_model_from_h5(self, model, filename):
        """
        Load model weights from an H5 file.
        
        Args:
            model: The model to load weights into
            filename: Path to the H5 file containing weights
        """
        with h5py.File(filename, 'r') as hf:
            state_dict = {key: torch.tensor(hf[key][()]) for key in hf.keys()}
        model.load_state_dict(state_dict)

    def predict(self, image_path):
        """
        Make a prediction on an ECG image.
        
        Args:
            image_path: Path to the input ECG image
            
        Returns:
            predicted_label: The predicted label for the ECG
            img: The processed image object
        """
        img = Image.open(image_path)
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
        predicted_label = self.id_to_label[predicted_class]
        return predicted_label, img

    @staticmethod
    def image_to_base64(image_path):
        """
        Convert an image to base64 encoding.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Base64 encoded string representation of the image
        """
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return encoded_image 