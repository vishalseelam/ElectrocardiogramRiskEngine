from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
import torch
from PIL import Image
import base64
import h5py

# Load VIT model and configurations
num_classes = 5
config = ViTConfig.from_pretrained('config.json')  # Adjust path as needed
config.num_labels = num_classes
model = ViTForImageClassification(config)

def load_model_from_h5(model, filename):
    with h5py.File(filename, 'r') as hf:
        state_dict = {key: torch.tensor(hf[key][()]) for key in hf.keys()}
    model.load_state_dict(state_dict)

# Load the model
load_model_from_h5(model, '/Users/vishalseelam/Computer Science/ECG/model.h5')

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
id_to_label = {0: 'Myocardial Infarction', 1: 'Abnormal Heartbeats', 2: 'Normal Heartbeats', 3: 'History of MI', 4: 'Covid_19'}

def get_vit_prediction(image_path):
    img = Image.open(image_path)
    inputs = feature_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    predicted_label = id_to_label[predicted_class]
    return predicted_label, img

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return encoded_image

