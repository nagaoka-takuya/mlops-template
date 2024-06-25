import json
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import os

# Model load
def model_fn(model_dir):
    print("Loading model from directory: ", model_dir)
    model = models.resnet18(pretrained=False)
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Input data processing
def input_fn(request_body, request_content_type):
    print("Processing input data")
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        image_url = input_data['url']
        
        # Read image from URL
        image = cv2.imread(image_url, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from url: {image_url}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image = preprocess(image)
        image = image.unsqueeze(0)  # Create a mini-batch as expected by the model
        return image
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

# Predict function
def predict_fn(input_data, model):
    print("Performing prediction")
    with torch.no_grad():
        output = model(input_data)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Output function
def output_fn(prediction, response_content_type):
    print("Processing output data")
    return json.dumps({'predicted_label': prediction})