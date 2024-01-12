from time import time
import argparse
import numpy as np
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser(description='Image Classification Script')

    # Required arguments
    parser.add_argument('imagepath', type=str, help='Path to a single image')
    parser.add_argument('save_path', type=str, help='file of the trained model')
    parser.add_argument('gpu', action='store_true', help = "to activate CUDA")
    # Optional arguments
    parser.add_argument('category_names', type=str, help='Mapping of categories to real names')
    parser.add_argument('topk', type=int, default=5, help='Top K most likely classes.')
 
    return parser.parse_args()
   

args_in = get_args()



if args_in.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

    
def load_checkpoint(filepath):
    # Define a default architecture 
    default_architecture = 'densenet121'
    
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    
    # Determine the architecture 
    architecture = checkpoint.get('arch', default_architecture)

    # Load the corresponding pretrained model
    if architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}. Supported architectures are 'densenet121' and 'vgg13'.")

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Load the rest of the checkpoint
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


checkpoint_path = args_in.save_path
model = load_checkpoint(checkpoint_path)
model.to(device);


def process_image(image_path):
    # Define the transformation pipeline only once
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)

    img = transform(img)

    return img.numpy() 

import torch

def predict(image_path, model, topk=5):
    # Ensure device is defined
    device = torch.device('cuda' if torch.cuda.is_available() and args_in.gpu else 'cpu')

    image = process_image(image_path)

    image = torch.from_numpy(image).to(device, dtype=torch.float)
    image = image.unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass without gradient computation
    with torch.no_grad():
        output = model.forward(image)

    # Calculate probabilities and indices
    output_prob = torch.exp(output)
    probs, indeces = output_prob.topk(topk)

    # Convert to CPU and extract lists
    probs, indeces = probs.cpu().numpy().tolist()[0], indeces.cpu().numpy().tolist()[0]

    # Map indices to class labels
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indeces]

    return probs, classes

imagepath, topk = args_in.imagepath, args_in.topk

probs, classes = predict(imagepath, model, topk)
category_names_path = args_in.category_names
if category_names_path:
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f, strict=False)
    class_names = [cat_to_name[key] for key in classes]
    print("Class names:" , class_names)

print(f"Class number: {classes}")

print("Probability:")

prob = [round(item * 100, 2) for item in probs]

print(prob)