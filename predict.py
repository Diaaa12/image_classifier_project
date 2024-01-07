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
    parser.add_argument('imagepath', type=str, help='Path to a single image (required)')
    parser.add_argument('save_path', type=str, help='Path to the file of the trained model (required)')

    # Optional arguments
    parser.add_argument('--category_names', type=str, help='Mapping of categories to real names')
    parser.add_argument('--topk', type=int, default=5, help='Top K most likely classes. Default value is 5')

    return parser.parse_args()

args_in = get_args()


use_gpu = True

if use_gpu:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if args_in.category_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[key] for key in classes]
    print("Class names:" , class_names)

print(f"Class number: {classes}")

print("Probability:")

prob = [round(item * 100, 2) for item in probs]

print(prob)


from time import time
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use Agg as a non-interactive backend
import matplotlib.pyplot as plt
import json

# ... (rest of the code remains the same)


# Rest of the code...



def timing_function(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_function
def predict_and_plot_topk(model: nn.Module,
                          img_transform: transforms.Compose,
                          class_list: list,
                          image_path: str,
                          device: torch.device,
                          topk: int = 5):

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = img_transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Moving model to device and switching to eval mode
    model.to(device)
    model.eval()

    # Make predictions
    with torch.no_grad():
        output = model(input_batch)

    # Convert the output to probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top-k class indices and probabilities
    topk_probs, topk_indices = torch.topk(probabilities, topk)
    topk_probs_np = topk_probs.cpu().numpy()
    topk_indices_np = topk_indices.cpu().numpy() - 1  # Subtract 1 to make indices zero-based

    # Debugging print statements
    print("Top-k indices:", topk_indices_np)
    print("Class list length:", len(class_list))

    # Ensure that the indices are within bounds
    topk_indices_np = np.clip(topk_indices_np, 0, len(class_list) - 1)

    # Convert tensor to numpy array for plotting
    probs_np = probabilities.cpu().numpy()

    # Create a horizontal bar graph
    plt.figure(figsize=(10, 6))

    # Plot the image
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    # Plot the top-k classes
    plt.subplot(2, 1, 2)

    # Debugging print statements
    print([class_list[i] for i in topk_indices_np])
    print(topk_probs_np)

    plt.barh([class_list[i] for i in topk_indices_np], topk_probs_np, color='blue')
    plt.xlabel('Predicted Probability')
    plt.title(f'Top-{topk} Predicted Classes')

    plt.tight_layout()
    plt.savefig('output_fig.png')


# Example usage
# Assuming you have a PyTorch model 'model', a list of class names 'dataset.classes', and an image file path 'imagepath'
predict_and_plot_topk(model,
                      transforms.Compose([
                          transforms.Resize(255),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ]),
                      classes,
                      args_in.imagepath,
                      device,
                      topk=args_in.topk)

 
