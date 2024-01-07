from time import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets,  models
import json
import argparse
from collections import OrderedDict
import torchvision.transforms as transforms


data_dir ='./flowers' 
save_dir = "./"       
arch = "densenet121"  
learning_rate = 0.001
hidden_units = 512
epochs = 1
use_gpu = True
layers = hidden_units

print("data is being loaded")

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir  = data_dir + '/test'

normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])

common_valid_test_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=common_valid_test_transform)
test_data = datasets.ImageFolder(test_dir, transform=common_valid_test_transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print("data loaded")

print("model is being built")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=True)
model = model.to(device)
import torch.nn.functional as F

# Define a dictionary mapping architecture names to model and input size
architectures = {
    'densenet121': (models.densenet121, 1024),
    'vgg13': (models.vgg13, 25088),
   
}

# Check if the specified architecture is in the dictionary
if arch in architectures:
    # Get the model and input size from the dictionary
    model_fn, input_size = architectures[arch]
    
    # Create the model
    model = model_fn(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier
    model.classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(input_size, layers)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(layers, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ])
    )
else:
    raise ValueError('Model arch error.')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device);

print("model arch: " + arch)
print("model building finished ")
print("training the model")

steps  = 0
runningloss = 0
print_every  = 10

model.to(device)

for epoch in range(epochs):
    t1 = time()
    model.train()

    for step, (inputs, labels) in enumerate(train_loader):
        steps = epoch * len(train_loader) + step
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()

        runningloss += loss.item()

        # Accumulate gradients and update every print_every steps
        if steps % print_every == 0 or step == len(train_loader) - 1:
            optimizer.step()
            model.zero_grad()

        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                testloss = 0
                accuracy = 0

                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batchloss = criterion(logps, labels)
                    
                    testloss += batchloss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {runningloss / print_every:.3f}.. "
                  f"Validation loss: {testloss / len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy / len(valid_loader):.3f}")

    t2 = time()
    print("Elapsed Runtime for epoch {}: {}s.".format(epoch + 1, t2 - t1))

print("model training finished")

print("model testing started")
model.to(device)
model.eval()

accuracy = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Use model.eval() to set the model to evaluation mode
        model.eval()
        
        # Forward pass
        logps = model(inputs)
        
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

# Calculate and print test accuracy
test_accuracy = accuracy / len(test_loader)
print(f"Test accuracy: {test_accuracy:.3f}")


model.train();

print("model testing finished")

model.class_to_idx = train_data.class_to_idx
checkpoint = {'class_to_idx': model.class_to_idx,
              'model_state_dict': model.state_dict(),
              'classifier': model.classifier,
              'arch': arch}

import os

save_dir = "./"
save_path = os.path.join(save_dir, 'checkpoint.pth')

# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


torch.save(checkpoint, save_path)

print("Model saved to {}".format(save_path))