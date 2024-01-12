from time import time
import numpy as np
import torch
from torch import nn , optim
from torchvision import datasets,  models
import json
import argparse
from collections import OrderedDict
import torchvision.transforms as transforms



parser = argparse.ArgumentParser()

parser.add_argument('arch', type = str,
                    help = 'densenet121 or vgg13')
parser.add_argument('data_dir', type = str,
                    help = 'data directory')
parser.add_argument('save_dir', type = str,
                    help = 'directory to save model')
parser.add_argument('epoch', type = int, 
                    help = 'Number of epochs')
parser.add_argument('learning_rate', type = float, default = 0.001,
                    help = 'Learning rate')
parser.add_argument('hidden_units', type = int, default = 512,
                    help = 'Number of hidden units')
parser.add_argument('gpu', action='store_true',
                    help = "to activate CUDA")

args_in = parser.parse_args()

if args_in.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


data_dir  = args_in.data_dir
train_dir = data_dir+ '/train'
test_dir  = data_dir+ '/test'
valid_dir = data_dir+ '/valid'

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
device = torch.device('cuda' if torch.cuda.is_available() and args_in.gpu else 'cpu')

epochs = args_in.epoch
layers        = args_in.hidden_units
learning_rate = args_in.learning_rate


#can choose architecture
import torch.nn as nn
from collections import OrderedDict
arch = args_in.arch
# Dictionary mapping architecture names to their corresponding models
architectures = {
    'densenet121': models.densenet121(pretrained=True),
    'vgg13': models.vgg13(pretrained=True)
}

if arch not in architectures:
    raise ValueError('Model arch error.')

model = architectures[arch]

for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier.in_features if arch == 'densenet121' else model.classifier[0].in_features

model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(in_features, layers)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.2)),
    ('fc2', nn.Linear(layers, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device);

print("model arch: " + args_in.arch)
print("model building finished ")

import time
print("Training the model")

steps = 0
runningloss = 0
print_every = 10

model.to(device)

for epoch in range(epochs):
    start_time = time.time()

    model.train()  # Set the model to training mode
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model(inputs)
        loss_value = criterion(logps, labels)
        loss_value.backward()
        optimizer.step()
        runningloss += loss_value.item()

        if steps % print_every == 0:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                test_loss, accuracy = 0, 0
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    batchloss = criterion(logps, labels)
                    test_loss += batchloss.item()

                    ps = torch.exp(logps)
                    
                    top_p, top_class = ps.topk(1, dim=1)
                    
                    equals = top_class == labels.view(*top_class.shape)
                    
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            avg_train_loss = runningloss / print_every
            avg_valid_loss = test_loss / len(valid_loader)
            avg_valid_accuracy = accuracy / len(valid_loader)

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {avg_train_loss:.3f}.. "
                  f"Validation loss: {avg_valid_loss:.3f}.. "
                  f"Validation accuracy: {avg_valid_accuracy:.3f}")
            
            running_loss = 0

    elapsed_time = time.time() - start_time
    print(f"Elapsed Runtime for epoch {epoch + 1}: {elapsed_time:.2f}s.")

print("Model training finished")


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

save_dir = args_in.save_dir

save_path = os.path.join(save_dir, 'checkpoint.pth')

# Check if the directory exists, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


torch.save(checkpoint, save_path)

print("Model saved to {}".format(save_path))