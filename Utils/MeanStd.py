import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# get image's mean & std
def calculate_mean_std(dataset_folder):
    # transform...
    transform = transforms.Compose([
        transforms.ToTensor(),  # 0~1
        transforms.Resize((224,224)), # input size in model
    ])
    
    # init
    mean = 0.
    std = 0.
    total_images = 0

    # get dataloader
    train_DS = datasets.ImageFolder(root=dataset_folder, transform=transform) 
    train_DL = torch.utils.data.DataLoader(train_DS, batch_size=64, shuffle=False)
    train_loader = DataLoader(train_DS, batch_size=64, shuffle=False)

    # loop image
    for images, _ in train_loader:
        batch_size = images.size(0) 
        images = images.view(batch_size, images.size(1), -1) # convert 4ch to 3ch... (batch_size, channels, height, width) ==> (batch_size, channels, height * width) 
        mean += images.mean(2).sum(0) # sum of Channel 3(height * width)'s mean
        std += images.std(2).sum(0) # sum of Channel 3(height * width)'s std

    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)
    return mean, std

# set dataset path
dataset_folder = f"/mnt/e/Mask_12K"

# calculate
mean, std = calculate_mean_std(dataset_folder)
print(f'Mean: {mean}')
print(f'STD: {std}')