from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import logging
import json
import time

import h5py
import numpy as np

from annoy import AnnoyIndex
from argparse import ArgumentParser
from PIL import Image
import torchvision.transforms.functional as TF


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model_ft.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    model_ft = models.resnet101(pretrained=True)
    set_parameter_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 19)
    
    input_size = 224
    # Print the model we just instantiated
    PATH='models/resnet_101.pth'
    model_ft.load_state_dict(torch.load(PATH))
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Send the model to GPU
    model_ft = model_ft.to(device)
    list_folder=next(os.walk("dataset/single_img"))[1]

    for each_folder in list_folder:
        folder_path=os.path.join("dataset/single_img", each_folder)
        list_img = next(os.walk(folder_path))[2]
        for each_img_name in list_img:
            img_path = os.path.join(folder_path, each_img_name)
            image = Image.open(img_path)
            image = image.resize((224, 224)) 
            x = TF.to_tensor(image)
            x.unsqueeze_(0)
            x = x.to(device)
            result = model_ft(x)
            result_cpu = result.to("cpu")
            result_np = result_cpu.detach().numpy()[:, :][0]
            max_pos = np.argmax(result_np)
            print(max_pos)