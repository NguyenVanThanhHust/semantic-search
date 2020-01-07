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
    model_ft.fc = nn.Linear(num_ftrs, 51)
    
    input_size = 224
    # Print the model we just instantiated
    PATH='models/resnet_101__281.pth'
    model_ft.load_state_dict(torch.load(PATH))
    # Data augmentation and normalization for training
    # Just normalization for validation

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Send the model to GPU
    model_ft = model_ft.to(device)
    dir = "./../Datasets/images/val"
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    print("author: ")
    print(classes)
    total_img = 0
    true_class = 0
    false_class = 0
    for each_folder in classes:
        folder_path = os.path.join(dir, each_folder)
        list_img = [d.name for d in os.scandir(folder_path) if d.is_file()]
        for each_img_name in list_img:
            img_path = os.path.join(dir, each_folder, each_img_name)
            image = Image.open(img_path)
            total_img += 1
            image = image.resize((224, 224)) 
            x = TF.to_tensor(image)
            x.unsqueeze_(0)
            x = x.to(device)
            result = model_ft(x)
            result_cpu = result.to("cpu")
            
            result_np = result_cpu.detach().numpy()[:, :][0]
            # print("Result: ", result_np)
            
            max_pos = np.argmax(result_np)
            print("Result: ", max_pos)
            
    #         if str(classes[max_pos]) == str(each_folder):
    #             print("This is true: ", classes[max_pos])
    #             true_class += 1
    #         else: 
    #             print("This is false. True is: ", each_folder, " misrecognized to: ", classes[max_pos])
    #             false_class += 1
                
    # print("accuracy = ", str(true_class/total_img))
    # print("false ratio = ", str(false_class/total_img))