import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

def load_paired_img_wrd(folder, word_vectors, use_word_vectors=True):
    '''
    If use_word_vectors = true, and using VGG16 with Imagenet:
    Will have 300 embedding layer at end of network
    Instead of 4096 imagenet class layer at the end of the network
    '''
    class_names = [fold for fold in os.listdir(folder) if ".DS" not in fold]
    image_list = []
    labels_list = []
    paths_list = []
    for cl in class_names:
        splits = cl.split("_")
        if use_word_vectors:
            vectors = np.array([word_vectors[split] if split in word_vectors else np.zeros(shape=300) for split in splits])
            class_vector = np.mean(vectors, axis=0)
        subfiles = [f for f in os.listdir(folder + "/" + cl) if ".DS" not in f]

        for subf in subfiles:
            full_path = os.path.join(folder, cl, subf)
            img = Image.open(full_path)
            img = img.resize((224, 224))
            x = np.asarray(img)
            image_list.append(x)
            if use_word_vectors:
                labels_list.append(class_vector)
            paths_list.append(full_path)
    img_data = np.array(image_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]

    return img_data, np.array(labels_list), paths_list

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def load_model(model_path, number_class = 19):
    model_ft = models.resnet101(pretrained=True)
    set_parameter_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, number_class)
    input_size = 224
    model_ft.load_state_dict(torch.load(model_path))
    return model_ft