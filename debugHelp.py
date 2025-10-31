from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
import PIL
from PIL import Image
import os

def displayNumpyArray(arr :np.array):
    """converts np.array to image then display image"""
    image = Image.fromarray(arr)
    image.show()

def NormaliseMinMax(ten: torch.tensor):
    t_min, _ = torch.min(ten, dim=1, keepdim=True)
    t_max, _ = torch.max(ten, dim=1, keepdim=True)
    return (ten - t_min) / (t_max - t_min)

def testNormalisemethod(ten):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    
    #process and normalise data
    temp = torch.from_numpy(ten).float()
    image_temp = temp.unsqueeze(0)

    transform_compose = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=0, std=1, inplace=True)
    ])

    a = torch.nn.functional.normalize(image_temp, dim = 1)
    transform = torchvision.transforms.Normalize(0, 1)
    ten2 = transform(image_temp)
    ten_compose = transform_compose(image_temp)#/16)
    q = torch.sigmoid(a) 
    
    #display different processing methods
    axes[0, 0].imshow(ten) 
    axes[1, 0].imshow(np.array(ten2.squeeze(0)))
    axes[1, 1].imshow(np.array(ten_compose.squeeze(0)))
    axes[1, 2].imshow(np.array(a.squeeze(0)))
    axes[2,0].imshow(np.array(q.squeeze(0)))
    axes[2,1].imshow(np.array(torch.relu(a).squeeze(0)))
    axes[2,2].imshow(np.array(torch.selu(a * -1).squeeze(0) * -1))    
    axes[0,2].imshow(np.array(torch.sigmoid(a * -1).squeeze(0) * -1))
    plt.show()