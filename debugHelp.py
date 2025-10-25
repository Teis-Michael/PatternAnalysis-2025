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

def displayNormalise(arr: np.array):
    #TODO fix dispaly
    """display image that has been normalised"""
    fig = plt.figure(gihsize=())
    #X, Y = np.meshgrid
    #plt.pcolormesh(X, Y, arr)
    plt.imshow(arr, interpolation='none')
    plt.show()


def NormaliseMinMax(ten: torch.tensor):
    t_min, _ = torch.min(ten, dim=1, keepdim=True)
    t_max, _ = torch.max(ten, dim=1, keepdim=True)
    return (ten - t_min) / (t_max - t_min)

def testNormalisemethod(ten):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes[0, 0].imshow(ten) 

    transform = torchvision.transforms.Normalize(0, 1)
    temp = torch.from_numpy(ten).float()
    print(torch.mean(temp))
    print(torch.var(temp))

    image_temp = temp.unsqueeze(0)

    transform_compose = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=0, std=1, inplace=True)
    ])

    a = torch.nn.functional.normalize(image_temp, dim = 1)
    
    a = torch.nn.functional.normalize(a, dim = 0)
    #torchvision.transforms.functional
    print(torch.mean(a))
    print(torch.var(a))
    #ten = transform(ten[0])
    ten = transform(image_temp)
    ten_compose = transform_compose(image_temp)#/16)
    print(torch.mean(ten))
    print(torch.var(ten))

    print(torch.mean(ten_compose))
    print(torch.var(ten_compose))
    axes[1, 0].imshow(np.array(ten.squeeze(0)))
    axes[1, 1].imshow(np.array(ten_compose.squeeze(0)))
    axes[1, 2].imshow(np.array(a.squeeze(0)))
    plt.show()