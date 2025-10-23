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

    #un normalise array
    #convert to image
    #show

def NormaliseMinMax(ten: torch.tensor):
    t_min, _ = torch.min(ten, dim=1, keepdim=True)
    t_max, _ = torch.max(ten, dim=1, keepdim=True)
    return (ten - t_min) / (t_max - t_min)