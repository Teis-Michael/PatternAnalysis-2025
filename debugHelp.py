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
