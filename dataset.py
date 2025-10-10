from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
import PIL
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
#dataget preproccessed from rangpur
#downloaded local 

number_png = 5
count = 0
test_set = []

directory_seg_test = r"C:\Users\teism\PatternAnalysis-2025-1\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_test"

directory_train = r"C:\Users\teism\PatternAnalysis-2025-1\keras_png_slices_data\keras_png_slices_data\keras_png_slices_test"

avg_seg_size = 0
avg_size = 0

for entry in os.scandir(directory):  
    if entry.is_file():
        a = PIL.Image.open(entry.path)
        #print(a.size)
        test_set.append(a)
        avg_size = avg_size + os.path.getsize(entry.path) / 2
        count+=1
    if count >= number_png:
        break

for entry in os.scandir(directory_seg):  
    if entry.is_file():
        a = PIL.Image.open(entry.path)
        #print(a.size)
        test_set.append(a)
        avg_seg_size = avg_seg_size + os.path.getsize(entry.path) / 2
        count+=1
    if count >= number_png:
        break

n_classes = 4
classes = ["background","CSF", "grey", "white"]

#uncomment to view png
#for i in test_set:
#    i.show()

#class to store the picture and convert into a usable numpy array
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        print(type(X))
        self.X = X
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        #TODO determine correct label value
        return np.array(self.X[idx]), idx

train_customdataset = ImageDataset(test_set)
batch_size = 3
#train_loader = torch.utils.data.DataLoader(dataset=train_customdataset, batch_size=batch_size, shuffle=True)

#test loader