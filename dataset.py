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
import debugHelp as dh

device = "cuda" if torch.cuda.is_available() else "cpu"
#dataget preproccessed from rangpur
#downloaded local 

number_png = 5
count = 0
test_set = []
seg_test_set = []

directory_seg_test = r"C:\Users\teism\PatternAnalysis-2025-1\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_test"

directory_train = r"C:\Users\teism\PatternAnalysis-2025-1\keras_png_slices_data\keras_png_slices_data\keras_png_slices_test"

avg_seg_size = 0
avg_size = 0

for entry in os.scandir(directory_train):  
    if entry.is_file():
        a = PIL.Image.open(entry.path)
        #print(a.size)
        test_set.append(a)
        avg_size = avg_size + os.path.getsize(entry.path) / 2
        count+=1
    if count >= number_png:
        break

for entry in os.scandir(directory_seg_test):  
    if entry.is_file():
        a = PIL.Image.open(entry.path)
        #print(a.size)
        seg_test_set.append(a)
        avg_seg_size = avg_seg_size + os.path.getsize(entry.path) / 2
        count+=1
    if count >= number_png:
        break


#keras_png_slices_train
#case_001_slice_0, case_001_slice_3, case_002_slice_8
#keras_png_slices_seg_train
#seg_001_slice_0, seg_001_slice_3, seg_002_slice_8


n_classes = 4
classes = ["background","CSF", "grey", "white"]

#uncomment to view png
#for i in test_set:
#    i.show()

#class to store the picture and convert into a usable numpy array
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        #return image and image mask
        image_temp = np.array(self.X[idx])
        dh.displayNumpyArray(image_temp)
        #image_temp = torchvision.transforms.Normalize((0.5), (0.5))
        mask = PIL.Image.fromarray(np.array(self.Y[idx]))
        mask = torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(mask)
        mask = np.array(mask)
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        
        binary_mask[mask == 85] = 1
        binary_mask[mask == 170] = 2
        binary_mask[mask == 255] = 3
        binary_mask[mask == 0] = 0
        #print(list(mask))
        #print(list(binary_mask))
        #dh.displayNumpyArray(mask)
        #dh.displayNumpyArray(binary_mask*85)
        dh.displayNormalise(binary_mask)
        return image_temp, binary_mask

train_customdataset = ImageDataset(test_set, seg_test_set)
#print(train_customdataset.__getitem__(0))
#print(train_customdataset.__getitem__(0)[0])
temp = train_customdataset.__getitem__(0)[1]
print(temp)
#print(type(train_customdataset.__getitem__(0)[0]))
#print(np.shape(train_customdataset.__getitem__(0)[0]))
#dh.displayNumpyArray(train_customdataset.__getitem__(0)[0])

print(type(temp))
print(np.shape(temp))
#dh.displayNumpyArray(temp)
batch_size = 3
#train_loader = torch.utils.data.DataLoader(dataset=train_customdataset, batch_size=batch_size, shuffle=True)

#test loader