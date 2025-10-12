import numpy as np
import debugHelp as dh
import torch
import torchvision
import PIL
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

#downloaded local 
number_png = 5
directory_seg_train = r"C:\Users\teism\PatternAnalysis-2025-1\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_train"
directory_train = r"C:\Users\teism\PatternAnalysis-2025-1\keras_png_slices_data\keras_png_slices_data\keras_png_slices_train"
directory_seg_test = r"C:\Users\teism\PatternAnalysis-2025-1\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_test"
directory_test = r"C:\Users\teism\PatternAnalysis-2025-1\keras_png_slices_data\keras_png_slices_data\keras_png_slices_test"

def dataFromFile(path: str, size: int) -> list[PIL.Image]:
    """imports 'size' number of images from 'path' return array of PIL."""
    count = 0
    sets = []
    for entry in os.scandir(directory_train):  
        if entry.is_file():
            a = PIL.Image.open(entry.path)
            sets.append(a)
            count+=1
        if count >= size:
            return sets

train_set = dataFromFile(directory_train, number_png)
seg_train_set = dataFromFile(directory_seg_train, number_png)
test_set = dataFromFile(directory_test, number_png)
seg_test_set = dataFromFile(directory_seg_test, number_png)

#define classes/labels
n_classes = 4
classes = ["background","CSF", "grey", "white"]

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
        #image_temp = torchvision.transforms.Normalize((0.5), (0.5))
        mask = PIL.Image.fromarray(np.array(self.Y[idx]))
        mask = torchvision.transforms.Resize((64, 64),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST)(mask)
        mask = np.array(mask)
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == 85] = 1
        binary_mask[mask == 170] = 2
        binary_mask[mask == 255] = 3
        binary_mask[mask == 0] = 0
        return image_temp, binary_mask

#dataset creator
test_customdataset = ImageDataset(test_set, seg_test_set)
train_customdataset = ImageDataset(train_set, seg_train_set)

#train loader
batch_size = 3
train_loader = torch.utils.data.DataLoader(dataset=train_customdataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_customdataset, batch_size=batch_size, shuffle=True)