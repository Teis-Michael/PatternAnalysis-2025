import modules
from modules import Unet
import dataset
from dataset import train_loader
from dataset import test_customdataset
import torch
#import train
#from train import model
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def show_predictions(model: Unet, dataset, title = "test", n = 3):
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            images = torch.from_numpy(image).to(device) #[1][0].to(device)
            images = images.float()
            #print(images.shape)
            images = images.unsqueeze(0)
            images = images.unsqueeze(0).to(device)
            #print(images.shape)
            pred = model(images)[:, 0] 
            pred = pred.detach().numpy()
                
            print("pred: ", pred.shape)
            print("mask: ", true_mask.shape)
            axes[i, 0].imshow(pred[0]) 
            axes[i, 1].imshow(true_mask)
            axes[i, 2].imshow(image)

    plt.tight_layout()
    plt.show()

