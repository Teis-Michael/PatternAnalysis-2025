import modules
from modules import Unet
from modules import *
import dataset
from dataset import train_loader
from dataset import test_customdataset
import torch
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def show_predictions(model: Unet, dataset, title = "test", n = 3):
    """visualise from dataset using model the prediction, true mask and original image"""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            images = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

            pred = model(images) #[:, 0] 
            pred = torch.argmax(pred,dim=1).squeeze().cpu().numpy()
                
            axes[i, 0].imshow(pred) 
            axes[i, 1].imshow(true_mask)
            axes[i, 2].imshow(image)

    plt.tight_layout()
    plt.show()

def test_dicevalue(model: Unet, dataset, n = 3):
    """display dice loss per class"""
    with torch.no_grad():
        class_avg = [0, 0, 0, 0]
        for i in range(n):
            image, true_mask = dataset[i]
            true_mask = torch.from_numpy(true_mask).unsqueeze(0)
            images = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            pred = model(images)
            loss_class = MultiClassDiceLoss().lossClass(pred, true_mask)
            class_avg = class_avg + loss_class
        
        print("class average: ", class_avg / n)
        x = np.array(["background","CSF", "grey", "white"])
        y = class_avg / n

        plt.bar(x,y)
        plt.xlabel("class")
        plt.ylabel("dice loss")
        plt.title("avg dice loss per class")
        plt.axhline(y=0.1, color='k', linestyle='--')
        plt.show()

if __name__ == '__main__':
    #load model
    model = Unet(in_channels=1, out_channels=4, dropout_p=0.1)
    model.load_state_dict(torch.load("model_1000_0.0005", weights_only=True))
    #show_predictions(model, test_customdataset)
    test_dicevalue(model, test_customdataset, n = 16)

