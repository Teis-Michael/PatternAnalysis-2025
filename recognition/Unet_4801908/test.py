from modules import Unet
from modules import *
import dataset
from dataset import getDataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import predict
from predict import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(in_channels=1, out_channels=4, dropout_p=0.1)
model.load_state_dict(torch.load("model_1000_0.0005", weights_only=True))
n = 50

train_loader, test_loader, test_customdataset, train_customdataset = getDataLoader(80, 10)

with torch.no_grad():
        class_avg = [0, 0, 0, 0]
        criterion = MultiClassDiceLoss()

        losses = []
        batch_id = []

        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device).float().unsqueeze(1)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            losses.append(loss.item())
            batch_id.append(batch_idx)

        plt.plot(batch_id, losses)
        plt.xlabel("batch id")
        plt.ylabel("loss")
        plt.axhline(y=0.1, color='b', linestyle='--')
        plt.show()

