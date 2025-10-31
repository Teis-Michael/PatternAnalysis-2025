import modules
from modules import Unet
from modules import *
import dataset
from dataset import train_loader
from dataset import test_customdataset
from dataset import test_loader
import torch
import matplotlib as plt
import numpy
import predict
from predict import show_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_loader, test_customdataset, epochs=3, lr=0.001):
    model.to(device)
    criterion = MultiClassDiceLoss() #torch.nn.BCEWithLogitsLoss #torch.nn.BCELoss() #diceloss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    epo_count = []

    print(" Starting training with Batch Norm, LeakyReLU, and Sigmoid activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            images = images.float()
            images = images.unsqueeze(1).to(device)
            outputs = model(images)

            pred = outputs[:, 0]  
            
            loss = criterion(pred, masks)
            optimiser.zero_grad()

            # Backward pass
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        epo_count.append(epoch)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

    print(" Training complete with enhanced U-Net!")

    #display single prediction, true mask and original image
    model.eval()
    fig, axes = plt.pyplot.subplots(1, 3, figsize=(12, 9))
    image, masks = test_customdataset[1]

    images = torch.from_numpy(image).to(device)
    images = images.float()
    images = images.unsqueeze(1).to(device)

    pred = pred.detach().numpy()
    #mask = mask.detach().numpy()
    print("masks",numpy.unique(masks))
    print("pred",pred)
    print("pred: ", pred.shape)
    print("mask: ", masks.shape)
    axes[0].imshow(pred[0]) 
    axes[1].imshow(masks)
    axes[2].imshow(image)

    plt.pyplot.show()
    model.train()

    return losses, epo_count

#model = Unet(ins=1, outs=4, dropout=0.1)

model = Unet(in_channels=1, out_channels=4, dropout_p=0.1)
#lowered learning rate to improve performance
#non optimal learning optima due to large region of the same value. 

losses, eco = train(model, train_loader, test_customdataset, epochs=10, lr = 0.05)
#losses, eco = train(model, train_loader, test_customdataset, epochs=10, lr = 0.0005)
#losses, eco = train(model, train_loader, test_customdataset, epochs=100, lr = 0.0005)

#create plot of losses over epoch
plt.pyplot.axhline(y=0, color='r', linestyle='--')
plt.pyplot.plot(eco, losses)
plt.pyplot.xlabel("epoch")
plt.pyplot.ylabel("loss")
plt.pyplot.title("losses over epochs")
plt.pyplot.show()

show_predictions(model, test_customdataset)
