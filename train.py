import modules
from modules import Unet
from modules import diceloss
import dataset
from dataset import train_loader
from dataset import test_customdataset
from dataset import test_loader
import torch
import matplotlib as plt
import numpy

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, train_loader, test_loader, epochs=3, lr=0.001):
    model.to(device)
    criterion = diceloss()
    #optimiser

    losses = []

    print(" Starting training with Batch Norm, LeakyReLU, and Sigmoid activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        losses = []

        # Training loop with progress
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            print(images.shape, " : ", masks.shape)
            
            #images = images.to(dtype = torch.float32)
            images = images.float()
            #optimizer.zero_grad()

            images = images.unsqueeze(1).to(device)
            #print(images.shape)
            outputs = model(images)

            pred = outputs[:, 0]  
            #print(f"pred shape: {pred.shape}, masks shape: {masks.shape}")
            loss = criterion(pred, masks)
            # Backward pass
            loss.backward()
            #optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        if(avg_loss < 0.05):
            print("Success")

        # Visualize predictions after each epoch
        """
        model.eval()
        fig, axes = plt.pyplot.subplots(3, 1, figsize=(12, 9))
        image, mask = enumerate(test_loader) #test_customdataset[1]
        #print(image)
        #print(len(image))
        #print(image[0])
        #print(image[1][0])
        #print(image[1][1])
        images = image[1][0].to(device)
        #images = torch.from_numpy(image)
        images = images.float()
        images = images.unsqueeze(1).to(device)
        for i in images:
            print(i.shape)
        pred = model(images)
        for i in images:
            print(i.shape)

        mask = image[1][1]
        pred = pred.detach().numpy() * 85
        mask = mask.detach().numpy() * 85
        
        print(pred[0].shape)
        axes[0].imshow(pred[0][0]) #numpy
        axes[1].imshow(pred[1][0]) #numpy
        axes[2].imshow(pred[2][0]) #numpy
        #axes[1].imshow(mask)

        plt.pyplot.show()
        model.train()
        """


    print(" Training complete with enhanced U-Net!")
    model.eval()
    fig, axes = plt.pyplot.subplots(3, 2, figsize=(12, 9))
    image, mask = enumerate(test_loader) #test_customdataset[1]
    #print("image, ", image.shape, " mask: ", mask.shape)
    images = image[1][0].to(device)
    images = images.float()
    images = images.unsqueeze(1).to(device)
    for i in images:
        print(i.shape)
    pred = model(images)
    for i in images:
        print(i.shape)

    mask = image[1][1]
    pred = pred.detach().numpy() #* 85
    mask = mask.detach().numpy() #* 85

    print("pred: ", pred.shape)
    axes[0, 0].imshow(pred[0][0]) #numpy
    axes[1, 0].imshow(pred[1][0]) #numpy
    axes[2, 0].imshow(pred[2][0]) #numpy
    axes[0, 1].imshow(mask[0])
    axes[1, 1].imshow(mask[1])
    axes[2, 1].imshow(mask[2])

    plt.pyplot.show()
    model.train()

    return losses

model = Unet(ins=1, outs=4, dropout=0.2)
losses = train(model, train_loader, test_loader, epochs=1, lr=0.001)