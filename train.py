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
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

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
            images = images.float()
            #optimizer.zero_grad()

            images = images.unsqueeze(1).to(device)
            
            outputs = model(images)

            pred = outputs[:, 0]  
            #print(f"pred shape: {pred.shape}, masks shape: {masks.shape}")
            loss = criterion(pred, masks)
            optimiser.zero_grad()
            # Backward pass
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        #if(avg_loss < 0.05):
        #    print("Success")

    print(" Training complete with enhanced U-Net!")
    model.eval()
    fig, axes = plt.pyplot.subplots(1, 3, figsize=(12, 9))
    #batch_idx, image, masks = enumerate(test_loader) #test_customdataset[1]
    image, masks = test_customdataset[1]

    images = torch.from_numpy(image).to(device) #[1][0].to(device)
    images = images.float()
    images = images.unsqueeze(1).to(device)

    #mask = image[1][1]
    pred = pred.detach().numpy()
    #mask = mask.detach().numpy()
    
    print("pred: ", pred.shape)
    print("mask: ", masks.shape)
    axes[0].imshow(pred[0]) 
    axes[1].imshow(masks)
    axes[2].imshow(image)

    plt.pyplot.show()
    model.train()

    return losses

model = Unet(ins=1, outs=4, dropout=0.2)
#lowered learning rate to improve performance
#non optimal learning optima due to large region of the same value. 
losses = train(model, train_loader, test_loader, epochs=50, lr=0.0005)