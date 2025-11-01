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
epochs = 1000
lr = 0.0008

def train(model, train_loader, test_customdataset, epochs=3, lr=0.001):
    model.to(device)
    criterion = MultiClassDiceLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    epo_count = []

    print(" Starting training with Batch Norm, LeakyReLU, and Sigmoid activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device).float().unsqueeze(1)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
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
    with torch.no_grad():
        fig, axes = plt.pyplot.subplots(1, 3, figsize=(12, 4))
        image, masks = test_customdataset[1]
        #(256,256) -> (1,256,256) ->(1,1,256,256)
        images = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        outputs = model(images)
        pred =  torch.argmax(outputs,dim=1).squeeze().cpu().numpy()

        #print("masks",numpy.unique(masks))
        #print("pred",pred)
        #print("pred: ", pred.shape)
        #print("mask: ", masks.shape)
        axes[0].imshow(pred) 
        axes[1].imshow(masks)
        axes[2].imshow(image)

        plt.pyplot.show()

    return losses, epo_count

model = Unet(in_channels=1, out_channels=4, dropout_p=0.1)
#lowered learning rate to improve performance
#non optimal learning optima due to large region of the same value. 

losses, eco = train(model, train_loader, test_customdataset, epochs=epochs, lr = lr)

#create plot of losses over epoch
plt.pyplot.axhline(y=0.1, color='b', linestyle='--')
plt.pyplot.axhline(y=0, color='k', linestyle='-')
plt.pyplot.plot(eco, losses)
plt.pyplot.xlabel("epoch")
plt.pyplot.ylabel("loss")
plt.pyplot.title("losses over epochs")
plt.pyplot.show()

torch.save(model.state_dict(), ("model_"+str(epochs)+"_"+str(lr)))

show_predictions(model, test_customdataset)
