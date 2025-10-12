import modules
from modules import Unet
import dataset
from dataset import train_loader
from dataset import test_loader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_loader, test_loader, epochs=3, lr=0.001):
    model.to(device)
    #criterion = diceloss()
    #optimiser

    losses = []

    print(" Starting training with Batch Norm, LeakyReLU, and Sigmoid activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        losses = []

        # Training loop with progress
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            #images = images.to(dtype = torch.float32)
            images = images.float()
            #optimizer.zero_grad()
            print(type(images))
            print(type(images[0]))
            print(type(images[0][0]))
            print(type(images[0][0][0]))
            print(images[0][0][0])
            
            outputs = model(images) #TODO convert to float

            pred_pet = outputs[:, 0]  # Pet class probability from sigmoid
            #print the shape of pred_pet and masks for debugging
            # print(f"pred_pet shape: {outputs.shape}, masks shape: {masks.shape}")
            #loss = criterion(pred_pet, masks)

            # Backward pass
            #loss.backward()
            #optimizer.step()

            #epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        if(avg_loss < 0.05):
            #equvalent ot 0.95 accuracy
            print("Success")

        # Visualize predictions after each epoch (or every few epochs)
        #if (epoch) % visualize_every == 0:
        #    show_epoch_predictions(model, test_dataset, epoch + 1, n=3)

    print(" Training complete with enhanced U-Net!")
    return losses

model = Unet(ins=3, outs=1, dropout=0.2)
losses = train(model, train_loader, test_loader, epochs=3, lr=0.001)