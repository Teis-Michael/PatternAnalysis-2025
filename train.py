import modules
from modules import Unet
from modules import diceloss
import dataset
from dataset import train_loader
from dataset import test_loader
import torch

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
            
            #images = images.to(dtype = torch.float32)
            images = images.float()
            #optimizer.zero_grad()
            
            print(images.size())
            images = images.unsqueeze(1).to(device)
            print(images.size())

            outputs = model(images) #TODO convert to float

            pred = outputs[:, 0]  # class probability from sigmoid
            #print the shape of pred_pet and masks for debugging
            print(f"pred shape: {pred.shape}, masks shape: {masks.shape}")
            loss = criterion(pred, masks)

            # Backward pass
            loss.backward()
            #optimizer.step()

            epoch_loss += loss.item()

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

model = Unet(ins=1, outs=1, dropout=0.2)
losses = train(model, train_loader, test_loader, epochs=3, lr=0.001)