import numpy as np
import torch
from torch import nn
#UNet_segmentation_code_demo.ipynb 

class Unet(nn.Module):
    """Simplified U-Net for demo purposes with batch normalization, LeakyReLU, dropout and sigmoid activation."""

    def __init__(self, in_channels=1, out_channels=1, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, 32, dropout_p)
        self.enc2 = self._conv_block(32, 64, dropout_p)
        self.enc3 = self._conv_block(64, 128, dropout_p)

        # Decoder (upsampling)
        self.dec3 = self._conv_block(128 + 64, 64, dropout_p)
        self.dec2 = self._conv_block(64 + 32, 32, dropout_p)
        self.dec1 = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for final output

    def _conv_block(self, in_ch, out_ch, dropout_p=0.2):
        """Conv block with batch normalization and LeakyReLU: Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_p)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(e3), e2], 1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1))
        out = self.dec1(d2)

        return out

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth
        self.num_classes = 4
    
    def forward(self, pred, targ):
        """return loss value"""
        #seperates the mask in 4 mask, 1 for each class
        target_one_hot = torch.nn.functional.one_hot(targ.long(),self.num_classes)
        target_one_hot = target_one_hot.permute(0,3,1,2).float()

        pred = torch.softmax(pred, dim=1)

        dice_scores = []

        for cls in range(self.num_classes):
            #performs dice loss for each class
            pred_cls = pred[:,cls].reshape(-1)
            target_cls = target_one_hot[:,cls].reshape(-1)
            intersection = (pred_cls * target_cls).sum()
            dice_coeff = (2.0 * intersection + self.smooth) / (pred_cls.sum() + target_cls.sum() + self.smooth)
            dice_scores.append(dice_coeff)

        return 1 - torch.mean(torch.stack(dice_scores))
    
    def lossClass(self, pred, targ):
        """return loss value per class"""
        #seperates the mask in 4 mask, 1 for each class
        target_one_hot = torch.nn.functional.one_hot(targ.long(),self.num_classes)
        target_one_hot =  target_one_hot.permute(0,3,1,2).float()

        pred =  torch.softmax(pred, dim=1)

        dice_scores = []

        for cls in range(self.num_classes):
            #performs dice loss for each class
            pred_cls = pred[:,cls].reshape(-1)
            target_cls = target_one_hot[:,cls].reshape(-1)
            intersection = (pred_cls * target_cls).sum()
            dice_coeff = (2.0 * intersection + self.smooth) / (pred_cls.sum() + target_cls.sum() + self.smooth)
            dice_scores.append(dice_coeff)

        #return (-1 * (np.array(dice_scores))) - 1
        return 1 - np.array(dice_scores)
