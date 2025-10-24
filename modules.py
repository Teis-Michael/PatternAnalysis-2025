import numpy as np
import torch
from torch import nn
#UNet_segmentation_code_demo.ipynb 

class Unet(nn.Module):
    def __init__(self, ins, outs, dropout):
        super(Unet, self).__init__()
        # Encoder (downsampling)
        #self.enc1 = self._conv_block(ins, 32, dropout)
        var = 4
        self.enca = self._conv_block(ins, var, dropout)
        self.encb = self._conv_block(var, 32, dropout)
        self.enc2 = self._conv_block(32, 64, dropout)
        self.enc3 = self._conv_block(64, 128, dropout)

        # Decoder (upsampling)
        self.dec3 = self._conv_block(128 + 64, 64, dropout)
        self.dec2 = self._conv_block(64 + 32, 32, dropout)
        self.deca = self._conv_block(32 + var, var, dropout)
        self.decb = nn.Conv2d(var, outs, 1)
        #self.decb = nn.Conv2d(16, outs, 3)
        #self.dec1 = nn.Conv2d(32, outs, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for final output


    def _conv_block(self, in_ch, out_ch, dropout_p=0.2):
        """3 segment of conv relu dropdown"""
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

    def forward(self, x: torch.tensor):
        # Encoder
        e0 = self.enca(x)
        e1 = self.encb(self.pool(e0))

        #e1 = self.enc1(x)          # 64x64
        e2 = self.enc2(self.pool(e1))  # 32x32
        e3 = self.enc3(self.pool(e2))  # 16x16

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(e3), e2], 1))  # 32x32
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1))  # 64x64
        d2 = self.deca(torch.cat([self.upsample(d2), e0], 1))
        out = self.decb(d2) #dec1(d2)

        # Apply sigmoid activation to final output
        out = self.sigmoid(self.pool(out))
        #out = self.sigmoid(out)

        return self.pool(out)
    
class diceloss(nn.Module):
    """loss function"""
    def __init__(self, smooth=1e-8):
        super(diceloss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, targ):
        """return loss value"""
        #flatten
        pred = pred.reshape(-1)
        targ = targ.reshape(-1).float()
    
        #intersect and union
        inter = (pred * targ).sum() + self.smooth
        union = pred.sum() + targ.sum() + self.smooth

        dice_coeff = (2. * inter) / union
        return 1 - dice_coeff