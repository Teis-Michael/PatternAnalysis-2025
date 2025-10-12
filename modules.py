import numpy as np
import torch
from torch import nn
#UNet_segmentation_code_demo.ipynb

class Unet(nn.Module):
    def __init__(self, ins, outs, dropout):
        super(Unet, self).__init__()
        #TODO alter channels number to align with actual dataset 256, 256 -> 64, 64
        # Encoder (downsampling)
        self.enc1 = self._conv_block(ins, 32, dropout)
        self.enc2 = self._conv_block(32, 64, dropout)
        self.enc3 = self._conv_block(64, 128, dropout)

        # Decoder (upsampling)
        self.dec3 = self._conv_block(128 + 64, 64, dropout)
        self.dec2 = self._conv_block(64 + 32, 32, dropout)
        self.dec1 = nn.Conv2d(32, outs, 1)

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

    def forward(self, x):
        #TODO 64x64 to 256x256
        # Encoder
        e1 = self.enc1(x)          # 64x64
        e2 = self.enc2(self.pool(e1))  # 32x32
        e3 = self.enc3(self.pool(e2))  # 16x16

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(e3), e2], 1))  # 32x32
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1))  # 64x64
        out = self.dec1(d2)

        # Apply sigmoid activation to final output
        out = self.sigmoid(out)

        return x