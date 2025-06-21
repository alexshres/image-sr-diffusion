# Alex Shrestha
# @FILE: unet.py
# Implementation of U-Net blocks and denoiser for Super-Resolution

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 2 convolutions followed by 2 ReLUs
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.conv(x)


class UNetDenoiser(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)        # downsample by 2
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)        # downsample by 2

        # Bottleneck
        self.bottleneck = UNetBlock(128, 256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)    # upsampled by 2
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # upsampled by 2 (to original size)
        self.dec1 = UNetBlock(128, 64)

        # Final convolution
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x, _):
        # Encode
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decode
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        return self.final(d1)
