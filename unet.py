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


