from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob
from unet_utils import *

class Unet(nn.Module):
    """
    Modified U-Net architecture using unet_utils.py components
    Aligned with standard U-Net channel progression and naming
    """
    def __init__(self, img_ch, fch_base=64, isBN=True, isDeconv=True):
        super(Unet, self).__init__()

        # Encoder path
        self.inc = ConvNoPool(img_ch, fch_base, isBN)                # 64
        self.down1 = ConvPool(fch_base, fch_base * 2, isBN)          # 128
        self.down2 = ConvPool(fch_base * 2, fch_base * 4, isBN)      # 256
        self.down3 = ConvPool(fch_base * 4, fch_base * 8, isBN)      # 512
        self.down4 = ConvPool(fch_base * 8, fch_base * 16, isBN)     # 1024

        # Decoder path
        self.up1 = UpsampleConv(fch_base * 16, fch_base * 8, isDeconv, isBN)
        self.up2 = UpsampleConv(fch_base * 8, fch_base * 4, isDeconv, isBN)
        self.up3 = UpsampleConv(fch_base * 4, fch_base * 2, isDeconv, isBN)
        self.up4 = UpsampleConv(fch_base * 2, fch_base, isDeconv, isBN)

        # Output layer with sigmoid inside
        self.outc = ConvOut(fch_base)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        output = self.outc(u4)  # Sigmoid is already included here

        return output
        