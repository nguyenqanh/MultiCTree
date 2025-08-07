from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob
import R2UNet as R2UNet_parts


class ATTUNet(nn.Module):
    """
    Attention U-Net wrapped to have explicit output conv layer like Unet
    """
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.encoder_decoder = R2UNet_parts.AttU_Net(img_ch=in_channels, output_ch=n_classes)

        # Replace the implicit sigmoid with a separate output conv block
        self.outc = nn.Sequential(
            nn.Conv2d(n_classes, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_decoder(x)
        return self.outc(x)
