"""Larger-receptive-field U-Net with configurable depth, trained on approved labels."""

import torch
from torch import nn
from torch.nn import functional as F

from solarseg.model import Block


class DeepUNet(nn.Module):
    def __init__(self, width=24, depth=4):
        super().__init__()
        if width < 4 or width % 4 or not 2 <= depth <= 5:
            raise ValueError("Width must be a positive multiple of four; depth must be 2–5")
        channels = [width * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList(
            [Block(1, channels[0])]
            + [Block(channels[i - 1], channels[i]) for i in range(1, depth + 1)]
        )
        self.dec = nn.ModuleList(
            [Block(channels[i + 1] + channels[i], channels[i]) for i in range(depth - 1, -1, -1)]
        )
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for block, skip in zip(self.dec, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return self.head(x)
