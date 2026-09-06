"""Small grayscale U-Net baseline; no external weights or labels."""

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, channels_in, channels_out):
        super().__init__(
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels_out),
            nn.SiLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels_out),
            nn.SiLU(),
        )


class UNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.enc = nn.ModuleList(
            [
                Block(1, width),
                Block(width, 2 * width),
                Block(2 * width, 4 * width),
                Block(4 * width, 8 * width),
            ]
        )
        self.dec = nn.ModuleList(
            [Block(12 * width, 4 * width), Block(6 * width, 2 * width), Block(3 * width, width)]
        )
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < 3:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for block, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return self.head(x)


def segmentation_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = (
        1
        - (
            (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        ).mean()
    )
    return bce + dice


def device_auto():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
