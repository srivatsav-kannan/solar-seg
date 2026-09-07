"""A zero-initialized residual context module for the competition-only U-Net."""

import torch
from torch import nn
from torch.nn import functional as F

from research.models import DeepUNet


class ContextUNet(DeepUNet):
    def __init__(self, width=24, depth=3):
        super().__init__(width, depth)
        channels = width * 2**depth
        hidden = width * 2
        layers = [nn.Conv2d(channels, hidden, 1, bias=False)]
        for dilation in (2, 4, 8):
            layers.extend([
                nn.GroupNorm(4, hidden), nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3, padding=dilation, dilation=dilation, bias=False),
            ])
        layers.extend([nn.GroupNorm(4, hidden), nn.SiLU(), nn.Conv2d(hidden, channels, 1)])
        self.context = nn.Sequential(*layers)
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def initialize_base(self, state):
        expected = {k for k in self.state_dict() if not k.startswith("context.")}
        if set(state) != expected:
            raise ValueError("Parent must contain exactly the original U-Net parameters")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {
            k for k in self.state_dict() if k.startswith("context.")
        }:
            raise ValueError("Unexpected warm-start parameter mismatch")

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        x = x + self.context(x)
        for block, skip in zip(self.dec, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return self.head(x)
