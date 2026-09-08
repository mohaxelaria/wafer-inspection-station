"""Wafer-map defect classifier.

Deliberately small: wafer maps are 32x32 two-channel binary images, so a deep
backbone would overfit long before it helped. Roughly 250k parameters, trains
in a few minutes on a laptop GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .labels import CLASSES


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            _block(2, width),            # 32 -> 16
            _block(width, width * 2),    # 16 -> 8
            _block(width * 2, width * 4),  # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(width * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
