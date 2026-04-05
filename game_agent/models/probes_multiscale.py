"""Multi-scale spatial probes for deep self-supervision.

Spatial probes operate on feature maps (not flattened latents), preserving
spatial structure. Applied at 14x14 and 7x7 resolutions.

Much smaller than the MLP probes (~10K params vs ~1.7M) and more accurate
because they preserve spatial information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPositionProbe(nn.Module):
    """Feature map -> player (col, row) position.

    Pools spatial features then predicts normalized [0, 1] coordinates.
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(in_channels, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2), nn.Sigmoid(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: (B, C, H, W) spatial features
        Returns:
            (B, 2) predicted (col, row) in [0, 1]
        """
        z = self.pool(feat_map).flatten(1)
        return self.net(z)


class SpatialCarProbe(nn.Module):
    """Feature map -> 12x12 car occupancy grid.

    Uses 1x1 convolutions to process features spatially, then upsamples
    to the target grid resolution. Much more natural than an MLP on
    flattened features.
    """

    def __init__(self, in_channels: int = 256, grid_size: int = 12):
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: (B, C, H, W) spatial features
        Returns:
            (B, 12, 12) occupancy logits
        """
        x = self.net(feat_map)
        x = F.interpolate(x, size=self.grid_size, mode='bilinear',
                          align_corners=False)
        return self.head(x).squeeze(1)  # (B, 12, 12)
