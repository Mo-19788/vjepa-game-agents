"""Dense spatial dynamics predictor for V-JEPA 2.1.

Predicts next frame's full 7x7 spatial feature map from the current
(partially masked) feature map + action. Forces every spatial position
to be informative since any position might be masked.

Used during training only — inference uses the global dynamics predictor.
"""

import torch
import torch.nn as nn


class DenseDynamicsPredictor(nn.Module):
    """Spatial dynamics: (B, 256, 7, 7) + action -> (B, 256, 7, 7)."""

    def __init__(self, feat_dim: int = 256, num_actions: int = 5,
                 action_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.action_embed = nn.Embedding(num_actions, action_dim)

        # Residual CNN predictor
        in_ch = feat_dim + action_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, feat_dim, 3, padding=1),
        )
        self.norm = nn.GroupNorm(1, feat_dim)

    def forward(self, feat: torch.Tensor, action: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            feat: (B, C, 7, 7) spatial features
            action: (B,) integer actions
            mask: (B, 1, 7, 7) binary mask. 1=keep, 0=masked. If None, no masking.
        Returns:
            (B, C, 7, 7) predicted next-frame features
        """
        B = feat.shape[0]

        # Apply mask if provided
        if mask is not None:
            feat = feat * mask

        # Broadcast action embedding to spatial grid
        a = self.action_embed(action.long())  # (B, 64)
        a = a.unsqueeze(-1).unsqueeze(-1).expand(B, -1, 7, 7)  # (B, 64, 7, 7)

        # Concat features + action
        x = torch.cat([feat, a], dim=1)  # (B, 320, 7, 7)

        # Predict with residual
        out = self.net(x)
        return self.norm(feat * (mask if mask is not None else 1) + out)


def generate_spatial_mask(batch_size: int, h: int = 7, w: int = 7,
                          mask_ratio: float = 0.4,
                          device: torch.device = None) -> torch.Tensor:
    """Generate random spatial mask. 1=keep, 0=masked.

    Returns: (B, 1, H, W) float tensor.
    """
    mask = (torch.rand(batch_size, 1, h, w, device=device) > mask_ratio).float()
    return mask
