"""V-JEPA 2.1 Multi-Scale Encoder.

Same CNN backbone as v1, but exposes intermediate feature maps for:
- Dense predictive loss (7x7 spatial features)
- Deep self-supervision (probes at 14x14 and 7x7)

Backward compatible: forward() still returns (B, 256) global latent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from game_agent.config import AgentConfig


class EncoderV2(nn.Module):
    """Multi-scale CNN encoder with intermediate feature access.

    Feature map resolutions:
        block1: (B, 32, 56, 56)
        block2: (B, 64, 28, 28)
        block3: (B, 128, 14, 14)
        block4: (B, 256, 7, 7)
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        in_ch = config.input_channels
        ch = config.encoder_channels  # (32, 64, 128, 256)
        latent_dim = config.latent_dim

        # Named blocks instead of one Sequential
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, ch[0], kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Project 14x14 features (128ch) to same dim as 7x7 (256ch)
        self.proj_14x14 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Positional embeddings for spatial features
        self.pos_embed_7x7 = nn.Parameter(torch.randn(1, ch[3], 7, 7) * 0.02)
        self.pos_embed_14x14 = nn.Parameter(torch.randn(1, ch[3], 14, 14) * 0.02)

        # Spatial normalization (GroupNorm with 1 group = LayerNorm per position)
        self.norm_7x7 = nn.GroupNorm(1, ch[3])
        self.norm_14x14 = nn.GroupNorm(1, ch[3])

        # Global pooling path (backward compat)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch[3], latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward_backbone(self, x: torch.Tensor) -> dict:
        """Run backbone and return all intermediate feature maps."""
        h1 = self.block1(x)   # (B, 32, 56, 56)
        h2 = self.block2(h1)  # (B, 64, 28, 28)
        h3 = self.block3(h2)  # (B, 128, 14, 14)
        h4 = self.block4(h3)  # (B, 256, 7, 7)
        return {'56x56': h1, '28x28': h2, '14x14': h3, '7x7': h4}

    def forward_multiscale(self, x: torch.Tensor) -> dict:
        """Return processed feature maps at 14x14 and 7x7.

        Both are projected to 256 channels with positional embeddings.

        Returns:
            dict with '7x7': (B, 256, 7, 7) and '14x14': (B, 256, 14, 14)
        """
        feats = self.forward_backbone(x)

        # 7x7: add positional embedding + normalize
        f7 = feats['7x7'] + self.pos_embed_7x7
        f7 = self.norm_7x7(f7)

        # 14x14: project to 256ch, add positional embedding + normalize
        f14 = self.proj_14x14(feats['14x14'])
        f14 = f14 + self.pos_embed_14x14
        f14 = self.norm_14x14(f14)

        return {'7x7': f7, '14x14': f14}

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Return 7x7 spatial features with pos embedding. (B, 256, 7, 7)"""
        feats = self.forward_backbone(x)
        f7 = feats['7x7'] + self.pos_embed_7x7
        return self.norm_7x7(f7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backward compatible: (B, C, H, W) -> (B, 256) global latent."""
        feats = self.forward_backbone(x)
        h = self.pool(feats['7x7']).flatten(1)
        h = self.fc(h)
        return self.norm(h)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Convenience: handles single observation without batch dim."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return self.forward(obs)

    def load_v1_weights(self, v1_state_dict: dict):
        """Load weights from a v1 Encoder (remaps Sequential indices to named blocks)."""
        mapping = {
            'convs.0.': 'block1.0.', 'convs.2.': 'block2.0.',
            'convs.4.': 'block3.0.', 'convs.6.': 'block4.0.',
        }
        new_sd = {}
        for k, v in v1_state_dict.items():
            new_k = k
            for old, new in mapping.items():
                if k.startswith(old):
                    new_k = k.replace(old, new)
                    break
            if new_k in self.state_dict():
                new_sd[new_k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        loaded = len(new_sd)
        print(f'Loaded {loaded} v1 weights, '
              f'{len(missing)} new params (proj, pos_embed, norms)')
