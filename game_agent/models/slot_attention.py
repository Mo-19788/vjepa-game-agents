"""Slot Attention encoder for object-centric world modelling.

Architecture:
  CNN backbone (shared with original Encoder) → 7x7x256 feature map
  + learned positional encoding
  → Slot Attention (K slots, D dims, T iterations)
  → K object slots of dimension D

  SlotDecoder (training only): spatial broadcast → CNN → reconstructed image
  Compatibility: slots can be aggregated back to a single 256-dim vector
                 so existing dynamics/probes keep working during transition.

Reference: Locatello et al., "Object-Centric Learning with Slot Attention", 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from game_agent.config import AgentConfig


class SlotAttentionModule(nn.Module):
    """Iterative slot attention over spatial features."""

    def __init__(self, num_slots: int = 20, slot_dim: int = 64,
                 input_dim: int = 256, num_iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.eps = eps

        # Slot initialisation — learnable mean + log-std
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Input projection
        self.norm_input = nn.LayerNorm(input_dim)
        self.proj_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, slot_dim, bias=False)

        # Slot update
        self.norm_slot = nn.LayerNorm(slot_dim)
        self.proj_q = nn.Linear(slot_dim, slot_dim, bias=False)

        # GRU for slot refinement
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP residual
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 4, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (B, N, D_in) spatial features (N = H*W positions).
        Returns:
            slots: (B, K, D_slot)
            attn_weights: (B, K, N) — per-slot attention over spatial positions
        """
        B, N, _ = inputs.shape
        K = self.num_slots

        # Project inputs to keys and values (shared across iterations)
        x = self.norm_input(inputs)
        k = self.proj_k(x)  # (B, N, D_slot)
        v = self.proj_v(x)  # (B, N, D_slot)

        # Sample initial slots
        mu = self.slot_mu.expand(B, K, -1)
        sigma = self.slot_log_sigma.exp().expand(B, K, -1)
        slots = mu + sigma * torch.randn_like(mu)

        attn = None
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slot(slots)
            q = self.proj_q(slots)  # (B, K, D_slot)

            # Dot-product attention: slots compete for spatial positions
            scale = self.slot_dim ** -0.5
            dots = torch.einsum('bkd,bnd->bkn', q, k) * scale  # (B, K, N)
            attn = F.softmax(dots, dim=1)  # Normalise over slots (competition)

            # Weighted mean of values
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)  # (B, K, N)
            updates = torch.einsum('bkn,bnd->bkd', attn_norm, v)  # (B, K, D_slot)

            # GRU update
            slots = self.gru(
                updates.reshape(B * K, -1),
                slots_prev.reshape(B * K, -1),
            ).reshape(B, K, -1)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class SlotEncoder(nn.Module):
    """CNN backbone → positional encoding → Slot Attention → K object slots."""

    def __init__(self, config: AgentConfig, num_slots: int = 20,
                 slot_dim: int = 64, num_iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        in_channels = config.input_channels
        ch = config.encoder_channels  # (32, 64, 128, 256)

        # Same CNN backbone as original Encoder (no pooling/FC)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[1], ch[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[2], ch[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        feat_dim = ch[3]  # 256

        # Learned positional encoding for the 7x7 spatial grid
        self.pos_embed = nn.Parameter(torch.randn(1, 49, feat_dim) * 0.02)

        # 1x1 conv to optionally change feature dim before slot attention
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        # Slot Attention
        self.slot_attn = SlotAttentionModule(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=feat_dim,
            num_iters=num_iters,
        )

        # Compatibility: aggregate slots → single latent for old probes/dynamics
        self.aggregate = nn.Sequential(
            nn.Linear(num_slots * slot_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to a single latent vector (backward-compatible).

        Args:
            x: (B, C, H, W)
        Returns:
            (B, latent_dim) aggregated latent
        """
        slots, _ = self.forward_slots(x)
        return self.aggregate(slots.flatten(1))

    def forward_slots(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to per-object slots.

        Args:
            x: (B, C, H, W)
        Returns:
            slots: (B, K, slot_dim)
            attn: (B, K, 49) attention masks over 7x7 grid
        """
        B = x.shape[0]
        features = self.backbone(x)  # (B, 256, 7, 7)
        features = features.flatten(2).permute(0, 2, 1)  # (B, 49, 256)
        features = features + self.pos_embed
        features = self.input_proj(features)
        slots, attn = self.slot_attn(features)
        return slots, attn

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Convenience: handles single observation without batch dim."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return self.forward(obs)

    def load_backbone_from(self, encoder_state_dict: dict):
        """Load CNN weights from a pretrained original Encoder."""
        backbone_keys = {}
        for k, v in encoder_state_dict.items():
            if k.startswith('convs.'):
                new_k = k.replace('convs.', 'backbone.')
                backbone_keys[new_k] = v
        missing, unexpected = self.backbone.load_state_dict(backbone_keys, strict=False)
        print(f'Loaded backbone: {len(backbone_keys)} keys, '
              f'missing={missing}, unexpected={unexpected}')


class SlotDecoder(nn.Module):
    """CNN decoder: spatial broadcast + transposed convolutions.

    Each slot is broadcast to a small spatial grid, then upsampled via
    learned transposed convolutions. Each slot produces RGB + alpha;
    slots are combined via softmax over alpha (competition for pixels).

    Used only during training to provide reconstruction signal.
    """

    def __init__(self, slot_dim: int = 64, output_size: int = 224,
                 output_channels: int = 3):
        super().__init__()
        self.output_size = output_size
        self.slot_dim = slot_dim
        self.init_res = 7  # start at 7x7 like the encoder feature map

        # Project slot to spatial feature map: slot → 7x7x128
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, 128 * self.init_res * self.init_res),
            nn.ReLU(inplace=True),
        )

        # Positional encoding for the 7x7 grid
        self.pos_embed = nn.Parameter(
            torch.randn(1, 128, self.init_res, self.init_res) * 0.02
        )

        # CNN upsample: 7x7 → 14x14 → 28x28 → 56x56 → 112x112 → 224x224
        self.decoder_cnn = nn.Sequential(
            # 7x7 → 14x14
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 14x14 → 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 28x28 → 56x56
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 56x56 → 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 112x112 → 224x224
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Final 1x1 conv → RGB + alpha
            nn.Conv2d(16, output_channels + 1, kernel_size=1),
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: (B, K, D_slot)
        Returns:
            recon: (B, C, H, W) reconstructed image
            masks: (B, K, H, W) per-slot soft masks
        """
        B, K, D = slots.shape
        R = self.init_res

        # Project each slot to a 7x7 feature map
        x = self.slot_proj(slots)  # (B, K, 128*7*7)
        x = x.reshape(B * K, 128, R, R)  # (B*K, 128, 7, 7)
        x = x + self.pos_embed  # add positional info

        # Decode through CNN
        out = self.decoder_cnn(x)  # (B*K, C+1, 224, 224)
        out = out.reshape(B, K, -1, self.output_size, self.output_size)

        # Split RGB and alpha
        rgb = out[:, :, :-1]   # (B, K, C, H, W)
        alpha = out[:, :, -1:]  # (B, K, 1, H, W)

        # Soft-combine: softmax over slots for each pixel
        masks = F.softmax(alpha, dim=1)  # (B, K, 1, H, W)
        recon = (rgb * masks).sum(dim=1)  # (B, C, H, W)

        masks = masks.squeeze(2)  # (B, K, H, W)
        return recon, masks
