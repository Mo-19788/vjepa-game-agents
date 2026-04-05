import torch
import torch.nn as nn

from game_agent.config import AgentConfig


class Encoder(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        in_channels = config.input_channels
        ch = config.encoder_channels  # (32, 64, 128, 256)
        latent_dim = config.latent_dim

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[1], ch[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[2], ch[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch[3], latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observation(s) to latent vector(s).

        Args:
            x: (batch, C, H, W) tensor.
        Returns:
            (batch, latent_dim) tensor.
        """
        h = self.convs(x)
        h = self.pool(h).flatten(1)
        h = self.fc(h)
        h = self.norm(h)
        return h

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Convenience: handles single observation without batch dim."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return self.forward(obs)
