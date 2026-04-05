import torch
import torch.nn as nn

from game_agent.config import AgentConfig


class RewardHead(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict scalar reward from latent state.

        Args:
            z: (batch, latent_dim)
        Returns:
            (batch,) predicted reward.
        """
        return self.net(z).squeeze(-1)
