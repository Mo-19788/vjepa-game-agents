import torch
import torch.nn as nn
import torch.nn.functional as F

from game_agent.config import AgentConfig


class DynamicsPredictor(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.num_actions = config.num_actions
        self.action_embed = nn.Embedding(config.num_actions, 64)
        input_dim = config.latent_dim + 64

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, config.latent_dim),
        )
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state with residual connection.

        Args:
            z: (batch, latent_dim) current latent state.
            action: (batch,) integer action indices.
        Returns:
            (batch, latent_dim) predicted next latent state.
        """
        action_emb = self.action_embed(action.long())
        x = torch.cat([z, action_emb], dim=-1)
        out = self.net(x)
        return self.norm(out)
