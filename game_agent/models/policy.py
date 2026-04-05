import torch
import torch.nn as nn
import torch.nn.functional as F

from game_agent.actions import Action
from game_agent.config import AgentConfig


class PolicyNetwork(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.num_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return raw action logits.

        Args:
            z: (batch, latent_dim)
        Returns:
            (batch, num_actions) logits.
        """
        return self.net(z)

    def act(self, z: torch.Tensor) -> Action:
        """Sample an action from the policy."""
        logits = self.forward(z)
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        return Action(idx)

    def act_greedy(self, z: torch.Tensor) -> Action:
        """Take the most probable action."""
        logits = self.forward(z)
        idx = logits.argmax(dim=-1).item()
        return Action(idx)
