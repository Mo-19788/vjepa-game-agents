"""Position-aware shooting planner.

Uses the position probe to score actions directly —
reward = how close the paddle is to the ball in the imagined future.
"""

import torch
import torch.nn as nn

from game_agent.actions import Action
from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.dynamics import DynamicsPredictor


class PositionProbe(nn.Module):
    """Predicts ball_x, ball_y, left_paddle_y, right_paddle_y from latent."""
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class ShootingPlanner:
    def __init__(self, encoder: Encoder, dynamics: DynamicsPredictor,
                 position_probe: PositionProbe, config: AgentConfig,
                 num_samples: int = 512):
        self.encoder = encoder
        self.dynamics = dynamics
        self.probe = position_probe
        self.horizon = config.planning_horizon
        self.discount = config.discount
        self.num_actions = config.num_actions
        self.num_samples = num_samples
        self.device = config.resolve_device()

    @torch.no_grad()
    def choose_action(self, obs: torch.Tensor) -> Action:
        """Plan by imagining futures and scoring paddle-ball distance."""
        z = self.encoder(obs)

        # Sample random action sequences
        action_seqs = torch.randint(0, self.num_actions,
                                     (self.num_samples, self.horizon),
                                     device=self.device)

        z_batch = z.expand(self.num_samples, -1).clone()
        scores = torch.zeros(self.num_samples, device=self.device)

        for step in range(self.horizon):
            actions_at_step = action_seqs[:, step]
            z_batch = self.dynamics(z_batch, actions_at_step)

            # Score: negative distance between ball_y and paddle_y
            pos = self.probe(z_batch)  # (N, 4): ball_x, ball_y, left_paddle_y, right_paddle_y
            ball_y = pos[:, 1]
            paddle_y = pos[:, 2]
            reward = -torch.abs(ball_y - paddle_y)  # closer = higher score

            scores += (self.discount ** step) * reward

        best_idx = scores.argmax().item()
        best_first_action = action_seqs[best_idx, 0].item()
        return Action(best_first_action)
