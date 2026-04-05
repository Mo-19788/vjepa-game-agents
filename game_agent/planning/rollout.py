import itertools

import torch

from game_agent.actions import Action
from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.dynamics import DynamicsPredictor
from game_agent.models.reward_head import RewardHead


class RolloutPlanner:
    def __init__(self, encoder: Encoder, dynamics: DynamicsPredictor,
                 reward_head: RewardHead, config: AgentConfig):
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward_head = reward_head
        self.horizon = config.planning_horizon
        self.discount = config.discount
        self.num_actions = config.num_actions
        self.device = config.resolve_device()

        # Precompute all action sequences: (num_sequences, horizon)
        seqs = list(itertools.product(range(self.num_actions), repeat=self.horizon))
        self.action_sequences = torch.tensor(seqs, dtype=torch.long, device=self.device)
        self.num_sequences = len(seqs)  # num_actions^horizon

    @torch.no_grad()
    def choose_action(self, obs: torch.Tensor) -> Action:
        """Rollout all action sequences and pick the first action of the best one.

        Args:
            obs: (1, C, H, W) preprocessed observation tensor.
        Returns:
            Best first Action.
        """
        z = self.encoder(obs)  # (1, latent_dim)

        # Expand z for all sequences
        z_batch = z.expand(self.num_sequences, -1).clone()  # (N, latent_dim)
        scores = torch.zeros(self.num_sequences, device=self.device)

        for step in range(self.horizon):
            actions_at_step = self.action_sequences[:, step]  # (N,)
            z_batch = self.dynamics(z_batch, actions_at_step)  # (N, latent_dim)
            rewards = self.reward_head(z_batch)  # (N,)
            scores += (self.discount ** step) * rewards

        best_idx = scores.argmax().item()
        best_first_action = self.action_sequences[best_idx, 0].item()
        return Action(best_first_action)
