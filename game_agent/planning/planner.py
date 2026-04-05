import torch

from game_agent.actions import Action
from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.dynamics import DynamicsPredictor
from game_agent.models.reward_head import RewardHead


class GreedyPlanner:
    def __init__(self, encoder: Encoder, dynamics: DynamicsPredictor,
                 reward_head: RewardHead, config: AgentConfig):
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward_head = reward_head
        self.num_actions = config.num_actions
        self.device = config.resolve_device()

    @torch.no_grad()
    def choose_action(self, obs: torch.Tensor) -> Action:
        """Pick the action whose predicted next state has highest reward.

        Args:
            obs: (1, C, H, W) preprocessed observation tensor.
        Returns:
            Best Action.
        """
        z = self.encoder(obs)  # (1, latent_dim)

        # Batch all actions: repeat z for each action
        z_batch = z.expand(self.num_actions, -1)  # (num_actions, latent_dim)
        actions = torch.arange(self.num_actions, device=self.device)  # (num_actions,)

        z_next = self.dynamics(z_batch, actions)  # (num_actions, latent_dim)
        rewards = self.reward_head(z_next)  # (num_actions,)

        best_idx = rewards.argmax().item()
        return Action(best_idx)
