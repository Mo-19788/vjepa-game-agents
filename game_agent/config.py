from dataclasses import dataclass, field
from typing import Tuple

import torch


@dataclass
class AgentConfig:
    # Capture
    window_title: str = "Retro Pong Game"
    capture_fps: int = 10

    # Preprocessing
    frame_size: Tuple[int, int] = (224, 224)
    grayscale: bool = False
    frame_stack: int = 1

    # Model dimensions
    latent_dim: int = 256
    num_actions: int = 3
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)

    # Planning
    planning_horizon: int = 10
    discount: float = 0.95

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    train_split: float = 0.9
    ema_tau: float = 0.996

    # Control
    action_hold_seconds: float = 0.1

    # Paths
    data_dir: str = "game_agent/data"
    checkpoint_dir: str = "game_agent/checkpoints"
    log_dir: str = "game_agent/logs"

    # Device
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def input_channels(self) -> int:
        c = 1 if self.grayscale else 3
        return c * self.frame_stack
