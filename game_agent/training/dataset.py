import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from game_agent.config import AgentConfig
from game_agent.preprocessing.transforms import Preprocessor


class TransitionDataset(Dataset):
    def __init__(self, config: AgentConfig):
        self.preprocessor = Preprocessor(config)
        self.files = sorted(glob.glob(os.path.join(config.data_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {config.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        obs = self.preprocessor(data["obs"])
        next_obs = self.preprocessor(data["next_obs"])
        action = torch.tensor(int(data["action"]), dtype=torch.long)
        reward = torch.tensor(float(data["reward"]), dtype=torch.float32)
        done = torch.tensor(bool(data["done"]), dtype=torch.bool)
        return obs, action, next_obs, reward, done


class TransitionBuffer:
    """In-memory buffer that flushes transitions to disk as .npz files."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.buffer: list = []
        # Count existing files to avoid overwriting
        self._counter = len(glob.glob(os.path.join(data_dir, "*.npz")))

    def add(self, obs: np.ndarray, action: int, next_obs: np.ndarray,
            reward: float = 0.0, done: bool = False):
        self.buffer.append({
            "obs": obs,
            "action": action,
            "next_obs": next_obs,
            "reward": reward,
            "done": done,
        })

    def flush(self):
        """Write all buffered transitions to disk."""
        for transition in self.buffer:
            path = os.path.join(self.data_dir, f"transition_{self._counter:06d}.npz")
            np.savez_compressed(path, **transition)
            self._counter += 1
        count = len(self.buffer)
        self.buffer.clear()
        return count

    def __len__(self):
        return self._counter + len(self.buffer)
