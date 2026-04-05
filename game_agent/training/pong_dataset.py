"""Dataset loader for the Pong environment's episode format.

Reads PNG frames + actions.jsonl from episode directories and produces
transition tuples for training the world model.
"""

import json
import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from game_agent.config import AgentConfig
from game_agent.preprocessing.transforms import Preprocessor


class PongTransitionDataset(Dataset):
    """Loads transitions from Pong episode directories.

    Each episode directory contains:
        - frame_XXXXXX.png files
        - actions.jsonl (one JSON line per timestep)
        - states.jsonl (optional, for reward computation)
    """

    def __init__(self, data_root: str, config: AgentConfig, max_episodes: int = 0):
        self.preprocessor = Preprocessor(config)
        self.transitions = []  # list of (episode_dir, frame_idx)

        # Find all episode directories
        episode_dirs = sorted(glob.glob(os.path.join(data_root, "**", "ep_*"), recursive=True))
        if max_episodes > 0:
            episode_dirs = episode_dirs[:max_episodes]

        if not episode_dirs:
            raise FileNotFoundError(f"No episode directories found in {data_root}")

        for ep_dir in episode_dirs:
            actions_path = os.path.join(ep_dir, "actions.jsonl")
            if not os.path.exists(actions_path):
                continue

            # Count frames
            frames = sorted(glob.glob(os.path.join(ep_dir, "frame_*.png")))
            if len(frames) < 2:
                continue

            # Load actions
            actions = []
            with open(actions_path) as f:
                for line in f:
                    actions.append(json.loads(line))

            # Load states if available (for reward)
            states_path = os.path.join(ep_dir, "states.jsonl")
            states = []
            if os.path.exists(states_path):
                with open(states_path) as f:
                    for line in f:
                        states.append(json.loads(line))

            # Build transitions: (frame_t, action_t, frame_t+1, reward, done)
            num_transitions = min(len(frames) - 1, len(actions) - 1)
            for t in range(num_transitions):
                reward = 0.0
                done = False
                if states and t + 1 < len(states):
                    # Reward: +1 if left scores, -1 if right scores (from left paddle perspective)
                    score_left_now = states[t + 1].get("score_left", 0)
                    score_left_prev = states[t].get("score_left", 0)
                    score_right_now = states[t + 1].get("score_right", 0)
                    score_right_prev = states[t].get("score_right", 0)
                    if score_left_now > score_left_prev:
                        reward = 1.0
                    elif score_right_now > score_right_prev:
                        reward = -1.0
                    done = states[t + 1].get("done", False)

                self.transitions.append({
                    "ep_dir": ep_dir,
                    "t": t,
                    "action_left": actions[t].get("action_left", 0),
                    "reward": reward,
                    "done": done,
                })

        if not self.transitions:
            raise FileNotFoundError(f"No valid transitions found in {data_root}")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        tr = self.transitions[idx]
        ep_dir = tr["ep_dir"]
        t = tr["t"]

        # Load frames
        frame_path = os.path.join(ep_dir, f"frame_{t:06d}.png")
        next_frame_path = os.path.join(ep_dir, f"frame_{t + 1:06d}.png")

        obs = cv2.imread(frame_path)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        next_obs = cv2.imread(next_frame_path)
        next_obs = cv2.cvtColor(next_obs, cv2.COLOR_BGR2RGB)

        obs_tensor = self.preprocessor(obs)
        next_obs_tensor = self.preprocessor(next_obs)

        action = torch.tensor(tr["action_left"], dtype=torch.long)
        reward = torch.tensor(tr["reward"], dtype=torch.float32)
        done = torch.tensor(tr["done"], dtype=torch.bool)

        return obs_tensor, action, next_obs_tensor, reward, done
