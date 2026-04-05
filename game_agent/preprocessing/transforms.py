from collections import deque

import cv2
import numpy as np
import torch

from game_agent.config import AgentConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Preprocessor:
    def __init__(self, config: AgentConfig):
        self.size = config.frame_size
        self.grayscale = config.grayscale

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        """Convert an RGB uint8 frame (H, W, 3) to a normalized tensor (C, H, W)."""
        img = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[:, :, np.newaxis]  # (H, W, 1)

        # Normalize to [0, 1] then apply ImageNet stats
        img = img.astype(np.float32) / 255.0
        if not self.grayscale:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        return tensor


class FrameStacker:
    def __init__(self, config: AgentConfig):
        self.n = config.frame_stack
        self.buffer: deque = deque(maxlen=self.n)
        self.preprocessor = Preprocessor(config)

    def reset(self, frame: np.ndarray):
        """Fill buffer with copies of the initial frame."""
        processed = self.preprocessor(frame)
        self.buffer.clear()
        for _ in range(self.n):
            self.buffer.append(processed)

    def push(self, frame: np.ndarray) -> torch.Tensor:
        """Add a new frame and return the stacked tensor (N*C, H, W)."""
        processed = self.preprocessor(frame)
        self.buffer.append(processed)
        return torch.cat(list(self.buffer), dim=0)
