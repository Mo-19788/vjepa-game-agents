"""Random and constrained-random policies."""

import numpy as np
from env.state import GameState, NOOP, UP, DOWN
from policies.base import Policy

ACTIONS = [NOOP, UP, DOWN]


class RandomPolicy(Policy):
    """Uniform random action each step."""

    def __init__(self, rng: np.random.RandomState = None):
        self.rng = rng or np.random.RandomState()

    def get_action(self, state: GameState, side: str) -> int:
        return int(self.rng.choice(ACTIONS))


class StickyRandomPolicy(Policy):
    """Random action that persists for several frames before changing."""

    def __init__(self, stick_frames: int = 8, rng: np.random.RandomState = None):
        self.stick_frames = stick_frames
        self.rng = rng or np.random.RandomState()
        self._current_action = NOOP
        self._frames_left = 0

    def reset(self):
        self._current_action = NOOP
        self._frames_left = 0

    def get_action(self, state: GameState, side: str) -> int:
        if self._frames_left <= 0:
            self._current_action = int(self.rng.choice(ACTIONS))
            self._frames_left = self.stick_frames
        self._frames_left -= 1
        return self._current_action


class ConstrainedRandomPolicy(Policy):
    """Random but biased toward the ball — mixes tracking with randomness."""

    def __init__(self, tracking_prob: float = 0.6, rng: np.random.RandomState = None):
        self.tracking_prob = tracking_prob
        self.rng = rng or np.random.RandomState()

    def get_action(self, state: GameState, side: str) -> int:
        if self.rng.random() < self.tracking_prob:
            # Track ball
            paddle = state.left_paddle if side == "left" else state.right_paddle
            diff = state.ball.y - paddle.y
            if diff < -2:
                return UP
            elif diff > 2:
                return DOWN
            return NOOP
        return int(self.rng.choice(ACTIONS))
