"""Heuristic bot policies with difficulty levels."""

import math
import numpy as np
from env.state import GameState, NOOP, UP, DOWN
from policies.base import Policy


class BotPolicy(Policy):
    """Tracks ball y-position with configurable skill."""

    DIFFICULTY_SETTINGS = {
        "easy":   {"reaction_lag": 8, "noise_std": 3.0, "speed_factor": 0.6},
        "medium": {"reaction_lag": 3, "noise_std": 1.0, "speed_factor": 0.85},
        "hard":   {"reaction_lag": 0, "noise_std": 0.0, "speed_factor": 1.0},
    }

    def __init__(self, difficulty: str = "medium", rng: np.random.RandomState = None):
        settings = self.DIFFICULTY_SETTINGS.get(difficulty, self.DIFFICULTY_SETTINGS["medium"])
        self.reaction_lag = settings["reaction_lag"]
        self.noise_std = settings["noise_std"]
        self.speed_factor = settings["speed_factor"]
        self.rng = rng or np.random.RandomState()
        self._lag_counter = 0
        self._last_target = None

    def reset(self):
        self._lag_counter = 0
        self._last_target = None

    def get_action(self, state: GameState, side: str) -> int:
        paddle = state.left_paddle if side == "left" else state.right_paddle
        ball = state.ball

        # Update target with lag
        self._lag_counter += 1
        if self._lag_counter >= self.reaction_lag or self._last_target is None:
            self._last_target = ball.y
            if self.noise_std > 0:
                self._last_target += self.rng.normal(0, self.noise_std)
            self._lag_counter = 0

        target_y = self._last_target
        diff = target_y - paddle.y
        threshold = paddle.speed * self.speed_factor

        if diff < -threshold:
            return UP
        elif diff > threshold:
            return DOWN
        return NOOP


class PerfectTrackingPolicy(Policy):
    """Always moves toward ball y with no lag."""

    def get_action(self, state: GameState, side: str) -> int:
        paddle = state.left_paddle if side == "left" else state.right_paddle
        diff = state.ball.y - paddle.y
        if diff < -1:
            return UP
        elif diff > 1:
            return DOWN
        return NOOP


class DelayedTrackingPolicy(Policy):
    """Tracks ball with fixed frame delay."""

    def __init__(self, delay: int = 10):
        self.delay = delay
        self._history = []

    def reset(self):
        self._history = []

    def get_action(self, state: GameState, side: str) -> int:
        self._history.append(state.ball.y)
        paddle = state.left_paddle if side == "left" else state.right_paddle

        idx = max(0, len(self._history) - self.delay)
        target_y = self._history[idx]
        diff = target_y - paddle.y
        if diff < -1:
            return UP
        elif diff > 1:
            return DOWN
        return NOOP


class OscillatoryPolicy(Policy):
    """Moves up and down in a sine pattern."""

    def __init__(self, period: int = 120, amplitude: float = 100.0):
        self.period = period
        self.amplitude = amplitude
        self._step = 0

    def reset(self):
        self._step = 0

    def get_action(self, state: GameState, side: str) -> int:
        paddle = state.left_paddle if side == "left" else state.right_paddle
        center = 240.0  # rough arena center
        target = center + self.amplitude * math.sin(2 * math.pi * self._step / self.period)
        self._step += 1
        diff = target - paddle.y
        if diff < -1:
            return UP
        elif diff > 1:
            return DOWN
        return NOOP


class WeakDefensePolicy(Policy):
    """Intentionally slow and inaccurate — for data diversity."""

    def __init__(self, rng: np.random.RandomState = None):
        self.rng = rng or np.random.RandomState()

    def get_action(self, state: GameState, side: str) -> int:
        # 50% chance of doing nothing
        if self.rng.random() < 0.5:
            return NOOP
        paddle = state.left_paddle if side == "left" else state.right_paddle
        diff = state.ball.y - paddle.y
        if diff < -5:
            return UP
        elif diff > 5:
            return DOWN
        return NOOP
