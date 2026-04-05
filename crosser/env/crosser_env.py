"""Street Crosser environment with clean API."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from config import Config
from env.state import GameState, NOOP
from env.physics import create_initial_state, step_physics
from env.renderer import Renderer


@dataclass
class Observation:
    frame: np.ndarray
    info: dict


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict


class CrosserEnv:
    def __init__(self, config: Config):
        self.config = config
        self._state: Optional[GameState] = None
        self._renderer: Optional[Renderer] = None
        self._rng = np.random.RandomState()

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._state = create_initial_state(self.config, self._rng)

        if self._renderer is None:
            self._renderer = Renderer(self.config)

        self._renderer.render(self._state)
        frame = self._renderer.get_frame()
        return Observation(frame=frame, info=self._make_info())

    def step(self, action: int = NOOP) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() first")

        reward, scored, hit = step_physics(self._state, action, self.config, self._rng)

        self._renderer.render(self._state)
        frame = self._renderer.get_frame()

        return StepResult(
            observation=Observation(frame=frame, info=self._make_info()),
            reward=reward,
            done=self._state.done,
            info={"scored": scored, "hit": hit, "score": self._state.score},
        )

    def get_state(self) -> dict:
        return self._state.to_dict() if self._state else {}

    def get_frame(self) -> np.ndarray:
        return self._renderer.get_frame()

    def close(self):
        if self._renderer:
            self._renderer.close()

    def _make_info(self) -> dict:
        return {
            "score": self._state.score,
            "step": self._state.step_count,
            "hit": self._state.hit,
        }
