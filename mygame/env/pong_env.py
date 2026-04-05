"""Main Pong environment class."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from config import Config
from env.state import GameState, NOOP
from env.physics import create_initial_state, step_physics
from env.renderer import Renderer
from utils.seeding import make_rng


@dataclass
class Observation:
    frame: np.ndarray
    info: dict


@dataclass
class StepResult:
    observation: Observation
    reward_left: float
    reward_right: float
    done: bool
    truncated: bool
    info: dict


class PongEnv:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._renderer: Optional[Renderer] = None
        self._state: Optional[GameState] = None
        self._rng: Optional[np.random.RandomState] = None
        self._seed: int = 0

    def reset(self, seed: Optional[int] = None,
              config_override: Optional[dict] = None) -> Observation:
        """Reset the environment and return initial observation."""
        if config_override:
            d = self.config.to_dict()
            d.update(config_override)
            self.config = Config.from_dict(d)

        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        self._seed = seed
        self._rng = make_rng(seed)

        self._state = create_initial_state(self.config, self._rng)

        # Initialize renderer if needed
        if self._renderer is None:
            self._renderer = Renderer(self.config)

        self._renderer.render(self._state)
        frame = self._renderer.get_frame()

        return Observation(frame=frame, info=self._make_info())

    def step(self, action_left: int, action_right: int = NOOP) -> StepResult:
        """Advance one step."""
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        reward_left, reward_right, scored = step_physics(
            self._state, action_left, action_right, self.config, self._rng
        )

        self._renderer.render(self._state)
        frame = self._renderer.get_frame()

        truncated = (self.config.max_steps > 0 and
                     self._state.step_count >= self.config.max_steps and
                     not (self._state.score_left >= self.config.target_score or
                          self._state.score_right >= self.config.target_score))

        info = self._make_info()
        info["scored"] = scored

        return StepResult(
            observation=Observation(frame=frame, info=info),
            reward_left=reward_left,
            reward_right=reward_right,
            done=self._state.done,
            truncated=truncated,
            info=info,
        )

    def render(self):
        """Force a render of current state."""
        if self._renderer and self._state:
            self._renderer.render(self._state)

    def get_frame(self) -> np.ndarray:
        """Get current frame as numpy array (H, W, 3)."""
        if self._renderer is None:
            return np.zeros((self.config.render_height, self.config.render_width, 3),
                            dtype=np.uint8)
        return self._renderer.get_frame()

    def get_state(self) -> dict:
        """Get full serializable game state."""
        if self._state is None:
            return {}
        return self._state.to_dict()

    def set_state(self, state_dict: dict):
        """Restore game state from dict."""
        self._state = GameState.from_dict(state_dict)

    def get_flat_state(self) -> dict:
        """Get flat state dict for logging."""
        if self._state is None:
            return {}
        return self._state.flat_dict()

    @property
    def state(self) -> Optional[GameState]:
        return self._state

    @property
    def seed_used(self) -> int:
        return self._seed

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    def _make_info(self) -> dict:
        s = self._state
        return {
            "score_left": s.score_left,
            "score_right": s.score_right,
            "step_count": s.step_count,
            "rally_length": s.rally_length,
        }
