"""Configuration system for the Pong environment."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Display
    render_width: int = 256
    render_height: int = 256
    fps: int = 60
    headless: bool = False

    # Arena (internal coordinates)
    arena_width: float = 640.0
    arena_height: float = 480.0

    # Ball
    ball_size: float = 14.0
    ball_speed: float = 5.0
    num_balls: int = 1

    # Paddles
    paddle_width: float = 10.0
    paddle_height: float = 60.0
    paddle_speed: float = 5.0
    paddle_margin: float = 20.0  # distance from wall

    # Episode
    target_score: int = 5
    max_steps: int = 10000
    max_rally_length: int = 0  # 0 = unlimited

    # Policies
    policy_left: str = "human"
    policy_right: str = "bot"
    bot_difficulty: str = "medium"  # easy, medium, hard

    # Logging
    save_frames: bool = False
    save_actions: bool = False
    save_states: bool = False
    save_dir: str = "dataset"

    # Seed
    seed: Optional[int] = None

    # Domain randomization
    randomize_visuals: bool = False
    randomize_dynamics: bool = False
    randomize_env: bool = False

    # Visual overrides (used by domain randomization)
    bg_color: tuple = (0, 0, 0)
    paddle_color: tuple = (255, 255, 255)
    ball_color: tuple = (255, 255, 255)
    line_color: tuple = (128, 128, 128)
    line_thickness: int = 2

    # Version
    version: str = "0.1.0"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuples to lists for JSON
        for key in ("bg_color", "paddle_color", "ball_color", "line_color"):
            d[key] = list(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        # Convert lists back to tuples
        for key in ("bg_color", "paddle_color", "ball_color", "line_color"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))
