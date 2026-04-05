"""Configuration for the Street Crosser game."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Display
    render_width: int = 512
    render_height: int = 512
    fps: int = 60
    headless: bool = False

    # Grid / world
    grid_cols: int = 12      # horizontal cells
    grid_rows: int = 12      # vertical cells (top = goal, bottom = start)
    cell_size: int = 42      # pixels per cell (42*12 ≈ 512)

    # Player
    player_speed: float = 1.0  # cells per move (0.5 = smooth, 1.0 = grid-snapping)
    player_size: int = 36      # pixels — big and visible
    player_color: tuple = (0, 255, 100)

    # Lanes
    num_lanes: int = 6       # number of car lanes
    safe_rows: int = 2       # safe rows at top and bottom

    # Cars
    min_cars_per_lane: int = 1
    max_cars_per_lane: int = 3
    min_car_speed: float = 1.0
    max_car_speed: float = 4.0
    car_width: int = 2       # cells wide
    car_height: int = 1      # cells tall (1 = car, 2 = van, 3+ = truck)
    car_colors: tuple = ((255, 60, 60), (60, 100, 255), (255, 220, 0), (255, 100, 220))

    # Layout
    free_roam: bool = False  # cars placed on random rows, not fixed lanes

    # Episode
    max_steps: int = 1000
    target_score: int = 999

    # Background colors
    road_color: tuple = (60, 60, 65)
    lane_line_color: tuple = (100, 100, 90)
    safe_zone_color: tuple = (30, 100, 40)
    goal_color: tuple = (30, 40, 120)

    # Policies
    policy: str = "human"

    # Logging
    save_frames: bool = False
    save_actions: bool = False
    save_states: bool = False
    save_dir: str = "dataset"
    seed: Optional[int] = None

    # Version
    version: str = "0.1.0"

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("player_color", "road_color", "lane_line_color",
                     "safe_zone_color", "goal_color", "car_colors"):
            val = d[key]
            if isinstance(val, tuple) and val and isinstance(val[0], tuple):
                d[key] = [list(c) for c in val]
            elif isinstance(val, tuple):
                d[key] = list(val)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        for key in ("player_color", "road_color", "lane_line_color",
                     "safe_zone_color", "goal_color"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        if "car_colors" in d and isinstance(d[key], list):
            d["car_colors"] = tuple(tuple(c) for c in d["car_colors"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))
