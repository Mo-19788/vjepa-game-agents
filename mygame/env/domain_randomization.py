"""Domain randomization for visual and dynamics variation."""

import numpy as np
from config import Config


def apply_randomization(config: Config, rng: np.random.RandomState) -> Config:
    """Apply seed-controlled randomization to a config copy.

    Returns a new Config with randomized parameters.
    """
    d = config.to_dict()

    if config.randomize_visuals:
        d["bg_color"] = _random_dark_color(rng)
        d["paddle_color"] = _random_bright_color(rng)
        d["ball_color"] = _random_bright_color(rng)
        d["line_color"] = [int(rng.randint(60, 180)) for _ in range(3)]
        d["line_thickness"] = int(rng.choice([1, 2, 3, 4]))
        # Vary paddle/ball sizes within reasonable range
        d["paddle_height"] = float(config.paddle_height * rng.uniform(0.6, 1.4))
        d["ball_size"] = float(config.ball_size * rng.uniform(0.7, 1.5))

    if config.randomize_dynamics:
        d["ball_speed"] = float(config.ball_speed * rng.uniform(0.6, 1.5))
        d["paddle_speed"] = float(config.paddle_speed * rng.uniform(0.7, 1.3))

    if config.randomize_env:
        d["arena_width"] = float(config.arena_width * rng.uniform(0.8, 1.2))
        d["arena_height"] = float(config.arena_height * rng.uniform(0.8, 1.2))
        d["target_score"] = int(rng.choice([3, 5, 7, 10]))

    return Config.from_dict(d)


def _random_dark_color(rng: np.random.RandomState) -> list:
    return [int(rng.randint(0, 50)) for _ in range(3)]


def _random_bright_color(rng: np.random.RandomState) -> list:
    return [int(rng.randint(150, 256)) for _ in range(3)]
