"""Seeding utilities for deterministic behavior."""

import random
import numpy as np
from typing import Optional


def seed_all(seed: Optional[int] = None) -> int:
    """Seed all RNGs and return the seed used."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def make_rng(seed: int) -> np.random.RandomState:
    """Create an independent RandomState for isolated randomness."""
    return np.random.RandomState(seed)
