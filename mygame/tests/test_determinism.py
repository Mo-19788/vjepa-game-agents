"""Tests for deterministic replay."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force headless pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
from config import Config
from env.pong_env import PongEnv
from env.state import NOOP
from policies.bot_policy import BotPolicy
from utils.seeding import make_rng


def test_same_seed_same_trajectory():
    """Same seed + same actions => same state trajectory."""
    config = Config(headless=True, target_score=3, max_steps=2000)

    def run_episode(seed):
        env = PongEnv(config)
        rng = make_rng(seed)
        bot_left = BotPolicy(difficulty="medium", rng=make_rng(seed + 100))
        bot_right = BotPolicy(difficulty="medium", rng=make_rng(seed + 200))
        bot_left.reset()
        bot_right.reset()

        env.reset(seed=seed)
        states = [env.get_flat_state()]
        actions = []

        while not env.state.done:
            al = bot_left.get_action(env.state, "left")
            ar = bot_right.get_action(env.state, "right")
            actions.append((al, ar))
            env.step(al, ar)
            states.append(env.get_flat_state())

        env.close()
        return states, actions

    states1, actions1 = run_episode(seed=12345)
    states2, actions2 = run_episode(seed=12345)

    assert len(states1) == len(states2), f"Length mismatch: {len(states1)} vs {len(states2)}"
    assert len(actions1) == len(actions2)

    for i, (s1, s2) in enumerate(zip(states1, states2)):
        for key in s1:
            v1, v2 = s1[key], s2[key]
            if isinstance(v1, float):
                assert abs(v1 - v2) < 1e-10, f"Mismatch at step {i}, key {key}: {v1} vs {v2}"
            else:
                assert v1 == v2, f"Mismatch at step {i}, key {key}: {v1} vs {v2}"

    print(f"PASS: test_same_seed_same_trajectory ({len(states1)} states match)")


def test_different_seed_different_trajectory():
    """Different seeds should produce different trajectories."""
    config = Config(headless=True, target_score=2, max_steps=1000)

    def get_first_states(seed, n=20):
        env = PongEnv(config)
        env.reset(seed=seed)
        states = []
        for _ in range(n):
            env.step(NOOP, NOOP)
            states.append(env.get_flat_state())
        env.close()
        return states

    states_a = get_first_states(111)
    states_b = get_first_states(222)

    differs = False
    for sa, sb in zip(states_a, states_b):
        if abs(sa["ball_x"] - sb["ball_x"]) > 0.01:
            differs = True
            break

    assert differs, "Different seeds should produce different ball positions"
    print("PASS: test_different_seed_different_trajectory")


if __name__ == "__main__":
    test_same_seed_same_trajectory()
    test_different_seed_different_trajectory()
    print("\nAll determinism tests passed!")
