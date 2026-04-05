"""Tests for policies."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import Config
from env.state import GameState, Ball, Paddle, NOOP, UP, DOWN
from env.physics import create_initial_state
from policies.bot_policy import BotPolicy, PerfectTrackingPolicy
from policies.random_policy import RandomPolicy, StickyRandomPolicy, ConstrainedRandomPolicy
from utils.seeding import make_rng

VALID_ACTIONS = {NOOP, UP, DOWN}


def make_state():
    config = Config(headless=True)
    rng = np.random.RandomState(42)
    return create_initial_state(config, rng)


def test_bot_valid_actions():
    state = make_state()
    rng = make_rng(42)
    for difficulty in ["easy", "medium", "hard"]:
        bot = BotPolicy(difficulty=difficulty, rng=rng)
        for _ in range(100):
            action = bot.get_action(state, "left")
            assert action in VALID_ACTIONS, f"Bot ({difficulty}) gave invalid action: {action}"
    print("PASS: test_bot_valid_actions")


def test_random_valid_actions():
    state = make_state()
    rng = make_rng(42)
    for PolicyCls in [RandomPolicy, StickyRandomPolicy, ConstrainedRandomPolicy]:
        policy = PolicyCls(rng=rng)
        for _ in range(200):
            action = policy.get_action(state, "right")
            assert action in VALID_ACTIONS, f"{PolicyCls.__name__} gave invalid action: {action}"
    print("PASS: test_random_valid_actions")


def test_perfect_tracking():
    state = make_state()
    state.ball.y = 100
    state.left_paddle.y = 200
    policy = PerfectTrackingPolicy()
    action = policy.get_action(state, "left")
    assert action == UP, "Should move up toward ball"

    state.ball.y = 300
    action = policy.get_action(state, "left")
    assert action == DOWN, "Should move down toward ball"
    print("PASS: test_perfect_tracking")


def test_sticky_random_persistence():
    state = make_state()
    rng = make_rng(42)
    policy = StickyRandomPolicy(stick_frames=5, rng=rng)
    policy.reset()

    # First action should persist for 5 frames
    first_action = policy.get_action(state, "left")
    for _ in range(4):
        action = policy.get_action(state, "left")
        assert action == first_action, "Sticky action should persist"
    print("PASS: test_sticky_random_persistence")


if __name__ == "__main__":
    test_bot_valid_actions()
    test_random_valid_actions()
    test_perfect_tracking()
    test_sticky_random_persistence()
    print("\nAll policy tests passed!")
