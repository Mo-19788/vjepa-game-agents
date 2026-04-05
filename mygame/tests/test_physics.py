"""Tests for physics engine."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import Config
from env.state import UP, DOWN, NOOP
from env.physics import create_initial_state, step_physics, _move_paddle


def make_config(**kwargs):
    return Config(headless=True, **kwargs)


def test_initial_state():
    config = make_config()
    rng = np.random.RandomState(42)
    state = create_initial_state(config, rng)
    assert state.ball.x == config.arena_width / 2
    assert state.ball.y == config.arena_height / 2
    assert state.score_left == 0
    assert state.score_right == 0
    assert state.step_count == 0
    assert not state.done
    print("PASS: test_initial_state")


def test_wall_bounce():
    config = make_config()
    rng = np.random.RandomState(42)
    state = create_initial_state(config, rng)
    # Force ball near top wall moving upward
    state.ball.y = 5
    state.ball.vy = -10
    step_physics(state, NOOP, NOOP, config, rng)
    assert state.ball.vy > 0, "Ball should bounce off top wall"
    print("PASS: test_wall_bounce")


def test_wall_bounce_bottom():
    config = make_config()
    rng = np.random.RandomState(42)
    state = create_initial_state(config, rng)
    state.ball.y = config.arena_height - 5
    state.ball.vy = 10
    step_physics(state, NOOP, NOOP, config, rng)
    assert state.ball.vy < 0, "Ball should bounce off bottom wall"
    print("PASS: test_wall_bounce_bottom")


def test_paddle_clamp():
    config = make_config()
    rng = np.random.RandomState(42)
    state = create_initial_state(config, rng)
    # Move paddle way up
    for _ in range(1000):
        _move_paddle(state.left_paddle, UP, config)
    half_h = state.left_paddle.height / 2
    assert state.left_paddle.y >= half_h, "Paddle should be clamped at top"

    # Move paddle way down
    for _ in range(1000):
        _move_paddle(state.left_paddle, DOWN, config)
    assert state.left_paddle.y <= config.arena_height - half_h, "Paddle should be clamped at bottom"
    print("PASS: test_paddle_clamp")


def test_scoring_right():
    config = make_config()
    rng = np.random.RandomState(42)
    state = create_initial_state(config, rng)
    # Force ball to exit left
    state.ball.x = 2
    state.ball.vx = -10
    r_left, r_right, scored = step_physics(state, NOOP, NOOP, config, rng)
    assert state.score_right == 1
    assert r_right == 1.0
    assert r_left == -1.0
    assert scored
    print("PASS: test_scoring_right")


def test_scoring_left():
    config = make_config()
    rng = np.random.RandomState(42)
    state = create_initial_state(config, rng)
    state.ball.x = config.arena_width - 2
    state.ball.vx = 10
    r_left, r_right, scored = step_physics(state, NOOP, NOOP, config, rng)
    assert state.score_left == 1
    assert r_left == 1.0
    assert r_right == -1.0
    assert scored
    print("PASS: test_scoring_left")


def test_episode_ends_on_target_score():
    config = make_config(target_score=1)
    rng = np.random.RandomState(42)
    state = create_initial_state(config, rng)
    state.ball.x = 2
    state.ball.vx = -10
    step_physics(state, NOOP, NOOP, config, rng)
    assert state.done
    print("PASS: test_episode_ends_on_target_score")


def test_paddle_bounce():
    config = make_config()
    rng = np.random.RandomState(42)
    state = create_initial_state(config, rng)
    # Position ball just in front of left paddle moving left
    lp = state.left_paddle
    state.ball.x = lp.right + state.ball.size / 2 + 1
    state.ball.y = lp.y
    state.ball.vx = -5
    state.ball.vy = 0
    old_vx = state.ball.vx
    step_physics(state, NOOP, NOOP, config, rng)
    assert state.ball.vx > 0, "Ball should bounce off left paddle"
    print("PASS: test_paddle_bounce")


if __name__ == "__main__":
    test_initial_state()
    test_wall_bounce()
    test_wall_bounce_bottom()
    test_paddle_clamp()
    test_scoring_right()
    test_scoring_left()
    test_episode_ends_on_target_score()
    test_paddle_bounce()
    print("\nAll physics tests passed!")
