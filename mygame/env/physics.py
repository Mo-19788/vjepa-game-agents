"""Deterministic 2D physics for Pong."""

import math
import numpy as np
from config import Config
from env.state import Ball, Paddle, GameState, UP, DOWN, NOOP


def create_initial_state(config: Config, rng: np.random.RandomState) -> GameState:
    """Create a fresh game state with ball(s) served from center."""
    cx = config.arena_width / 2
    cy = config.arena_height / 2

    ball = _serve_ball(cx, cy, config.ball_size, config.ball_speed, rng)

    left_paddle = Paddle(
        x=config.paddle_margin + config.paddle_width / 2,
        y=cy,
        width=config.paddle_width,
        height=config.paddle_height,
        speed=config.paddle_speed,
    )
    right_paddle = Paddle(
        x=config.arena_width - config.paddle_margin - config.paddle_width / 2,
        y=cy,
        width=config.paddle_width,
        height=config.paddle_height,
        speed=config.paddle_speed,
    )

    extra_balls = []
    for _ in range(config.num_balls - 1):
        extra_balls.append(_serve_ball(cx, cy, config.ball_size, config.ball_speed, rng))

    return GameState(ball=ball, left_paddle=left_paddle, right_paddle=right_paddle,
                     extra_balls=extra_balls)


def _serve_ball(cx: float, cy: float, size: float, speed: float,
                rng: np.random.RandomState) -> Ball:
    """Serve ball from center with random angle."""
    # Random angle between -45 and 45 degrees, randomly left or right
    angle = rng.uniform(-math.pi / 4, math.pi / 4)
    direction = rng.choice([-1, 1])
    vx = speed * math.cos(angle) * direction
    vy = speed * math.sin(angle)
    return Ball(x=cx, y=cy, vx=vx, vy=vy, size=size)


def _step_single_ball(ball: Ball, state: GameState, config: Config,
                      rng: np.random.RandomState) -> tuple:
    """Process one ball's physics. Returns (reward_left, reward_right, scored)."""
    lp = state.left_paddle
    rp = state.right_paddle

    ball.x += ball.vx
    ball.y += ball.vy

    half = ball.size / 2
    if ball.y - half <= 0:
        ball.y = half
        ball.vy = abs(ball.vy)
    elif ball.y + half >= config.arena_height:
        ball.y = config.arena_height - half
        ball.vy = -abs(ball.vy)

    _check_paddle_collision(ball, lp, direction=1)
    _check_paddle_collision(ball, rp, direction=-1)

    reward_left = 0.0
    reward_right = 0.0
    scored = False

    if ball.x - half <= 0:
        state.score_right += 1
        reward_right = 1.0
        reward_left = -1.0
        scored = True
        _reset_ball(ball, config, rng)
    elif ball.x + half >= config.arena_width:
        state.score_left += 1
        reward_left = 1.0
        reward_right = -1.0
        scored = True
        _reset_ball(ball, config, rng)

    return reward_left, reward_right, scored


def step_physics(state: GameState, action_left: int, action_right: int,
                 config: Config, rng: np.random.RandomState) -> tuple:
    """
    Advance one physics step. Returns (reward_left, reward_right, scored).
    Mutates state in place.
    """
    # Move paddles
    _move_paddle(state.left_paddle, action_left, config)
    _move_paddle(state.right_paddle, action_right, config)

    # Process all balls
    total_rl = 0.0
    total_rr = 0.0
    any_scored = False

    for ball in state.all_balls:
        rl, rr, scored = _step_single_ball(ball, state, config, rng)
        total_rl += rl
        total_rr += rr
        if scored:
            any_scored = True

    if any_scored:
        state.rally_length = 0
    else:
        state.rally_length += 1

    state.step_count += 1

    # Check episode end
    if state.score_left >= config.target_score or state.score_right >= config.target_score:
        state.done = True
    if config.max_steps > 0 and state.step_count >= config.max_steps:
        state.done = True
    if config.max_rally_length > 0 and state.rally_length >= config.max_rally_length:
        state.done = True

    return total_rl, total_rr, any_scored


def _move_paddle(paddle: Paddle, action: int, config: Config):
    """Move paddle based on action, clamped to arena."""
    if action == UP:
        paddle.y -= paddle.speed
    elif action == DOWN:
        paddle.y += paddle.speed

    half_h = paddle.height / 2
    paddle.y = max(half_h, min(config.arena_height - half_h, paddle.y))


def _check_paddle_collision(ball: Ball, paddle: Paddle, direction: int):
    """
    Check and resolve ball-paddle collision.
    direction: 1 for left paddle (ball bounces right), -1 for right paddle (ball bounces left).
    """
    half = ball.size / 2

    # Check overlap
    if (ball.x - half <= paddle.right and
        ball.x + half >= paddle.left and
        ball.y + half >= paddle.top and
        ball.y - half <= paddle.bottom):

        # Only bounce if ball is moving toward the paddle
        if direction == 1 and ball.vx < 0:
            ball.x = paddle.right + half
            # Vary vertical velocity based on hit position
            offset = (ball.y - paddle.y) / (paddle.height / 2)
            offset = max(-1.0, min(1.0, offset))
            speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)
            angle = offset * (math.pi / 4)
            ball.vx = speed * math.cos(angle)
            ball.vy = speed * math.sin(angle)
        elif direction == -1 and ball.vx > 0:
            ball.x = paddle.left - half
            offset = (ball.y - paddle.y) / (paddle.height / 2)
            offset = max(-1.0, min(1.0, offset))
            speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)
            angle = offset * (math.pi / 4)
            ball.vx = -speed * math.cos(angle)
            ball.vy = speed * math.sin(angle)


def _reset_ball(ball: Ball, config: Config, rng: np.random.RandomState):
    """Reset ball to center with new serve direction."""
    cx = config.arena_width / 2
    cy = config.arena_height / 2
    new_ball = _serve_ball(cx, cy, ball.size, config.ball_speed, rng)
    ball.x = new_ball.x
    ball.y = new_ball.y
    ball.vx = new_ball.vx
    ball.vy = new_ball.vy
