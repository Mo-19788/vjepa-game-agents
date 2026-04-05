"""Game state dataclasses."""

from dataclasses import dataclass
import numpy as np


# Actions
NOOP = 0
UP = 1
DOWN = 2


@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    size: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy, "size": self.size}

    @classmethod
    def from_dict(cls, d: dict) -> "Ball":
        return cls(**d)


@dataclass
class Paddle:
    x: float
    y: float
    width: float
    height: float
    speed: float

    @property
    def top(self) -> float:
        return self.y - self.height / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height / 2

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width,
                "height": self.height, "speed": self.speed}

    @classmethod
    def from_dict(cls, d: dict) -> "Paddle":
        return cls(**d)


from typing import List


@dataclass
class GameState:
    ball: Ball
    left_paddle: Paddle
    right_paddle: Paddle
    score_left: int = 0
    score_right: int = 0
    step_count: int = 0
    rally_length: int = 0
    done: bool = False
    extra_balls: List[Ball] = None  # additional balls for multi-ball mode

    def __post_init__(self):
        if self.extra_balls is None:
            self.extra_balls = []

    @property
    def all_balls(self) -> List[Ball]:
        """All balls including the primary one."""
        return [self.ball] + self.extra_balls

    def to_dict(self) -> dict:
        d = {
            "ball": self.ball.to_dict(),
            "left_paddle": self.left_paddle.to_dict(),
            "right_paddle": self.right_paddle.to_dict(),
            "score_left": self.score_left,
            "score_right": self.score_right,
            "step_count": self.step_count,
            "rally_length": self.rally_length,
            "done": self.done,
        }
        if self.extra_balls:
            d["extra_balls"] = [b.to_dict() for b in self.extra_balls]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        extra = [Ball.from_dict(b) for b in d.get("extra_balls", [])]
        return cls(
            ball=Ball.from_dict(d["ball"]),
            left_paddle=Paddle.from_dict(d["left_paddle"]),
            right_paddle=Paddle.from_dict(d["right_paddle"]),
            score_left=d["score_left"],
            score_right=d["score_right"],
            step_count=d["step_count"],
            rally_length=d["rally_length"],
            done=d["done"],
            extra_balls=extra,
        )

    def flat_dict(self) -> dict:
        """Flat dictionary for states.jsonl logging."""
        d = {
            "t": self.step_count,
            "ball_x": self.ball.x,
            "ball_y": self.ball.y,
            "ball_vx": self.ball.vx,
            "ball_vy": self.ball.vy,
            "left_paddle_y": self.left_paddle.y,
            "right_paddle_y": self.right_paddle.y,
            "score_left": self.score_left,
            "score_right": self.score_right,
            "done": self.done,
        }
        for i, b in enumerate(self.extra_balls):
            d[f"ball{i+2}_x"] = b.x
            d[f"ball{i+2}_y"] = b.y
        return d
