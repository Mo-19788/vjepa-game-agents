"""Game state for Street Crosser."""

from dataclasses import dataclass, field
from typing import List

# Actions
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


@dataclass
class Car:
    x: float          # horizontal position (in cells, can be fractional)
    row: int           # which lane row (top row of the car)
    speed: float       # cells per step (positive = right, negative = left)
    width: int         # cells wide
    color: tuple
    height: int = 1   # cells tall (1 = normal, 2+ = truck/bus)

    def to_dict(self) -> dict:
        return {"x": self.x, "row": self.row, "speed": self.speed,
                "width": self.width, "height": self.height,
                "color": list(self.color)}


@dataclass
class Player:
    col: float         # grid column (fractional for smooth movement)
    row: float         # grid row (fractional for smooth movement)

    def to_dict(self) -> dict:
        return {"col": self.col, "row": self.row}


@dataclass
class GameState:
    player: Player
    cars: List[Car]
    score: int = 0           # successful crossings
    step_count: int = 0
    done: bool = False
    hit: bool = False        # player was hit this step

    def to_dict(self) -> dict:
        return {
            "player": self.player.to_dict(),
            "cars": [c.to_dict() for c in self.cars],
            "score": self.score,
            "step_count": self.step_count,
            "done": self.done,
            "hit": self.hit,
        }

    def flat_dict(self) -> dict:
        return {
            "t": self.step_count,
            "player_col": self.player.col,
            "player_row": self.player.row,
            "score": self.score,
            "done": self.done,
            "hit": self.hit,
            "num_cars": len(self.cars),
        }
