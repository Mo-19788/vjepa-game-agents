"""Base policy interface."""

from abc import ABC, abstractmethod
from env.state import GameState


class Policy(ABC):
    @abstractmethod
    def get_action(self, state: GameState, side: str) -> int:
        """Return action for the given side ('left' or 'right')."""
        ...

    def reset(self):
        """Called at episode start."""
        pass
