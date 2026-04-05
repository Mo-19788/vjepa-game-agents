"""Human keyboard policy using pygame."""

from env.state import GameState, NOOP, UP, DOWN
from policies.base import Policy


class HumanPolicy(Policy):
    """Keyboard-controlled policy.

    Left paddle: W/S
    Right paddle: UP/DOWN
    """

    def get_action(self, state: GameState, side: str) -> int:
        import pygame
        keys = pygame.key.get_pressed()
        if side == "left":
            if keys[pygame.K_w]:
                return UP
            elif keys[pygame.K_s]:
                return DOWN
        else:
            if keys[pygame.K_UP]:
                return UP
            elif keys[pygame.K_DOWN]:
                return DOWN
        return NOOP
