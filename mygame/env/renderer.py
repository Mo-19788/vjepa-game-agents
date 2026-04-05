"""Pygame renderer with windowed and offscreen modes."""

import numpy as np
from config import Config
from env.state import GameState

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class Renderer:
    def __init__(self, config: Config):
        self.config = config
        self.screen = None
        self.surface = None
        self._font = None
        self._scale_x = config.render_width / config.arena_width
        self._scale_y = config.render_height / config.arena_height

        if not PYGAME_AVAILABLE:
            if not config.headless:
                raise RuntimeError("pygame is required for windowed rendering")
            return

        if config.headless:
            # Offscreen rendering
            pygame.init()
            self.surface = pygame.Surface((config.render_width, config.render_height))
        else:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config.render_width, config.render_height)
            )
            pygame.display.set_caption("Retro Pong Game")
            self.surface = self.screen
            self._font = pygame.font.Font(None, 36)

    def render(self, state: GameState):
        """Draw the current game state."""
        if self.surface is None:
            return

        cfg = self.config
        surf = self.surface

        # Background
        surf.fill(cfg.bg_color)

        # Center line
        mid_x = int(cfg.render_width / 2)
        for y in range(0, cfg.render_height, 12):
            pygame.draw.rect(surf, cfg.line_color,
                             (mid_x - cfg.line_thickness // 2, y,
                              cfg.line_thickness, 6))

        # Paddles
        self._draw_paddle(surf, state.left_paddle, cfg.paddle_color)
        self._draw_paddle(surf, state.right_paddle, cfg.paddle_color)

        # Balls (primary + extra)
        for ball in state.all_balls:
            bx = int(ball.x * self._scale_x)
            by = int(ball.y * self._scale_y)
            bs = max(2, int(ball.size * self._scale_x))
            pygame.draw.rect(surf, cfg.ball_color,
                             (bx - bs // 2, by - bs // 2, bs, bs))

        # Don't draw score on the surface — keep frames matching headless training data
        # Score will be drawn separately after get_frame() if needed

        if self.screen is not None:
            # Show score on display only (after get_frame captures clean frame)
            if self._font:
                score_text = f"{state.score_left}  {state.score_right}"
                text_surf = self._font.render(score_text, True, (200, 200, 200))
                text_rect = text_surf.get_rect(centerx=cfg.render_width // 2, top=10)
                surf.blit(text_surf, text_rect)
            pygame.display.flip()
            # Redraw without score for clean get_frame()
            surf.fill(cfg.bg_color)
            mid_x = int(cfg.render_width / 2)
            for y in range(0, cfg.render_height, 12):
                pygame.draw.rect(surf, cfg.line_color,
                                 (mid_x - cfg.line_thickness // 2, y,
                                  cfg.line_thickness, 6))
            self._draw_paddle(surf, state.left_paddle, cfg.paddle_color)
            self._draw_paddle(surf, state.right_paddle, cfg.paddle_color)
            bx = int(state.ball.x * self._scale_x)
            by = int(state.ball.y * self._scale_y)
            bs = max(2, int(state.ball.size * self._scale_x))
            pygame.draw.rect(surf, cfg.ball_color,
                             (bx - bs // 2, by - bs // 2, bs, bs))

    def _draw_paddle(self, surf, paddle, color):
        px = int(paddle.left * self._scale_x)
        py = int(paddle.top * self._scale_y)
        pw = max(2, int(paddle.width * self._scale_x))
        ph = int(paddle.height * self._scale_y)
        pygame.draw.rect(surf, color, (px, py, pw, ph))

    def get_frame(self) -> np.ndarray:
        """Return current surface as RGB numpy array (H, W, 3)."""
        if self.surface is None:
            return np.zeros((self.config.render_height, self.config.render_width, 3),
                            dtype=np.uint8)
        arr = pygame.surfarray.array3d(self.surface)
        # pygame returns (W, H, 3), transpose to (H, W, 3)
        return arr.transpose(1, 0, 2)

    def close(self):
        if PYGAME_AVAILABLE and pygame.get_init():
            pygame.quit()
