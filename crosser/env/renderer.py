"""Pygame renderer for Street Crosser."""

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
        self.cell = config.cell_size

        if not PYGAME_AVAILABLE:
            if not config.headless:
                raise RuntimeError("pygame required")
            return

        if config.headless:
            pygame.init()
            self.surface = pygame.Surface((config.render_width, config.render_height))
        else:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config.render_width, config.render_height)
            )
            pygame.display.set_caption("Street Crosser")
            self.surface = self.screen
            self._font = pygame.font.Font(None, 20)

    def render(self, state: GameState):
        if self.surface is None:
            return

        cfg = self.config
        surf = self.surface
        cell = self.cell

        # Background — safe zones and road
        for row in range(cfg.grid_rows):
            if row < cfg.safe_rows:
                color = cfg.goal_color
            elif row >= cfg.grid_rows - cfg.safe_rows:
                color = cfg.safe_zone_color
            else:
                color = cfg.road_color
            pygame.draw.rect(surf, color, (0, row * cell, cfg.render_width, cell))

        # Lane dividers
        for row in range(cfg.safe_rows, cfg.grid_rows - cfg.safe_rows):
            y = row * cell
            pygame.draw.line(surf, cfg.lane_line_color, (0, y), (cfg.render_width, y), 1)

        # Cars — drawn as car shapes with body + windows + wheels
        for car in state.cars:
            self._draw_car(surf, car, cell, cfg)

        # Player — drawn as a circle (distinct from rectangular cars)
        px = int(state.player.col * cell + cell // 2)
        py = int(state.player.row * cell + cell // 2)
        radius = cfg.player_size // 2
        pygame.draw.circle(surf, cfg.player_color, (px, py), radius)
        # Inner highlight
        pygame.draw.circle(surf, (min(255, cfg.player_color[0] + 60),
                                   min(255, cfg.player_color[1] + 60),
                                   min(255, cfg.player_color[2] + 60)),
                           (px - 2, py - 2), radius // 3)

        # Score and status — only on display, not in get_frame()
        if self.screen is not None:
            pygame.display.flip()
            # Draw score overlay after flip
            if self._font:
                score_text = self._font.render(f"Score: {state.score}", True, (255, 255, 255))
                self.screen.blit(score_text, (5, 5))
                if state.hit:
                    hit_text = self._font.render("HIT!", True, (255, 0, 0))
                    self.screen.blit(hit_text, (cfg.render_width - 50, 5))
                pygame.display.flip()
                # Redraw clean frame for get_frame (without score)
                self._redraw_clean(state)

    def _draw_car(self, surf, car, cell, cfg):
        """Draw a car with body, roof, windows, and wheels."""
        cx = int(car.x * cell)
        cy = car.row * cell
        cw = car.width * cell
        ch = car.height * cell
        margin = 3
        going_right = car.speed > 0

        def _draw_one(x_offset):
            bx = cx + x_offset
            # Body
            body_rect = (bx + margin, cy + margin + 4, cw - margin * 2, ch - margin * 2 - 4)
            pygame.draw.rect(surf, car.color, body_rect, border_radius=6)

            # Roof (darker, smaller rectangle on top)
            roof_color = tuple(max(0, c - 40) for c in car.color)
            roof_w = cw // 2
            roof_x = bx + cw // 4
            pygame.draw.rect(surf, roof_color,
                             (roof_x, cy + margin, roof_w, ch // 3), border_radius=4)

            # Windows (dark blue)
            win_color = (30, 30, 60)
            win_w = max(4, cw // 6)
            win_h = max(4, ch // 4)
            win_y = cy + margin + 2
            # Front window
            front_x = bx + cw - margin - win_w - 4 if going_right else bx + margin + 4
            pygame.draw.rect(surf, win_color, (front_x, win_y, win_w, win_h), border_radius=2)
            # Rear window
            rear_x = bx + margin + 4 if going_right else bx + cw - margin - win_w - 4
            pygame.draw.rect(surf, win_color, (rear_x, win_y, win_w, win_h), border_radius=2)

            # Wheels (dark circles)
            wheel_color = (30, 30, 30)
            wheel_r = max(3, ch // 8)
            wheel_y = cy + ch - margin - 1
            pygame.draw.circle(surf, wheel_color, (bx + margin + wheel_r + 2, wheel_y), wheel_r)
            pygame.draw.circle(surf, wheel_color, (bx + cw - margin - wheel_r - 2, wheel_y), wheel_r)

            # Headlight
            light_color = (255, 255, 200)
            light_x = bx + cw - margin - 2 if going_right else bx + margin
            pygame.draw.rect(surf, light_color,
                             (light_x, cy + ch // 2 - 2, 3, 4))

        _draw_one(0)
        # Wrap around
        if cx + cw > cfg.render_width:
            _draw_one(-cfg.render_width)
        if cx < 0:
            _draw_one(cfg.render_width)

    def _redraw_clean(self, state: GameState):
        """Redraw without score text for get_frame()."""
        cfg = self.config
        surf = self.surface
        cell = self.cell

        for row in range(cfg.grid_rows):
            if row < cfg.safe_rows:
                color = cfg.goal_color
            elif row >= cfg.grid_rows - cfg.safe_rows:
                color = cfg.safe_zone_color
            else:
                color = cfg.road_color
            pygame.draw.rect(surf, color, (0, row * cell, cfg.render_width, cell))

        for row in range(cfg.safe_rows, cfg.grid_rows - cfg.safe_rows):
            y = row * cell
            pygame.draw.line(surf, cfg.lane_line_color, (0, y), (cfg.render_width, y), 1)

        for car in state.cars:
            self._draw_car(surf, car, cell, cfg)

        px = state.player.col * cell + cell // 2
        py = state.player.row * cell + cell // 2
        radius = cfg.player_size // 2
        pygame.draw.circle(surf, cfg.player_color, (px, py), radius)
        pygame.draw.circle(surf, (min(255, cfg.player_color[0] + 60),
                                   min(255, cfg.player_color[1] + 60),
                                   min(255, cfg.player_color[2] + 60)),
                           (px - 2, py - 2), radius // 3)

    def get_frame(self) -> np.ndarray:
        if self.surface is None:
            return np.zeros((self.config.render_height, self.config.render_width, 3), dtype=np.uint8)
        arr = pygame.surfarray.array3d(self.surface)
        return arr.transpose(1, 0, 2)

    def close(self):
        if PYGAME_AVAILABLE and pygame.get_init():
            pygame.quit()
