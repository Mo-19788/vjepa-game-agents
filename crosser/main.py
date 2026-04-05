"""Street Crosser — Frogger-style lane crossing game.

Usage:
    python main.py                       # Play with settings panel
    python main.py --mode play           # Human plays with arrow keys
    python main.py --mode bot            # Bot plays
    python main.py --mode generate       # Generate training data
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from config import Config
from env.crosser_env import CrosserEnv
from env.state import NOOP, UP, DOWN, LEFT, RIGHT


# ── UI Helpers ────────────────────────────────────────────────────────

class Button:
    def __init__(self, x, y, w, h, text, color=(70, 70, 90),
                 hover_color=(90, 90, 120), text_color=(230, 230, 230)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self._font = None

    def draw(self, screen):
        if self._font is None:
            self._font = pygame.font.Font(None, 22)
        mouse = pygame.mouse.get_pos()
        c = self.hover_color if self.rect.collidepoint(mouse) else self.color
        pygame.draw.rect(screen, c, self.rect, border_radius=6)
        pygame.draw.rect(screen, (120, 120, 140), self.rect, 1, border_radius=6)
        label = self._font.render(self.text, True, self.text_color)
        screen.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


def play_human(config: Config):
    """Human plays with arrow keys + in-game settings panel."""
    # Use headless renderer — we manage the display ourselves
    config.headless = True

    pygame.init()
    panel_w = 220
    game_w = config.render_width
    game_h = config.render_height
    screen = pygame.display.set_mode((game_w + panel_w, game_h))
    pygame.display.set_caption("Street Crosser")

    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    big_font = pygame.font.Font(None, 36)
    title_font = pygame.font.Font(None, 28)

    # Settings state
    lanes = config.num_lanes
    cars = config.max_cars_per_lane
    car_speed = config.max_car_speed
    player_color_idx = 0
    player_colors = [
        ((0, 255, 100), "Green"),
        ((255, 80, 80), "Red"),
        ((80, 150, 255), "Blue"),
        ((255, 255, 0), "Yellow"),
        ((255, 130, 255), "Pink"),
        ((0, 255, 255), "Cyan"),
        ((255, 165, 0), "Orange"),
    ]

    def apply_settings():
        nonlocal config
        config.num_lanes = lanes
        config.max_cars_per_lane = cars
        config.max_car_speed = car_speed
        config.player_color = player_colors[player_color_idx][0]

    apply_settings()
    env = CrosserEnv(config)
    obs = env.reset(seed=config.seed)

    # Build buttons
    bw, bh = 36, 30
    px = game_w + 15  # panel x start

    btn_lanes_m = Button(0, 0, bw, bh, "-")
    btn_lanes_p = Button(0, 0, bw, bh, "+")
    btn_cars_m = Button(0, 0, bw, bh, "-")
    btn_cars_p = Button(0, 0, bw, bh, "+")
    btn_speed_m = Button(0, 0, bw, bh, "-")
    btn_speed_p = Button(0, 0, bw, bh, "+")
    btn_color = Button(0, 0, 130, bh, "Color")
    btn_restart = Button(0, 0, 190, 36, "RESTART",
                         color=(140, 50, 50), hover_color=(180, 70, 70))

    clock = pygame.time.Clock()
    running = True
    total_score = 0
    total_hits = 0
    best_score = 0
    flash_timer = 0
    flash_text = ""
    flash_color = (255, 255, 255)

    while running:
        action = NOOP

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    action = UP
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    action = DOWN
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    action = LEFT
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    action = RIGHT
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    apply_settings()
                    env = CrosserEnv(config)
                    obs = env.reset()
                    total_score = 0; total_hits = 0

            # Button clicks
            if btn_lanes_m.clicked(event):
                lanes = max(1, lanes - 1)
            elif btn_lanes_p.clicked(event):
                lanes = min(8, lanes + 1)
            elif btn_cars_m.clicked(event):
                cars = max(1, cars - 1)
            elif btn_cars_p.clicked(event):
                cars = min(5, cars + 1)
            elif btn_speed_m.clicked(event):
                car_speed = max(0.5, car_speed - 0.5)
            elif btn_speed_p.clicked(event):
                car_speed = min(8.0, car_speed + 0.5)
            elif btn_color.clicked(event):
                player_color_idx = (player_color_idx + 1) % len(player_colors)
                config.player_color = player_colors[player_color_idx][0]
                env._renderer.config.player_color = config.player_color
            elif btn_restart.clicked(event):
                apply_settings()
                env = CrosserEnv(config)
                obs = env.reset()
                total_score = 0; total_hits = 0
                flash_text = "NEW GAME!"; flash_color = (100, 255, 200); flash_timer = 60

        result = env.step(action)
        obs = result.observation

        if result.info.get("scored"):
            total_score += 1
            best_score = max(best_score, total_score)
            flash_text = f"CROSSED!  Score: {total_score}"
            flash_color = (100, 255, 150)
            flash_timer = 45
        if result.info.get("hit"):
            total_hits += 1
            flash_text = "HIT!"
            flash_color = (255, 80, 80)
            flash_timer = 30

        if result.done:
            best_score = max(best_score, total_score)
            flash_text = f"GAME OVER!  Score: {total_score}"
            flash_color = (255, 200, 50)
            flash_timer = 90
            obs = env.reset()
            total_score = 0; total_hits = 0

        # ── Draw ──────────────────────────────────────────────────

        screen.fill((25, 25, 35))

        # Game frame
        frame = obs.frame
        game_surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(game_surface, (0, 0))

        # Flash text overlay on game
        if flash_timer > 0:
            flash_timer -= 1
            alpha = min(255, flash_timer * 6)
            ft = big_font.render(flash_text, True, flash_color)
            fr = ft.get_rect(center=(game_w // 2, game_h // 2))
            bg = pygame.Surface((ft.get_width() + 20, ft.get_height() + 10))
            bg.set_alpha(min(180, alpha))
            bg.fill((0, 0, 0))
            screen.blit(bg, (fr.x - 10, fr.y - 5))
            screen.blit(ft, fr)

        # ── Settings Panel ────────────────────────────────────────

        panel_rect = pygame.Rect(game_w, 0, panel_w, game_h)
        pygame.draw.rect(screen, (35, 35, 50), panel_rect)
        pygame.draw.line(screen, (60, 60, 80), (game_w, 0), (game_w, game_h), 2)

        y = 15

        # Title
        title = title_font.render("STREET CROSSER", True, (0, 230, 180))
        screen.blit(title, (px, y)); y += 32

        # Score display
        pygame.draw.rect(screen, (25, 25, 40), (px, y, 190, 60), border_radius=8)
        score_t = font.render(f"Score: {total_score}", True, (255, 255, 255))
        screen.blit(score_t, (px + 10, y + 5))
        hits_t = small_font.render(f"Hits: {total_hits}", True, (255, 120, 120))
        screen.blit(hits_t, (px + 120, y + 7))
        best_t = small_font.render(f"Best: {best_score}", True, (255, 220, 100))
        screen.blit(best_t, (px + 10, y + 30))
        ratio = total_score / max(1, total_hits)
        ratio_t = small_font.render(f"Ratio: {ratio:.1f}", True, (150, 200, 255))
        screen.blit(ratio_t, (px + 100, y + 30))
        y += 70

        # Divider
        pygame.draw.line(screen, (60, 60, 80), (px, y), (px + 190, y)); y += 10

        # Controls label
        ctrl = small_font.render("SETTINGS  (restart to apply)", True, (180, 180, 200))
        screen.blit(ctrl, (px, y)); y += 22

        # Lanes
        lbl = font.render(f"Lanes: {lanes}", True, (220, 220, 220))
        screen.blit(lbl, (px, y + 4))
        btn_lanes_m.rect.topleft = (px + 110, y)
        btn_lanes_p.rect.topleft = (px + 152, y)
        btn_lanes_m.draw(screen); btn_lanes_p.draw(screen)
        y += 38

        # Cars per lane
        lbl = font.render(f"Cars: {cars}", True, (220, 220, 220))
        screen.blit(lbl, (px, y + 4))
        btn_cars_m.rect.topleft = (px + 110, y)
        btn_cars_p.rect.topleft = (px + 152, y)
        btn_cars_m.draw(screen); btn_cars_p.draw(screen)
        y += 38

        # Car speed
        lbl = font.render(f"Speed: {car_speed:.1f}", True, (220, 220, 220))
        screen.blit(lbl, (px, y + 4))
        btn_speed_m.rect.topleft = (px + 110, y)
        btn_speed_p.rect.topleft = (px + 152, y)
        btn_speed_m.draw(screen); btn_speed_p.draw(screen)
        y += 38

        # Player color
        color_name = player_colors[player_color_idx][1]
        color_val = player_colors[player_color_idx][0]
        lbl = font.render("Player:", True, (220, 220, 220))
        screen.blit(lbl, (px, y + 4))
        pygame.draw.circle(screen, color_val, (px + 80, y + 14), 10)
        btn_color.text = color_name
        btn_color.rect.topleft = (px + 95, y)
        btn_color.draw(screen)
        y += 42

        # Divider
        pygame.draw.line(screen, (60, 60, 80), (px, y), (px + 190, y)); y += 12

        # Restart button
        btn_restart.rect.topleft = (px, y)
        btn_restart.draw(screen)
        y += 46

        # Controls help
        pygame.draw.line(screen, (60, 60, 80), (px, y), (px + 190, y)); y += 10
        help_lbl = small_font.render("HOW TO PLAY", True, (180, 180, 200))
        screen.blit(help_lbl, (px, y)); y += 20

        controls = [
            ("Arrow keys / WASD", "Move"),
            ("R", "Restart"),
            ("ESC", "Quit"),
        ]
        for key, desc in controls:
            kt = small_font.render(key, True, (120, 200, 255))
            dt = small_font.render(f"  {desc}", True, (160, 160, 160))
            screen.blit(kt, (px + 5, y))
            screen.blit(dt, (px + 5 + kt.get_width(), y))
            y += 18

        y += 10
        tips = small_font.render("Reach the blue zone at top!", True, (100, 180, 255))
        screen.blit(tips, (px + 5, y)); y += 16
        tips2 = small_font.render("Avoid the cars!", True, (255, 130, 130))
        screen.blit(tips2, (px + 5, y))

        pygame.display.flip()
        clock.tick(config.fps)

    env.close()


def play_bot(config: Config):
    """Simple bot that tries to cross safely."""
    config.headless = False
    env = CrosserEnv(config)
    obs = env.reset(seed=config.seed)

    clock = pygame.time.Clock()
    running = True
    step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simple bot: move up if safe, dodge left/right if car nearby
        state = env._state
        player = state.player

        # Check if there's a car in the row above
        target_row = player.row - 1
        safe = True
        if target_row >= config.safe_rows:
            for car in state.cars:
                if car.row == target_row:
                    car_left = car.x
                    car_right = car.x + car.width
                    # Check if player would be hit
                    if car_left - 1.5 <= player.col <= car_right + 0.5:
                        safe = False
                        break

        if safe and step % 3 == 0:  # move up every 3 steps
            action = UP
        elif not safe:
            # Try to dodge
            action = LEFT if np.random.random() < 0.5 else RIGHT
        else:
            action = NOOP

        result = env.step(action)
        step += 1

        if result.info.get("scored"):
            print(f"  Crossed! Score: {result.info['score']}")
        if result.info.get("hit"):
            print(f"  Hit! Score: {result.info['score']}")
        if result.done:
            print(f"Done! Score: {result.info['score']}")
            obs = env.reset()
            step = 0

        clock.tick(config.fps)

    env.close()


def generate_data(config: Config, episodes: int, out_dir: str):
    """Generate training data using bot play."""
    import json
    import os

    config.headless = True
    config.save_frames = True
    config.save_actions = True
    config.save_states = True

    split_dir = Path(out_dir) / "split_train"
    split_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(split_dir.glob("ep_*")))

    for ep_idx in range(episodes):
        ep_id = f"ep_{existing + ep_idx:06d}"
        ep_dir = split_dir / ep_id
        ep_dir.mkdir(exist_ok=True)

        seed = np.random.randint(0, 999999)
        env = CrosserEnv(config)
        obs = env.reset(seed=seed)

        actions_file = open(ep_dir / "actions.jsonl", "w")
        states_file = open(ep_dir / "states.jsonl", "w")

        import cv2
        frame_idx = 0
        cv2.imwrite(str(ep_dir / f"frame_{frame_idx:06d}.png"),
                     cv2.cvtColor(obs.frame, cv2.COLOR_RGB2BGR))

        state = env._state
        states_file.write(json.dumps(state.flat_dict()) + "\n")

        while not state.done:
            # Bot logic
            player = state.player
            target_row = player.row - 1
            safe = True
            if target_row >= config.safe_rows:
                for car in state.cars:
                    if car.row == target_row:
                        car_left = car.x
                        car_right = car.x + car.width
                        if car_left - 1.5 <= player.col <= car_right + 0.5:
                            safe = False
                            break

            if safe and frame_idx % 3 == 0:
                action = UP
            elif not safe:
                action = LEFT if np.random.random() < 0.5 else RIGHT
            else:
                action = NOOP

            actions_file.write(json.dumps({"t": frame_idx, "action": action}) + "\n")

            result = env.step(action)
            state = env._state
            frame_idx += 1

            cv2.imwrite(str(ep_dir / f"frame_{frame_idx:06d}.png"),
                         cv2.cvtColor(result.observation.frame, cv2.COLOR_RGB2BGR))
            states_file.write(json.dumps(state.flat_dict()) + "\n")

        actions_file.close()
        states_file.close()

        # Metadata
        metadata = {
            "episode_id": ep_id,
            "seed": seed,
            "total_steps": frame_idx,
            "final_score": state.score,
            "frame_count": frame_idx + 1,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  {ep_id}: {frame_idx} steps, score={state.score}")

    print(f"Generated {episodes} episodes in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Street Crosser")
    parser.add_argument("--mode", default="play",
                        choices=["play", "bot", "generate"],
                        help="play | bot | generate")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--num-lanes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        config = Config.load(Path(args.config))
    else:
        config = Config()

    config.seed = args.seed
    config.fps = args.fps
    config.num_lanes = args.num_lanes
    config.max_steps = args.max_steps
    if args.headless:
        config.headless = True

    if args.mode == "play":
        play_human(config)
    elif args.mode == "bot":
        play_bot(config)
    elif args.mode == "generate":
        generate_data(config, args.episodes, args.out)


if __name__ == "__main__":
    main()
