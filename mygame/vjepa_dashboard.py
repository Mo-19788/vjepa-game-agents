"""V-JEPA Pong Dashboard — full control panel with real-time visualization.

Controls:
- Planning horizon slider
- Bot difficulty selector
- Ball/paddle speed sliders
- Start/Stop/Reset buttons
- Mode toggle: Planning vs Reactive
- Think frequency slider
"""

import sys
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from env.pong_env import PongEnv
from policies.bot_policy import BotPolicy
from utils.seeding import make_rng

from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.dynamics import DynamicsPredictor
from game_agent.models.reward_head import RewardHead
from game_agent.planning.shooting import PositionProbe
from game_agent.preprocessing.transforms import Preprocessor

import pygame

# Colors
BG = (20, 20, 30)
PANEL_BG = (30, 30, 45)
ACCENT = (0, 255, 200)
YELLOW = (255, 200, 0)
WHITE = (200, 200, 200)
GRAY = (120, 120, 120)
DARK_GRAY = (60, 60, 60)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (100, 200, 255)
ORANGE = (255, 150, 100)


class Slider:
    def __init__(self, x, y, w, min_val, max_val, val, label, step=1, fmt="{:.0f}"):
        self.rect = pygame.Rect(x, y, w, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.val = val
        self.label = label
        self.step = step
        self.fmt = fmt
        self.dragging = False

    def draw(self, screen, font):
        # Label
        text = font.render(f"{self.label}: {self.fmt.format(self.val)}", True, WHITE)
        screen.blit(text, (self.rect.x, self.rect.y - 14))
        # Track
        pygame.draw.rect(screen, DARK_GRAY, self.rect, border_radius=3)
        # Fill
        pct = (self.val - self.min_val) / max(self.max_val - self.min_val, 0.001)
        fill_w = int(pct * self.rect.w)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.h)
        pygame.draw.rect(screen, ACCENT, fill_rect, border_radius=3)
        # Handle
        hx = self.rect.x + fill_w
        pygame.draw.circle(screen, WHITE, (hx, self.rect.centery), 8)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
            self._update_val(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_val(event.pos[0])

    def _update_val(self, mx):
        pct = max(0, min(1, (mx - self.rect.x) / self.rect.w))
        raw = self.min_val + pct * (self.max_val - self.min_val)
        self.val = round(raw / self.step) * self.step


class Button:
    def __init__(self, x, y, w, h, label, color=ACCENT):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color
        self.hovered = False

    def draw(self, screen, font):
        c = tuple(min(255, c + 30) for c in self.color) if self.hovered else self.color
        pygame.draw.rect(screen, c, self.rect, border_radius=5)
        text = font.render(self.label, True, (0, 0, 0))
        tr = text.get_rect(center=self.rect.center)
        screen.blit(text, tr)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False


class Toggle:
    def __init__(self, x, y, label_a, label_b, active=0):
        self.x, self.y = x, y
        self.labels = [label_a, label_b]
        self.active = active
        self.rects = [pygame.Rect(x, y, 90, 25), pygame.Rect(x + 95, y, 90, 25)]

    def draw(self, screen, font):
        for i, (rect, label) in enumerate(zip(self.rects, self.labels)):
            color = ACCENT if i == self.active else DARK_GRAY
            pygame.draw.rect(screen, color, rect, border_radius=4)
            text = font.render(label, True, (0, 0, 0) if i == self.active else WHITE)
            tr = text.get_rect(center=rect.center)
            screen.blit(text, tr)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.rects):
                if rect.collidepoint(event.pos):
                    self.active = i
                    return True
        return False


class Stats:
    def __init__(self, max_history=200):
        self.error_ball_y = deque(maxlen=max_history)
        self.error_paddle_y = deque(maxlen=max_history)
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.total_actions = 0
        self.planned_count = 0
        self.reactive_count = 0

    def update(self, pred_ball_y, gt_ball_y, pred_pad_y, gt_pad_y, action, used_planner):
        self.error_ball_y.append(abs(pred_ball_y - gt_ball_y))
        self.error_paddle_y.append(abs(pred_pad_y - gt_pad_y))
        self.action_counts[action] += 1
        self.total_actions += 1
        if used_planner:
            self.planned_count += 1
        else:
            self.reactive_count += 1

    def reset(self):
        self.error_ball_y.clear()
        self.error_paddle_y.clear()
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.total_actions = 0
        self.planned_count = 0
        self.reactive_count = 0


def draw_graph(screen, x, y, w, h, values, color, max_val=0.1, label=""):
    pygame.draw.rect(screen, (20, 20, 30), (x, y, w, h))
    pygame.draw.rect(screen, DARK_GRAY, (x, y, w, h), 1)
    if len(values) >= 2:
        points = []
        for i, v in enumerate(values):
            px = x + int(i * w / len(values))
            py = y + h - int(min(v / max_val, 1.0) * h)
            points.append((px, py))
        pygame.draw.lines(screen, color, False, points, 1)
    font = pygame.font.Font(None, 14)
    screen.blit(font.render(label, True, color), (x + 3, y + 2))


def draw_trajectory(screen, ax, ay, aw, ah, latent_z, dynamics, probe, device, horizon=5):
    colors = {1: BLUE, 2: ORANGE}
    for action in [1, 2]:
        z_sim = latent_z.clone()
        points = []
        for step in range(horizon):
            z_sim = dynamics(z_sim, torch.tensor([action], device=device))
            pos = probe(z_sim)[0]
            bx = int(ax + pos[0].item() * aw)
            by = int(ay + pos[1].item() * ah)
            points.append((bx, by))
        if len(points) >= 2:
            for i in range(len(points) - 1):
                alpha = 1.0 - (i / len(points)) * 0.7
                color = tuple(int(c * alpha) for c in colors[action])
                pygame.draw.line(screen, color, points[i], points[i + 1], 1)
            pygame.draw.circle(screen, colors[action], points[-1], 3)


def main():
    # Load V2 config
    config_path = Path(os.path.dirname(__file__)) / "config_v2.json"
    if config_path.exists():
        game_config = Config.load(config_path)
    else:
        game_config = Config()
    game_config.target_score = 999
    game_config.headless = True

    agent_config = AgentConfig()
    device = torch.device('cpu')
    preprocessor = Preprocessor(agent_config)

    ckpt = os.path.join(os.path.dirname(__file__), "..", "game_agent", "checkpoints_v2")
    encoder = Encoder(agent_config).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ckpt, "encoder.pt"), map_location=device, weights_only=True))
    encoder.eval()
    dynamics = DynamicsPredictor(agent_config).to(device)
    dynamics.load_state_dict(torch.load(os.path.join(ckpt, "dynamics.pt"), map_location=device, weights_only=True))
    dynamics.eval()
    probe = PositionProbe(agent_config).to(device)
    probe.load_state_dict(torch.load(os.path.join(ckpt, "position_probe.pt"), map_location=device, weights_only=True))
    probe.eval()

    # GUI setup
    pygame.init()
    GAME_W = 256
    PANEL_W = 360
    SCREEN_H = 700
    screen = pygame.display.set_mode((GAME_W + PANEL_W, SCREEN_H))
    pygame.display.set_caption("V-JEPA Pong Dashboard")
    font = pygame.font.Font(None, 18)
    big_font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    # Controls
    px = GAME_W + 15
    pw = PANEL_W - 30

    mode_toggle = Toggle(px, 40, "Planning", "Reactive", active=0)
    horizon_slider = Slider(px, 90, pw, 1, 20, 10, "Planning Horizon")
    think_slider = Slider(px, 140, pw, 2, 20, 6, "Think Every N Frames")
    ball_speed_slider = Slider(px, 190, pw, 2, 15, 5, "Ball Speed", step=0.5, fmt="{:.1f}")
    paddle_speed_slider = Slider(px, 240, pw, 2, 15, 5, "Paddle Speed", step=0.5, fmt="{:.1f}")
    num_balls_slider = Slider(px, 290, pw, 1, 5, 1, "Number of Balls")

    difficulty_btns = [
        Button(px, 325, 65, 25, "Easy", GREEN),
        Button(px + 70, 325, 75, 25, "Medium", YELLOW),
        Button(px + 150, 325, 65, 25, "Hard", RED),
    ]
    current_difficulty = "easy"

    start_btn = Button(px, 365, 80, 30, "Start", GREEN)
    stop_btn = Button(px + 85, 365, 80, 30, "Stop", RED)
    reset_btn = Button(px + 170, 365, 80, 30, "Reset", BLUE)

    sliders = [horizon_slider, think_slider, ball_speed_slider, paddle_speed_slider, num_balls_slider]

    # Game state
    import random
    seed = random.randint(0, 999999)
    rng = make_rng(seed)
    game_config.bot_difficulty = current_difficulty
    bot = BotPolicy(difficulty=current_difficulty, rng=rng)
    env = PongEnv(game_config)
    obs = env.reset(seed=seed)

    score_left = 0
    score_right = 0
    step_count = 0
    action_left = 0
    think_counter = 0
    probe_pos = [0.5, 0.5, 0.5, 0.5]
    latent_z = torch.zeros(1, 256)
    stats = Stats()
    running_game = True
    used_planner = False

    print("V-JEPA Dashboard running. Close window to stop.\n")

    app_running = True
    while app_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app_running = False

            mode_toggle.handle_event(event)
            for s in sliders:
                s.handle_event(event)

            for i, btn in enumerate(difficulty_btns):
                if btn.handle_event(event):
                    current_difficulty = ["easy", "medium", "hard"][i]
                    game_config.bot_difficulty = current_difficulty

            if start_btn.handle_event(event):
                running_game = True
            if stop_btn.handle_event(event):
                running_game = False
            if reset_btn.handle_event(event):
                seed = random.randint(0, 999999)
                game_config.ball_speed = ball_speed_slider.val
                game_config.paddle_speed = paddle_speed_slider.val
                game_config.num_balls = int(num_balls_slider.val)
                game_config.bot_difficulty = current_difficulty
                rng = make_rng(seed)
                bot = BotPolicy(difficulty=current_difficulty, rng=rng)
                env = PongEnv(game_config)
                obs = env.reset(seed=seed)
                score_left = 0; score_right = 0; step_count = 0
                stats.reset()
                print(f"  Reset! Seed={seed} Diff={current_difficulty} Balls={int(num_balls_slider.val)} Speed={ball_speed_slider.val}")

        if running_game:
            frame = obs.frame
            think_counter += 1
            think_freq = int(think_slider.val)

            if think_counter >= think_freq:
                think_counter = 0
                obs_tensor = preprocessor(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    latent_z = encoder(obs_tensor)
                    pos = probe(latent_z)[0]

                probe_pos = [pos[i].item() for i in range(4)]
                ball_y = probe_pos[1]
                paddle_y = probe_pos[2]

                gt_ball_y = env._state.ball.y / 480.0
                gt_paddle_y = env._state.left_paddle.y / 480.0

                if mode_toggle.active == 0:  # Planning
                    horizon = int(horizon_slider.val)
                    scores = torch.zeros(3, device=device)
                    z_batch = latent_z.expand(3, -1).clone()
                    actions_t = torch.arange(3, device=device)
                    with torch.no_grad():
                        for plan_step in range(horizon):
                            z_batch = dynamics(z_batch, actions_t)
                            pos_pred = probe(z_batch)
                            scores += (0.95 ** plan_step) * (-torch.abs(pos_pred[:, 1] - pos_pred[:, 2]))
                    action_left = scores.argmax().item()
                    used_planner = True
                else:  # Reactive
                    diff = ball_y - paddle_y
                    if diff > 0.01:
                        action_left = 2
                    elif diff < -0.01:
                        action_left = 1
                    else:
                        action_left = 0
                    used_planner = False

                stats.update(ball_y, gt_ball_y, paddle_y, gt_paddle_y, action_left, used_planner)

            action_right = bot.get_action(env._state, "right")
            result = env.step(action_left, action_right)
            obs = result.observation

            new_state = env.get_state()
            if new_state["score_left"] > score_left:
                score_left = new_state["score_left"]
                print(f"  Agent scored! {score_left} - {score_right}")
            if new_state["score_right"] > score_right:
                score_right = new_state["score_right"]
                print(f"  Bot scored!   {score_left} - {score_right}")
            step_count += 1

            if result.done:
                print(f"  Episode done. {score_left} - {score_right}")
                obs = env.reset()
                score_left = 0; score_right = 0

        # ===== DRAW =====
        screen.fill(BG)

        # Game frame
        game_surface = pygame.surfarray.make_surface(obs.frame.transpose(1, 0, 2) if running_game else np.zeros((256, 256, 3), dtype=np.uint8).transpose(1, 0, 2))
        screen.blit(game_surface, (0, 0))
        # Score overlay
        score_text = big_font.render(f"{score_left}  {score_right}", True, WHITE)
        screen.blit(score_text, score_text.get_rect(centerx=GAME_W // 2, top=8))

        # Panel background
        pygame.draw.rect(screen, PANEL_BG, (GAME_W, 0, PANEL_W, SCREEN_H))

        # Title
        title = big_font.render("V-JEPA Dashboard", True, ACCENT)
        screen.blit(title, (px, 12))

        # Mode toggle
        mode_toggle.draw(screen, font)

        # Sliders
        for s in sliders:
            s.draw(screen, font)

        # Difficulty buttons
        font.render("Bot Difficulty:", True, WHITE)
        screen.blit(font.render("Bot Difficulty:", True, WHITE), (px, 260))
        for i, btn in enumerate(difficulty_btns):
            if ["easy", "medium", "hard"][i] == current_difficulty:
                pygame.draw.rect(screen, WHITE, btn.rect.inflate(4, 4), 2, border_radius=6)
            btn.draw(screen, font)

        # Start/Stop/Reset
        start_btn.draw(screen, font)
        stop_btn.draw(screen, font)
        reset_btn.draw(screen, font)

        # Status
        y = 410
        mode_name = "PLANNING" if mode_toggle.active == 0 else "REACTIVE"
        status_color = BLUE if mode_toggle.active == 0 else YELLOW
        screen.blit(big_font.render(f"Mode: {mode_name}", True, status_color), (px, y))
        y += 22
        screen.blit(font.render(f"Step: {step_count}  Score: {score_left}-{score_right}", True, WHITE), (px, y))
        y += 18

        action_names = {0: "NOOP", 1: "UP", 2: "DOWN"}
        action_colors = {0: GRAY, 1: BLUE, 2: ORANGE}
        screen.blit(font.render(f"Action: {action_names[action_left]}", True, action_colors[action_left]), (px, y))
        y += 22

        # Mini arena
        pygame.draw.line(screen, DARK_GRAY, (px, y), (px + pw, y))
        y += 5
        screen.blit(font.render("Model's View + Trajectory", True, YELLOW), (px, y))
        y += 18

        arena_x = px + 10
        arena_w = pw - 20
        arena_h = 100
        pygame.draw.rect(screen, (0, 0, 0), (arena_x, y, arena_w, arena_h))
        pygame.draw.rect(screen, DARK_GRAY, (arena_x, y, arena_w, arena_h), 1)

        with torch.no_grad():
            draw_trajectory(screen, arena_x, y, arena_w, arena_h, latent_z, dynamics, probe, device, horizon=int(horizon_slider.val))

        # Predicted positions
        bx = int(arena_x + probe_pos[0] * arena_w)
        by = int(y + probe_pos[1] * arena_h)
        pygame.draw.circle(screen, YELLOW, (bx, by), 5)
        ppx = arena_x + 5
        ppy = int(y + probe_pos[2] * arena_h)
        pygame.draw.rect(screen, YELLOW, (ppx, ppy - 8, 4, 16))

        # Ground truth
        gt_ball_y = env._state.ball.y / 480.0
        gt_ball_x = env._state.ball.x / 640.0
        gt_pad_y = env._state.left_paddle.y / 480.0
        gt_bx = int(arena_x + gt_ball_x * arena_w)
        gt_by = int(y + gt_ball_y * arena_h)
        pygame.draw.circle(screen, GREEN, (gt_bx, gt_by), 3, 1)
        gt_ppy = int(y + gt_pad_y * arena_h)
        pygame.draw.rect(screen, GREEN, (ppx + 6, gt_ppy - 8, 4, 16), 1)

        y += arena_h + 5
        screen.blit(pygame.font.Font(None, 14).render("Yellow=predicted  Green=actual  Blue/Orange=trajectory", True, GRAY), (px, y))
        y += 18

        # Error graphs
        pygame.draw.line(screen, DARK_GRAY, (px, y), (px + pw, y))
        y += 5
        avg_b = np.mean(stats.error_ball_y) if stats.error_ball_y else 0
        avg_p = np.mean(stats.error_paddle_y) if stats.error_paddle_y else 0
        screen.blit(font.render(f"Probe Error (ball={avg_b:.3f} pad={avg_p:.3f})", True, YELLOW), (px, y))
        y += 16
        draw_graph(screen, px, y, pw, 40, stats.error_ball_y, RED, label="Ball Y")
        y += 43
        draw_graph(screen, px, y, pw, 40, stats.error_paddle_y, GREEN, label="Paddle Y")
        y += 48

        # Action distribution
        pygame.draw.line(screen, DARK_GRAY, (px, y), (px + pw, y))
        y += 5
        screen.blit(font.render("Actions", True, YELLOW), (px, y))
        y += 16
        total = max(stats.total_actions, 1)
        for a, name, color in [(0, "NOOP", GRAY), (1, "UP", BLUE), (2, "DOWN", ORANGE)]:
            pct = stats.action_counts[a] / total
            pygame.draw.rect(screen, DARK_GRAY, (px, y, pw, 12))
            pygame.draw.rect(screen, color, (px, y, int(pct * pw), 12))
            screen.blit(pygame.font.Font(None, 14).render(f"{name} {pct*100:.0f}%", True, WHITE), (px + pw + 5, y))
            y += 16

        # Planner stats
        if stats.total_actions > 0:
            y += 5
            plan_pct = stats.planned_count / max(stats.total_actions, 1) * 100
            screen.blit(font.render(f"Planned: {plan_pct:.0f}%  Reactive: {100-plan_pct:.0f}%", True, GRAY), (px, y))

        pygame.display.flip()
        clock.tick(60)  # fixed display rate, independent of game speed

    pygame.quit()


if __name__ == "__main__":
    main()
