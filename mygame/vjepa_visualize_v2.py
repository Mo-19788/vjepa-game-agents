"""V-JEPA agent with real-time visualization of what the model sees.

Shows:
- Game frame with predicted positions overlaid
- Latent space activity
- Probe predictions vs actual game state
- Action decision reasoning
- Running accuracy & error graph
- Action distribution
- Ball trajectory prediction from world model
"""

import sys
import os
import time
from collections import deque

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
from game_agent.planning.shooting import PositionProbe
from game_agent.preprocessing.transforms import Preprocessor

import pygame


class Stats:
    """Track running stats for visualization."""
    def __init__(self, max_history=200):
        self.error_ball_y = deque(maxlen=max_history)
        self.error_paddle_y = deque(maxlen=max_history)
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.total_actions = 0
        self.max_history = max_history

    def update(self, pred_ball_y, gt_ball_y, pred_pad_y, gt_pad_y, action):
        self.error_ball_y.append(abs(pred_ball_y - gt_ball_y))
        self.error_paddle_y.append(abs(pred_pad_y - gt_pad_y))
        self.action_counts[action] += 1
        self.total_actions += 1

    @property
    def mean_ball_err(self):
        return np.mean(self.error_ball_y) if self.error_ball_y else 0

    @property
    def mean_pad_err(self):
        return np.mean(self.error_paddle_y) if self.error_paddle_y else 0


def draw_graph(screen, x, y, w, h, values, color, max_val=0.15, label=""):
    """Draw a scrolling line graph."""
    pygame.draw.rect(screen, (20, 20, 30), (x, y, w, h))
    pygame.draw.rect(screen, (60, 60, 60), (x, y, w, h), 1)

    if len(values) < 2:
        return

    points = []
    for i, v in enumerate(values):
        px = x + int(i * w / len(values))
        py = y + h - int(min(v / max_val, 1.0) * h)
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(screen, color, False, points, 1)

    font = pygame.font.Font(None, 14)
    text = font.render(label, True, color)
    screen.blit(text, (x + 3, y + 2))


def draw_action_pie(screen, x, y, radius, action_counts, total):
    """Draw action distribution as colored bars."""
    if total == 0:
        return

    colors = {0: (150, 150, 150), 1: (100, 200, 255), 2: (255, 150, 100)}
    names = {0: "NOOP", 1: "UP", 2: "DOWN"}
    font = pygame.font.Font(None, 16)

    bar_w = radius * 2
    bar_h = 12
    for i, (action, count) in enumerate(sorted(action_counts.items())):
        pct = count / total
        filled_w = int(pct * bar_w)
        by = y + i * (bar_h + 4)

        pygame.draw.rect(screen, (40, 40, 40), (x, by, bar_w, bar_h))
        pygame.draw.rect(screen, colors[action], (x, by, filled_w, bar_h))

        label = font.render(f"{names[action]} {pct*100:.0f}%", True, (200, 200, 200))
        screen.blit(label, (x + bar_w + 5, by))


def draw_trajectory(screen, arena_x, arena_y, arena_w, arena_h,
                    latent_z, dynamics, probe, device):
    """Draw predicted ball trajectory from world model for each action."""
    colors = {0: (150, 150, 150, 100), 1: (100, 200, 255), 2: (255, 150, 100)}
    action_names = {0: "NOOP", 1: "UP", 2: "DOWN"}

    for action in [1, 2]:  # UP and DOWN trajectories
        z_sim = latent_z.clone()
        points = []

        for step in range(15):
            z_sim = dynamics(z_sim, torch.tensor([action], device=device))
            pos = probe(z_sim)[0]
            bx = int(arena_x + pos[0].item() * arena_w)
            by = int(arena_y + pos[1].item() * arena_h)
            points.append((bx, by))

        if len(points) >= 2:
            # Draw trajectory as dotted line
            for i in range(len(points) - 1):
                alpha = 1.0 - (i / len(points)) * 0.7
                color = tuple(int(c * alpha) for c in colors[action])
                pygame.draw.line(screen, color, points[i], points[i + 1], 1)

            # Draw endpoint
            pygame.draw.circle(screen, colors[action], points[-1], 3)


def draw_debug_panel(screen, panel_x, probe_pos, gt_state, action, latent_z,
                     score_left, score_right, step_count, config, stats,
                     dynamics, probe_model, device):
    """Draw debug visualization panel next to the game."""
    font = pygame.font.Font(None, 20)
    small_font = pygame.font.Font(None, 16)
    panel_w = 350
    h = config.render_height + 200  # extra height

    # Background
    pygame.draw.rect(screen, (30, 30, 40), (panel_x, 0, panel_w, h))

    y = 10
    # Title
    title = font.render("V-JEPA Brain", True, (0, 255, 200))
    screen.blit(title, (panel_x + 10, y))
    y += 25

    # Score
    score = font.render(f"Score: {score_left} - {score_right}  Step: {step_count}", True, (200, 200, 200))
    screen.blit(score, (panel_x + 10, y))
    y += 25

    # Predicted positions
    pygame.draw.line(screen, (100, 100, 100), (panel_x + 10, y), (panel_x + panel_w - 10, y))
    y += 5
    header = font.render("Position Probe Output", True, (255, 200, 0))
    screen.blit(header, (panel_x + 10, y))
    y += 22

    pred_ball_y = probe_pos[1]
    pred_paddle_y = probe_pos[2]
    pred_ball_x = probe_pos[0]

    gt_ball_y = gt_state.ball.y / 480.0
    gt_paddle_y = gt_state.left_paddle.y / 480.0
    gt_ball_x = gt_state.ball.x / 640.0

    for label, pred, gt in [
        ("Ball X", pred_ball_x, gt_ball_x),
        ("Ball Y", pred_ball_y, gt_ball_y),
        ("Paddle Y", pred_paddle_y, gt_paddle_y),
    ]:
        text = small_font.render(f"{label}: pred={pred:.3f} gt={gt:.3f} err={abs(pred-gt):.3f}", True, (180, 180, 180))
        screen.blit(text, (panel_x + 10, y))
        y += 16

        bar_x = panel_x + 10
        bar_w = panel_w - 20
        bar_h = 8
        pygame.draw.rect(screen, (60, 60, 60), (bar_x, y, bar_w, bar_h))
        gt_px = int(bar_x + gt * bar_w)
        pygame.draw.rect(screen, (0, 200, 0), (gt_px - 2, y, 4, bar_h))
        pred_px = int(bar_x + pred * bar_w)
        pygame.draw.rect(screen, (255, 255, 0), (pred_px - 1, y, 2, bar_h))
        y += 14

    # Decision
    y += 5
    pygame.draw.line(screen, (100, 100, 100), (panel_x + 10, y), (panel_x + panel_w - 10, y))
    y += 5
    header = font.render("Decision", True, (255, 200, 0))
    screen.blit(header, (panel_x + 10, y))
    y += 22

    diff = pred_ball_y - pred_paddle_y
    diff_text = small_font.render(f"ball_y - paddle_y = {diff:+.4f}", True, (180, 180, 180))
    screen.blit(diff_text, (panel_x + 10, y))
    y += 18

    action_names = {0: "NOOP", 1: "UP", 2: "DOWN"}
    action_colors = {0: (150, 150, 150), 1: (100, 200, 255), 2: (255, 150, 100)}
    action_text = font.render(f"Action: {action_names[action]}", True, action_colors[action])
    screen.blit(action_text, (panel_x + 10, y))
    y += 25

    # Model's View with trajectory prediction
    pygame.draw.line(screen, (100, 100, 100), (panel_x + 10, y), (panel_x + panel_w - 10, y))
    y += 5
    header = font.render("Model's View + Trajectory", True, (255, 200, 0))
    screen.blit(header, (panel_x + 10, y))
    y += 22

    arena_x = panel_x + 20
    arena_w = panel_w - 40
    arena_h = 140
    pygame.draw.rect(screen, (0, 0, 0), (arena_x, y, arena_w, arena_h))
    pygame.draw.rect(screen, (80, 80, 80), (arena_x, y, arena_w, arena_h), 1)

    # Center line
    for dy in range(0, arena_h, 8):
        pygame.draw.rect(screen, (50, 50, 50), (arena_x + arena_w // 2, y + dy, 1, 4))

    # Draw trajectory predictions from world model
    with torch.no_grad():
        draw_trajectory(screen, arena_x, y, arena_w, arena_h,
                        latent_z, dynamics, probe_model, device)

    # Predicted ball (yellow)
    bx = int(arena_x + pred_ball_x * arena_w)
    by_draw = int(y + pred_ball_y * arena_h)
    pygame.draw.circle(screen, (255, 255, 0), (bx, by_draw), 5)
    pygame.draw.circle(screen, (255, 255, 200), (bx, by_draw), 5, 1)

    # Predicted paddle (yellow)
    px = arena_x + 5
    py = int(y + pred_paddle_y * arena_h)
    pygame.draw.rect(screen, (255, 255, 0), (px, py - 12, 5, 24))

    # Ground truth ball (green outline)
    gt_bx = int(arena_x + gt_ball_x * arena_w)
    gt_by = int(y + gt_ball_y * arena_h)
    pygame.draw.circle(screen, (0, 255, 0), (gt_bx, gt_by), 4, 1)

    # Ground truth paddle (green outline)
    gt_py = int(y + gt_paddle_y * arena_h)
    pygame.draw.rect(screen, (0, 255, 0), (px + 7, gt_py - 12, 5, 24), 1)

    y += arena_h + 5
    legend = small_font.render("Yellow=predicted  Green=actual  Lines=trajectory", True, (120, 120, 120))
    screen.blit(legend, (panel_x + 10, y))
    y += 20

    # Error graph over time
    pygame.draw.line(screen, (100, 100, 100), (panel_x + 10, y), (panel_x + panel_w - 10, y))
    y += 5
    header = font.render(f"Probe Error (avg ball={stats.mean_ball_err:.3f} pad={stats.mean_pad_err:.3f})", True, (255, 200, 0))
    screen.blit(header, (panel_x + 10, y))
    y += 20

    graph_h = 50
    draw_graph(screen, panel_x + 10, y, panel_w - 20, graph_h,
               stats.error_ball_y, (255, 100, 100), max_val=0.1, label="Ball Y err")
    y += graph_h + 3
    draw_graph(screen, panel_x + 10, y, panel_w - 20, graph_h,
               stats.error_paddle_y, (100, 255, 100), max_val=0.1, label="Paddle Y err")
    y += graph_h + 10

    # Action distribution
    pygame.draw.line(screen, (100, 100, 100), (panel_x + 10, y), (panel_x + panel_w - 10, y))
    y += 5
    header = font.render("Action Distribution", True, (255, 200, 0))
    screen.blit(header, (panel_x + 10, y))
    y += 20
    draw_action_pie(screen, panel_x + 10, y, 80, stats.action_counts, stats.total_actions)
    y += 50

    # Latent space
    pygame.draw.line(screen, (100, 100, 100), (panel_x + 10, y), (panel_x + panel_w - 10, y))
    y += 5
    header = font.render("Latent Space (64 dims)", True, (255, 200, 0))
    screen.blit(header, (panel_x + 10, y))
    y += 18

    z = latent_z[0].cpu().numpy()[:64]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min > 0:
        z_norm = (z - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z)

    cell_w = 8
    cell_h = 8
    for i, val in enumerate(z_norm):
        col = i % 16
        row = i // 16
        r = int(val * 255)
        b = int((1 - val) * 255)
        color = (r, 50, b)
        pygame.draw.rect(screen, color,
                         (panel_x + 10 + col * (cell_w + 1),
                          y + row * (cell_h + 1), cell_w, cell_h))


def main():
    # V2: bigger, colorful visuals
    from pathlib import Path
    config_path = Path(os.path.dirname(__file__)) / "config_v2.json"
    if config_path.exists():
        game_config = Config.load(config_path)
    else:
        game_config = Config()
        game_config.ball_size = 24
        game_config.paddle_height = 100
        game_config.paddle_width = 16
        game_config.ball_color = (255, 255, 0)
        game_config.paddle_color = (0, 200, 255)
        game_config.bg_color = (20, 20, 40)
        game_config.line_color = (60, 60, 100)
    game_config.target_score = 999
    game_config.headless = True  # We handle display ourselves
    game_config.bot_difficulty = "easy"

    agent_config = AgentConfig()
    device = torch.device('cpu')
    preprocessor = Preprocessor(agent_config)

    ckpt = os.path.join(os.path.dirname(__file__), "..", "game_agent", "checkpoints_v2")

    encoder = Encoder(agent_config).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ckpt, "encoder.pt"),
                                        map_location=device, weights_only=True))
    encoder.eval()

    dynamics = DynamicsPredictor(agent_config).to(device)
    dynamics.load_state_dict(torch.load(os.path.join(ckpt, "dynamics.pt"),
                                         map_location=device, weights_only=True))
    dynamics.eval()

    probe = PositionProbe(agent_config).to(device)
    probe.load_state_dict(torch.load(os.path.join(ckpt, "position_probe.pt"),
                                      map_location=device, weights_only=True))
    probe.eval()

    # Setup display: game + debug panel
    pygame.init()
    panel_width = 350
    screen_h = max(256, 700)
    screen = pygame.display.set_mode(
        (game_config.render_width + panel_width, screen_h)
    )
    pygame.display.set_caption("V-JEPA Pong - Debug View")

    import random
    seed = random.randint(0, 999999)
    rng = make_rng(seed)
    bot = BotPolicy(difficulty=game_config.bot_difficulty, rng=rng)

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

    clock = pygame.time.Clock()

    print("V-JEPA Debug Visualizer running. Close window to stop.\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame = obs.frame

        think_counter += 1
        if think_counter >= 6:
            think_counter = 0
            obs_tensor = preprocessor(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                latent_z = encoder(obs_tensor)
                pos = probe(latent_z)[0]

            probe_pos = [pos[i].item() for i in range(4)]
            ball_y = probe_pos[1]
            paddle_y = probe_pos[2]

            # Update stats
            gt_ball_y = env._state.ball.y / 480.0
            gt_paddle_y = env._state.left_paddle.y / 480.0
            stats.update(ball_y, gt_ball_y, paddle_y, gt_paddle_y, action_left)

            diff = ball_y - paddle_y
            if diff > 0.01:
                action_left = 2
            elif diff < -0.01:
                action_left = 1
            else:
                action_left = 0

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
            print(f"\nEpisode done. Final: {score_left} - {score_right}")
            obs = env.reset()
            score_left = 0
            score_right = 0

        # Clear screen
        screen.fill((20, 20, 25))

        # Draw game frame on left side
        game_surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(game_surface, (0, 0))

        # Draw score on game area
        score_font = pygame.font.Font(None, 36)
        score_text = score_font.render(f"{score_left}  {score_right}", True, (200, 200, 200))
        score_rect = score_text.get_rect(centerx=game_config.render_width // 2, top=10)
        screen.blit(score_text, score_rect)

        # Draw debug panel on right side
        draw_debug_panel(screen, game_config.render_width, probe_pos,
                         env._state, action_left, latent_z,
                         score_left, score_right, step_count, game_config,
                         stats, dynamics, probe, device)

        pygame.display.flip()
        clock.tick(game_config.fps)

    pygame.quit()
    print(f"Final: {score_left} - {score_right}")


if __name__ == "__main__":
    main()
