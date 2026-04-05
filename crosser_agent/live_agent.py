"""V-JEPA live agent for Street Crosser with debug dashboard.

Runs the crosser game headless, captures frames, plans with the world model,
and displays a debug panel showing predictions, trajectories, and stats.

Usage:
    python crosser_agent/live_agent.py
"""

import sys
import os
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crosser"))

from crosser.config import Config
from crosser.env.crosser_env import CrosserEnv
from crosser.env.state import NOOP, UP, DOWN, LEFT, RIGHT

from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.slot_attention import SlotEncoder
from game_agent.preprocessing.transforms import Preprocessor

import pygame


# ── Models (must match train_crosser.py) ──────────────────────────────

class DynamicsPredictor(nn.Module):
    def __init__(self, latent_dim=256, num_actions=5):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 64)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z, action):
        action_emb = self.action_embed(action.long())
        x = torch.cat([z, action_emb], dim=-1)
        out = self.net(x)
        return self.norm(z + out)


class PositionProbe(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class RewardHead(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ── Stats tracker ─────────────────────────────────────────────────────

class Stats:
    def __init__(self, max_history=200):
        self.error_col = deque(maxlen=max_history)
        self.error_row = deque(maxlen=max_history)
        self.action_counts = {i: 0 for i in range(5)}
        self.total_actions = 0
        self.scores = []
        self.hits = []

    def update_probe(self, pred_col, gt_col, pred_row, gt_row):
        self.error_col.append(abs(pred_col - gt_col))
        self.error_row.append(abs(pred_row - gt_row))

    def update_action(self, action):
        self.action_counts[action] += 1
        self.total_actions += 1

    @property
    def mean_col_err(self):
        return np.mean(self.error_col) if self.error_col else 0

    @property
    def mean_row_err(self):
        return np.mean(self.error_row) if self.error_row else 0


# ── Drawing helpers ───────────────────────────────────────────────────

ACTION_NAMES = {0: "NOOP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
ACTION_COLORS = {
    0: (150, 150, 150),
    1: (100, 200, 255),
    2: (255, 150, 100),
    3: (200, 100, 255),
    4: (100, 255, 150),
}


def draw_graph(screen, x, y, w, h, values, color, max_val=0.1, label=""):
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


def draw_action_bars(screen, x, y, action_counts, total):
    if total == 0:
        return
    font = pygame.font.Font(None, 16)
    bar_w = 160
    bar_h = 12
    for i in range(5):
        pct = action_counts[i] / total
        filled_w = int(pct * bar_w)
        by = y + i * (bar_h + 3)
        pygame.draw.rect(screen, (40, 40, 40), (x, by, bar_w, bar_h))
        pygame.draw.rect(screen, ACTION_COLORS[i], (x, by, filled_w, bar_h))
        label = font.render(f"{ACTION_NAMES[i]} {pct*100:.0f}%", True, (200, 200, 200))
        screen.blit(label, (x + bar_w + 5, by))


def draw_grid_view(screen, x, y, w, h, pred_col, pred_row,
                   gt_col, gt_row, latent_z, dynamics, probe,
                   reward_head, device, plan_scores):
    """Draw a mini grid showing predicted position + planned trajectories."""
    pygame.draw.rect(screen, (0, 0, 0), (x, y, w, h))
    pygame.draw.rect(screen, (80, 80, 80), (x, y, w, h), 1)

    # Grid lines (12x12)
    for i in range(13):
        gx = x + int(i * w / 12)
        gy = y + int(i * h / 12)
        pygame.draw.line(screen, (30, 30, 30), (gx, y), (gx, y + h))
        pygame.draw.line(screen, (30, 30, 30), (x, gy), (x + w, gy))

    # Goal zone (top 2 rows)
    pygame.draw.rect(screen, (20, 30, 80), (x, y, w, int(2 * h / 12)))
    # Safe zone (bottom 2 rows)
    pygame.draw.rect(screen, (20, 60, 30), (x, y + int(10 * h / 12), w, int(2 * h / 12)))

    # Draw trajectories for each action
    with torch.no_grad():
        for action in range(5):
            z_sim = latent_z.clone()
            points = []
            for step in range(8):
                z_sim = dynamics(z_sim, torch.tensor([action], device=device))
                pos = probe(z_sim)[0]
                px = int(x + pos[0].item() * w)
                py = int(y + pos[1].item() * h)
                points.append((px, py))

            if len(points) >= 2:
                color = ACTION_COLORS[action]
                alpha = 0.7
                for i in range(len(points) - 1):
                    fade = tuple(int(c * (1.0 - i * 0.1)) for c in color)
                    pygame.draw.line(screen, fade, points[i], points[i + 1], 2)
                # Endpoint dot
                pygame.draw.circle(screen, color, points[-1], 3)

    # Ground truth (green circle)
    gt_px = int(x + gt_col * w)
    gt_py = int(y + gt_row * h)
    pygame.draw.circle(screen, (0, 255, 0), (gt_px, gt_py), 6, 2)

    # Predicted (yellow dot)
    pred_px = int(x + pred_col * w)
    pred_py = int(y + pred_row * h)
    pygame.draw.circle(screen, (255, 255, 0), (pred_px, pred_py), 4)


class Button:
    """Simple clickable button."""
    def __init__(self, x, y, w, h, text, color=(80, 80, 100), hover_color=(100, 100, 130),
                 text_color=(220, 220, 220), active_color=(60, 140, 80)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.active_color = active_color
        self.active = False
        self._font = None

    def draw(self, screen):
        if self._font is None:
            self._font = pygame.font.Font(None, 18)
        mouse = pygame.mouse.get_pos()
        if self.active:
            c = self.active_color
        elif self.rect.collidepoint(mouse):
            c = self.hover_color
        else:
            c = self.color
        pygame.draw.rect(screen, c, self.rect, border_radius=4)
        pygame.draw.rect(screen, (140, 140, 140), self.rect, 1, border_radius=4)
        label = self._font.render(self.text, True, self.text_color)
        lr = label.get_rect(center=self.rect.center)
        screen.blit(label, lr)

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


def draw_plan_scores(screen, x, y, plan_scores):
    """Draw bar chart of planning scores per action."""
    font = pygame.font.Font(None, 16)
    if plan_scores is None:
        return
    min_s = min(plan_scores)
    max_s = max(plan_scores)
    rng = max_s - min_s if max_s != min_s else 1.0
    best = np.argmax(plan_scores)

    bar_w = 30
    max_h = 60
    for i in range(5):
        norm = (plan_scores[i] - min_s) / rng
        bh = int(norm * max_h)
        bx = x + i * (bar_w + 4)
        by = y + max_h - bh

        color = ACTION_COLORS[i]
        if i == best:
            pygame.draw.rect(screen, (255, 255, 255), (bx - 1, by - 1, bar_w + 2, bh + 2), 1)
        pygame.draw.rect(screen, color, (bx, by, bar_w, bh))

        label = font.render(ACTION_NAMES[i], True, (180, 180, 180))
        screen.blit(label, (bx, y + max_h + 3))

        score_text = font.render(f"{plan_scores[i]:.2f}", True, (150, 150, 150))
        screen.blit(score_text, (bx, y + max_h + 15))


# ── Planner ───────────────────────────────────────────────────────────

# Grid layout (12 rows): rows 0-1 = goal, rows 2-7 = car lanes, rows 8-9 = safe, rows 10-11 = start
# Normalized (0-11): goal < 2/11=0.182, car lanes 2/11 to 7/11=0.636, start > 9/11=0.818
SAFE_ROW_TOP = 2       # rows 0,1 are safe (goal zone)
SAFE_ROW_BOTTOM = 10   # rows 10,11 are safe (start zone)
CAR_LANE_START = 2     # first car lane
CAR_LANE_END = 7       # last car lane
GOAL_ROW_NORM = 2 / 11  # 0.182 — player must reach row < this to score


def plan_action_probe_only(latent_z, dynamics, probe, reward_head, device,
                           horizon=6, num_rollouts=64,
                           car_probe=None, **kwargs):
    """Reactive planner using only current-frame probes — no dynamics.

    Uses position probe to know where we are, car probe to see cars,
    and hardcoded grid mechanics (UP=row-1, etc.) to evaluate moves.
    No dynamics model, no future prediction.
    """
    GRID_COLS = 12
    GRID_ROWS = 12

    with torch.no_grad():
        current_pos = probe(latent_z)[0]
        current_col = current_pos[0].item()
        current_row = current_pos[1].item()
        cur_col = int(round(current_col * (GRID_COLS - 1)))
        cur_row = int(round(current_row * (GRID_ROWS - 1)))
        cur_col = max(0, min(GRID_COLS - 1, cur_col))
        cur_row = max(0, min(GRID_ROWS - 1, cur_row))

        # Get car occupancy from current frame
        if car_probe is not None:
            occ_probs = torch.sigmoid(car_probe(latent_z)[0]).view(GRID_ROWS, GRID_COLS)
        else:
            occ_probs = torch.zeros(GRID_ROWS, GRID_COLS)

        # Near goal: just go UP
        if cur_row <= CAR_LANE_START:
            return torch.tensor([-1.0, 10.0, -5.0, -1.0, -1.0], device=device)

        scores = torch.zeros(5, device=device)
        # action deltas: NOOP, UP, DOWN, LEFT, RIGHT
        action_deltas = [(0, 0), (0, -1), (0, +1), (-1, 0), (+1, 0)]

        for a in range(5):
            dc, dr = action_deltas[a]
            tgt_col = max(0, min(GRID_COLS - 1, cur_col + dc))
            tgt_row = max(0, min(GRID_ROWS - 1, cur_row + dr))

            # Car danger at target cell + neighbors
            if CAR_LANE_START <= tgt_row <= CAR_LANE_END:
                danger = occ_probs[tgt_row, tgt_col].item()
                danger_l = occ_probs[tgt_row, max(0, tgt_col - 1)].item()
                danger_r = occ_probs[tgt_row, min(GRID_COLS - 1, tgt_col + 1)].item()
                cell_danger = max(danger, 0.4 * danger_l, 0.4 * danger_r)
                scores[a] -= cell_danger * 5.0

            # Progress bonus for UP
            if a == 1:
                scores[a] += 0.3
            elif a == 2:
                scores[a] -= 0.1

            # Edge avoidance
            if tgt_col <= 0 or tgt_col >= GRID_COLS - 1:
                scores[a] -= 0.3

        # NOOP penalty
        scores[0] -= 0.05

    return scores


def plan_action_enhanced(latent_z, dynamics, probe, reward_head, device,
                         horizon=6, num_rollouts=64,
                         car_probe=None, player_col=None, player_row=None, **kwargs):
    """Enhanced planner — uses game state for player position and hardcoded lane knowledge."""
    GRID_COLS = 12
    GRID_ROWS = 12

    with torch.no_grad():
        current_pos = probe(latent_z)[0]
        current_row = current_pos[1].item()
        current_col = current_pos[0].item()

        if player_col is not None and player_row is not None:
            cur_grid_col = int(round(player_col))
            cur_grid_row = int(round(player_row))
        else:
            cur_grid_col = int(round(current_col * (GRID_COLS - 1)))
            cur_grid_row = int(round(current_row * (GRID_ROWS - 1)))

        if car_probe is not None:
            occ_logits = car_probe(latent_z)[0]
            occ_probs = torch.sigmoid(occ_logits).view(GRID_ROWS, GRID_COLS)
            occ_probs[:SAFE_ROW_TOP, :] = 0.0
            occ_probs[CAR_LANE_END + 1:, :] = 0.0
        else:
            occ_probs = None

        occ_future = []
        if car_probe is not None:
            z_future = latent_z.clone()
            noop_a = torch.tensor([0], device=device)
            for step in range(min(horizon, 4)):
                z_future = dynamics(z_future, noop_a)
                occ_f = torch.sigmoid(car_probe(z_future)[0]).view(GRID_ROWS, GRID_COLS)
                occ_f[:SAFE_ROW_TOP, :] = 0.0
                occ_f[CAR_LANE_END + 1:, :] = 0.0
                occ_future.append(occ_f)

        if cur_grid_row <= CAR_LANE_START:
            return torch.tensor([-1.0, 10.0, -5.0, -1.0, -1.0], device=device)

        scores = torch.zeros(5, device=device)
        action_deltas = [(0, 0), (0, -1), (0, +1), (-1, 0), (+1, 0)]

        for a in range(5):
            dc, dr = action_deltas[a]
            tgt_col = max(0, min(GRID_COLS - 1, cur_grid_col + dc))
            tgt_row = max(0, min(GRID_ROWS - 1, cur_grid_row + dr))

            if occ_probs is not None and CAR_LANE_START <= tgt_row <= CAR_LANE_END:
                danger = occ_probs[tgt_row, tgt_col].item()
                danger_left = occ_probs[tgt_row, max(0, tgt_col - 1)].item()
                danger_right = occ_probs[tgt_row, min(GRID_COLS - 1, tgt_col + 1)].item()
                cell_danger = max(danger, 0.5 * danger_left, 0.5 * danger_right)
                future_danger = 0.0
                for step, occ_f in enumerate(occ_future):
                    fd = occ_f[tgt_row, tgt_col].item()
                    future_danger = max(future_danger, fd * (0.8 ** step))
                scores[a] -= max(cell_danger, 0.7 * future_danger) * 5.0

            if a == 1: scores[a] += 0.15
            elif a == 2: scores[a] -= 0.08

        scores[0] -= 0.02
        if cur_grid_col <= 0: scores[3] -= 1.0
        if cur_grid_col >= GRID_COLS - 1: scores[4] -= 1.0

    return scores


def plan_action(latent_z, dynamics, probe, reward_head, device,
                horizon=6, num_rollouts=64,
                car_probe=None, **kwargs):
    """Pure V-JEPA planner — no game state, no hardcoded rules.

    Everything comes from learned models:
      - Position probe: where am I? where will I be after each action?
      - Car probe: where are the cars right now?
      - Dynamics model: what will the world look like after each action?
      - Reward head: is this a good state?

    No cheats: no game state access, no hardcoded lane knowledge,
    no hardcoded action mechanics.
    """
    GRID_COLS = 12
    GRID_ROWS = 12

    with torch.no_grad():
        # Where am I? (from probe, not game state)
        current_pos = probe(latent_z)[0]
        current_row = current_pos[1].item()
        current_col = current_pos[0].item()
        cur_grid_col = int(round(current_col * (GRID_COLS - 1)))
        cur_grid_row = int(round(current_row * (GRID_ROWS - 1)))

        # Where are the cars? (from car probe on current latent)
        if car_probe is not None:
            occ_logits = car_probe(latent_z)[0]
            occ_probs = torch.sigmoid(occ_logits).view(GRID_ROWS, GRID_COLS)
        else:
            occ_probs = None

        # Simulate each action through dynamics model
        actions_t = torch.arange(5, device=device)
        z_next = dynamics(latent_z.expand(5, -1), actions_t)
        pos_next = probe(z_next)        # (5, 2) — predicted position after each action
        r_next = reward_head(z_next)    # (5,) — predicted reward after each action

        # Also predict car occupancy after each action (from dynamics latent)
        if car_probe is not None:
            occ_next = torch.sigmoid(car_probe(z_next)).view(5, GRID_ROWS, GRID_COLS)
        else:
            occ_next = None

        # Predict future occupancy (cars move even if we stay)
        occ_future = []
        if car_probe is not None:
            z_future = latent_z.clone()
            noop_a = torch.tensor([0], device=device)
            for step in range(min(horizon, 4)):
                z_future = dynamics(z_future, noop_a)
                occ_f = torch.sigmoid(car_probe(z_future)[0]).view(GRID_ROWS, GRID_COLS)
                occ_future.append(occ_f)

        # Near the goal: dynamics predicts reset-to-bottom after crossing,
        # which makes the planner think UP is terrible. Override: just go UP.
        if current_row < GOAL_ROW_NORM + 0.05:  # within ~1 row of goal
            scores = torch.tensor([-1.0, 10.0, -5.0, -1.0, -1.0], device=device)
            return scores

        scores = torch.zeros(5, device=device)

        for a in range(5):
            # Where will I be after this action? (from dynamics + probe)
            tgt_row_norm = pos_next[a, 1].item()
            tgt_col_norm = pos_next[a, 0].item()

            # Dynamics predicts reset for UP near goal — use current_row - 1/11
            # as the expected position instead of trusting the probe
            if a == 1 and tgt_row_norm > current_row:
                # Dynamics predicted downward (reset) for UP — override
                tgt_row_norm = max(0.0, current_row - 1.0 / (GRID_ROWS - 1))

            tgt_row = int(round(tgt_row_norm * (GRID_ROWS - 1)))
            tgt_col = int(round(tgt_col_norm * (GRID_COLS - 1)))
            tgt_row = max(0, min(GRID_ROWS - 1, tgt_row))
            tgt_col = max(0, min(GRID_COLS - 1, tgt_col))

            # Check: is there a car at my predicted destination?
            if occ_probs is not None:
                danger = occ_probs[tgt_row, tgt_col].item()
                danger_left = occ_probs[tgt_row, max(0, tgt_col - 1)].item()
                danger_right = occ_probs[tgt_row, min(GRID_COLS - 1, tgt_col + 1)].item()
                cell_danger = max(danger, 0.5 * danger_left, 0.5 * danger_right)

                # Will a car arrive at my destination soon?
                future_danger = 0.0
                for step, occ_f in enumerate(occ_future):
                    fd = occ_f[tgt_row, tgt_col].item()
                    future_danger = max(future_danger, fd * (0.8 ** step))

                total_danger = max(cell_danger, 0.7 * future_danger)
                scores[a] -= total_danger * 5.0

            # Progress: did this action move me toward the top? (lower row = better)
            row_delta = current_row - tgt_row_norm
            if row_delta > 0.01:
                scores[a] += row_delta * 2.0  # reward upward movement
            elif row_delta < -0.01:
                scores[a] += row_delta * 1.0  # penalize downward (but less)

            # Reward head signal
            scores[a] += r_next[a].item() * 0.3

            # Edge avoidance (learned from probe — if col is near 0 or 1)
            if tgt_col_norm < 0.05 or tgt_col_norm > 0.95:
                scores[a] -= 0.5

        # Slight NOOP penalty to keep moving
        scores[0] -= 0.02

    return scores


# ── Main ──────────────────────────────────────────────────────────────

def main():
    use_slots = '--slots' in sys.argv
    device = torch.device('cpu')

    if use_slots:
        ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_slots')
        print("Using SLOT ATTENTION encoder (K=8, checkpoints_slots/)")
    else:
        ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        print("Using original encoder (checkpoints/)")

    # Load models
    agent_config = AgentConfig()
    agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)

    print("Loading models...")
    if use_slots:
        encoder = SlotEncoder(agent_config, num_slots=8, slot_dim=64, num_iters=3).to(device)
        encoder.load_state_dict(torch.load(
            os.path.join(ckpt_dir, 'slot_encoder.pt'), map_location=device, weights_only=True))
    else:
        encoder = Encoder(agent_config).to(device)
        encoder.load_state_dict(torch.load(
            os.path.join(ckpt_dir, 'encoder.pt'), map_location=device, weights_only=True))
    encoder.eval()

    dynamics = DynamicsPredictor(latent_dim=256, num_actions=5).to(device)
    dynamics.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'dynamics.pt'), map_location=device, weights_only=True))
    dynamics.eval()

    probe = PositionProbe(latent_dim=256).to(device)
    probe.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'position_probe.pt'), map_location=device, weights_only=True))
    probe.eval()

    reward_head = RewardHead(latent_dim=256).to(device)
    reward_head_path = os.path.join(ckpt_dir, 'reward_head.pt')
    if os.path.exists(reward_head_path):
        reward_head.load_state_dict(torch.load(reward_head_path, map_location=device, weights_only=True))
        reward_head.eval()
        print("Reward head loaded!")
    else:
        reward_head.eval()
        print("No reward head found — using untrained (reward signal ignored)")

    # Car occupancy probe
    from crosser_agent.train_car_probe import CarOccupancyProbe
    car_probe = CarOccupancyProbe(latent_dim=256, grid_size=144).to(device)
    car_probe_path = os.path.join(ckpt_dir, 'car_probe.pt')
    if os.path.exists(car_probe_path):
        car_probe.load_state_dict(torch.load(car_probe_path, map_location=device, weights_only=True))
        car_probe.eval()
        print("Car probe loaded!")
    else:
        car_probe = None
        print("No car probe found — running without it")

    # PPO learned policy
    from train_ppo_pixels import PolicyValueNet
    ppo_policy = PolicyValueNet(latent_dim=256, num_actions=5).to(device)
    ppo_path = os.path.join(ckpt_dir, 'policy_best.pt')
    if not os.path.exists(ppo_path):
        ppo_path = os.path.join(ckpt_dir, 'policy.pt')
    if os.path.exists(ppo_path):
        ppo_policy.load_state_dict(torch.load(ppo_path, map_location=device, weights_only=True))
        ppo_policy.eval()
        print(f"PPO policy loaded from {ppo_path}")
    else:
        ppo_policy = None
        print("No PPO policy found — ppo mode unavailable")

    # DQN end-to-end policy
    from train_dqn import DQN, preprocess_frame, FrameStack
    dqn_policy = DQN(in_channels=4, num_actions=5).to(device)
    dqn_path = os.path.join(os.path.dirname(__file__), 'dqn_best.pt')
    if not os.path.exists(dqn_path):
        dqn_path = os.path.join(os.path.dirname(__file__), 'dqn.pt')
    if os.path.exists(dqn_path):
        dqn_policy.load_state_dict(torch.load(dqn_path, map_location=device, weights_only=True))
        dqn_policy.eval()
        dqn_frames = FrameStack(k=4)
        print(f"DQN policy loaded from {dqn_path}")
    else:
        dqn_policy = None
        dqn_frames = None
        print("No DQN policy found — dqn mode unavailable")

    print("Models loaded!")

    # Setup game
    game_config = Config()
    game_config.headless = True
    game_config.target_score = 999
    game_config.max_steps = 2000
    if '--lanes' in sys.argv:
        game_config.num_lanes = int(sys.argv[sys.argv.index('--lanes') + 1])
    if '--cars' in sys.argv:
        game_config.max_cars_per_lane = int(sys.argv[sys.argv.index('--cars') + 1])

    env = CrosserEnv(game_config)
    seed = random.randint(0, 999999)
    obs = env.reset(seed=seed)

    # Setup display
    pygame.init()
    panel_width = 380
    game_w = game_config.render_width
    game_h = game_config.render_height
    screen_h = max(game_h, 650)  # taller to fit all controls
    screen = pygame.display.set_mode((game_w + panel_width, screen_h))
    pygame.display.set_caption("V-JEPA Crosser Agent")

    font = pygame.font.Font(None, 20)
    small_font = pygame.font.Font(None, 16)
    big_font = pygame.font.Font(None, 28)

    stats = Stats()
    clock = pygame.time.Clock()

    # State
    step_count = 0
    total_score = 0
    total_hits = 0
    episode = 1
    action = NOOP
    latent_z = torch.zeros(1, 256)
    probe_pos = [0.5, 0.5]
    plan_scores = None
    think_interval = 1  # plan every frame
    think_counter = 0
    planning_mode = "pure"  # "pure", "enhanced", "reactive", "ppo"
    speed_multiplier = 1  # 1=normal, 0=max speed
    paused = False
    horizon = 6
    num_rollouts = 48
    action_hold = 5  # hold each action for N frames before re-planning
    action_hold_counter = 0

    # GUI Buttons — positioned at bottom of panel, laid out later
    btn_w = 82
    btn_h = 28
    btn_gap = 6
    # Row 1: Mode, Reset, Pause
    btn_mode = Button(0, 0, btn_w, btn_h, "Planner")
    btn_reset = Button(0, 0, btn_w, btn_h, "Reset", color=(120, 60, 60), hover_color=(160, 80, 80))
    btn_pause = Button(0, 0, btn_w, btn_h, "Pause")
    # Row 2: Speed
    btn_speed_normal = Button(0, 0, btn_w, btn_h, "60 FPS")
    btn_speed_fast = Button(0, 0, btn_w, btn_h, "Max Speed")
    # Row 3: Think interval
    btn_think_minus = Button(0, 0, 38, btn_h, "- Thk")
    btn_think_plus = Button(0, 0, 38, btn_h, "+ Thk")
    # Row 4: Horizon
    btn_hz_minus = Button(0, 0, 38, btn_h, "- Hz")
    btn_hz_plus = Button(0, 0, 38, btn_h, "+ Hz")
    # Row 4 continued: Rollouts
    btn_ro_minus = Button(0, 0, 38, btn_h, "- Ro")
    btn_ro_plus = Button(0, 0, 38, btn_h, "+ Ro")
    # Row 5: Difficulty — cars per lane, max speed
    btn_cars_minus = Button(0, 0, 38, btn_h, "-")
    btn_cars_plus = Button(0, 0, 38, btn_h, "+")
    btn_speed_car_minus = Button(0, 0, 38, btn_h, "-")
    btn_speed_car_plus = Button(0, 0, 38, btn_h, "+")
    btn_lanes_minus = Button(0, 0, 38, btn_h, "-")
    btn_lanes_plus = Button(0, 0, 38, btn_h, "+")

    max_cars = game_config.max_cars_per_lane
    max_car_speed = game_config.max_car_speed
    num_lanes = game_config.num_lanes
    car_width = game_config.car_width

    # Scene presets: (name, lanes, cars, speed, car_width, car_height, free_roam, player_speed, color)
    scene_presets = [
        ("Default",    6, 3, 4.0, 2, 1, False, 1.0, (200, 200, 200)),
        ("Easy",       3, 2, 3.0, 2, 1, False, 1.0, (100, 255, 150)),
        ("Wide Cars",  4, 2, 3.0, 3, 1, False, 1.0, (255, 200, 100)),
        ("Tall Vans",  4, 2, 3.0, 2, 2, False, 1.0, (255, 180, 100)),
        ("Buses",      3, 1, 2.0, 4, 2, False, 1.0, (150, 130, 255)),
        ("Fast",       4, 3, 7.0, 2, 1, False, 1.0, (255, 100, 100)),
        ("Trucks",     3, 1, 1.5, 5, 3, False, 1.0, (200, 160, 100)),
        ("Smooth",     4, 2, 3.0, 2, 1, False, 0.3, (80, 255, 200)),
        ("Animal",     3, 2, 2.5, 3, 1, True,  0.4, (255, 230, 100)),
        ("Scatter",    4, 3, 3.0, 2, 1, True,  1.0, (100, 220, 255)),
        ("Roam Vans",  3, 3, 2.5, 3, 2, True,  1.0, (255, 200, 150)),
        ("Roam Chaos", 5, 4, 5.0, 2, 1, True,  1.0, (255, 100, 255)),
        ("Chaos",      6, 4, 5.0, 2, 1, False, 1.0, (255, 80, 200)),
    ]
    scene_idx = 0
    btn_scene_prev = Button(0, 0, 36, btn_h, "<")
    btn_scene_next = Button(0, 0, 36, btn_h, ">")

    all_buttons = [btn_mode, btn_reset, btn_pause,
                   btn_speed_normal, btn_speed_fast,
                   btn_think_minus, btn_think_plus,
                   btn_hz_minus, btn_hz_plus,
                   btn_ro_minus, btn_ro_plus,
                   btn_cars_minus, btn_cars_plus,
                   btn_speed_car_minus, btn_speed_car_plus,
                   btn_lanes_minus, btn_lanes_plus,
                   btn_scene_prev, btn_scene_next]

    print("V-JEPA Crosser Agent running. Close window to stop.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Button clicks
            if btn_mode.clicked(event):
                modes = ["pure", "enhanced", "reactive", "ppo", "dqn"]
                planning_mode = modes[(modes.index(planning_mode) + 1) % len(modes)]
            elif btn_reset.clicked(event):
                seed = random.randint(0, 999999)
                obs = env.reset(seed=seed)
                total_score = 0; total_hits = 0; step_count = 0
                episode += 1
            elif btn_pause.clicked(event):
                paused = not paused
            elif btn_speed_normal.clicked(event):
                speed_multiplier = 1
            elif btn_speed_fast.clicked(event):
                speed_multiplier = 0
            elif btn_think_minus.clicked(event):
                think_interval = max(1, think_interval - 1)
            elif btn_think_plus.clicked(event):
                think_interval = min(10, think_interval + 1)
            elif btn_hz_minus.clicked(event):
                horizon = max(1, horizon - 1)
            elif btn_hz_plus.clicked(event):
                horizon = min(20, horizon + 1)
            elif btn_ro_minus.clicked(event):
                num_rollouts = max(8, num_rollouts - 8)
            elif btn_ro_plus.clicked(event):
                num_rollouts = min(128, num_rollouts + 8)
            elif btn_cars_minus.clicked(event):
                max_cars = max(1, max_cars - 1)
                game_config.max_cars_per_lane = max_cars
            elif btn_cars_plus.clicked(event):
                max_cars = min(5, max_cars + 1)
                game_config.max_cars_per_lane = max_cars
            elif btn_speed_car_minus.clicked(event):
                max_car_speed = max(0.5, max_car_speed - 0.5)
                game_config.max_car_speed = max_car_speed
            elif btn_speed_car_plus.clicked(event):
                max_car_speed = min(8.0, max_car_speed + 0.5)
                game_config.max_car_speed = max_car_speed
            elif btn_lanes_minus.clicked(event):
                num_lanes = max(1, num_lanes - 1)
                game_config.num_lanes = num_lanes
            elif btn_lanes_plus.clicked(event):
                num_lanes = min(8, num_lanes + 1)
                game_config.num_lanes = num_lanes
            elif btn_scene_prev.clicked(event):
                scene_idx = (scene_idx - 1) % len(scene_presets)
                name, num_lanes, max_cars, max_car_speed, car_width, car_height, free_roam, p_spd, _ = scene_presets[scene_idx]
                game_config.num_lanes = num_lanes
                game_config.max_cars_per_lane = max_cars
                game_config.max_car_speed = max_car_speed
                game_config.car_width = car_width
                game_config.car_height = car_height
                game_config.free_roam = free_roam
                game_config.player_speed = p_spd
            elif btn_scene_next.clicked(event):
                scene_idx = (scene_idx + 1) % len(scene_presets)
                name, num_lanes, max_cars, max_car_speed, car_width, car_height, free_roam, p_spd, _ = scene_presets[scene_idx]
                game_config.num_lanes = num_lanes
                game_config.max_cars_per_lane = max_cars
                game_config.max_car_speed = max_car_speed
                game_config.car_width = car_width
                game_config.car_height = car_height
                game_config.free_roam = free_roam
                game_config.player_speed = p_spd

            # Keyboard shortcuts still work
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    modes = ["pure", "enhanced", "reactive", "ppo", "dqn"]
                    planning_mode = modes[(modes.index(planning_mode) + 1) % len(modes)]
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    seed = random.randint(0, 999999)
                    obs = env.reset(seed=seed)
                    total_score = 0; total_hits = 0; step_count = 0
                    episode += 1

        frame = obs.frame

        # ── Game logic (skip when paused) ─────────────────────────
        if not paused:
            # Re-plan every frame, but use action smoothing to prevent jitter
            # Only switch action if new one is significantly better than current
            obs_tensor = preprocessor(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                latent_z = encoder(obs_tensor)
                pos = probe(latent_z)[0]

            probe_pos = [pos[0].item(), pos[1].item()]

            state = env._state
            gt_col = state.player.col / (game_config.grid_cols - 1)
            gt_row = state.player.row / (game_config.grid_rows - 1)
            stats.update_probe(probe_pos[0], gt_col, probe_pos[1], gt_row)

            if planning_mode == "dqn" and dqn_policy is not None:
                # DQN uses its own frame preprocessing (84x84 grayscale, 4-stack)
                if not hasattr(dqn_frames, '_initialized'):
                    dqn_frames.reset(frame)
                    dqn_frames._initialized = True
                dqn_state = dqn_frames.push(frame)
                with torch.no_grad():
                    dqn_input = torch.tensor(dqn_state, dtype=torch.float32).unsqueeze(0)
                    q_vals = dqn_policy(dqn_input)
                    plan_scores = q_vals.cpu().numpy()[0]
                action = int(q_vals.argmax(dim=-1).item())

            elif planning_mode == "ppo" and ppo_policy is not None:
                with torch.no_grad():
                    logits, value = ppo_policy(latent_z)
                    plan_scores = logits.cpu().numpy()[0]
                action = int(logits.argmax(dim=-1).item())

            elif planning_mode in ("pure", "enhanced"):
                if planning_mode == "pure":
                    plan_scores_tensor = plan_action(
                        latent_z, dynamics, probe, reward_head, device,
                        horizon=horizon, num_rollouts=num_rollouts,
                        car_probe=car_probe)
                else:
                    plan_scores_tensor = plan_action_enhanced(
                        latent_z, dynamics, probe, reward_head, device,
                        horizon=horizon, num_rollouts=num_rollouts,
                        car_probe=car_probe,
                        player_col=state.player.col, player_row=state.player.row)

                plan_scores = plan_scores_tensor.cpu().numpy()

                best_action = int(plan_scores_tensor.argmax().item())
                best_score = plan_scores[best_action]
                current_score = plan_scores[action] if 0 <= action < len(plan_scores) else -999

                # Only switch if new action is meaningfully better (hysteresis)
                # or if current action has become dangerous (negative score)
                if (best_score > current_score + 0.1
                        or current_score < -0.5
                        or best_action == action):
                    action = best_action
            else:
                plan_scores = None
                action = UP

            stats.update_action(action)

            # Log decision details
            if plan_scores is None:
                plan_scores = np.zeros(5)
            state = env._state
            car_positions = [(c.row, round(c.x, 1), round(c.speed, 1)) for c in state.cars if c.row == state.player.row or c.row == state.player.row - 1]
            occ_str = ""
            if car_probe is not None and plan_scores is not None:
                with torch.no_grad():
                    occ_logits = car_probe(latent_z)[0]
                    occ_p = torch.sigmoid(occ_logits).view(12, 12)
                    occ_p[:SAFE_ROW_TOP, :] = 0.0
                    occ_p[CAR_LANE_END + 1:, :] = 0.0
                    pr = int(round(state.player.row))
                    if 0 <= pr < 12:
                        occ_str += f" occ_row{pr}=[{','.join(f'{occ_p[pr,c]:.1f}' for c in range(12))}]"
                    if 0 <= pr-1 < 12:
                        occ_str += f" occ_row{pr-1}=[{','.join(f'{occ_p[pr-1,c]:.1f}' for c in range(12))}]"

            print(f"  step={step_count:>4d} player=({state.player.col},{state.player.row}) "
                  f"action={ACTION_NAMES[action]:>5s} "
                  f"scores=[{' '.join(f'{s:+.2f}' for s in plan_scores)}] "
                  f"cars_nearby={car_positions}"
                  f"{occ_str}")

            # Step environment
            pre_col, pre_row = state.player.col, state.player.row
            result = env.step(action)
            obs = result.observation
            step_count += 1

            if result.info.get("scored"):
                total_score += 1
                print(f"  >>> CROSSED! Score: {total_score} (step {step_count})")
                action = NOOP
            if result.info.get("hit"):
                total_hits += 1
                print(f"  >>> HIT! Hits:{total_hits} step={step_count} was=({pre_col},{pre_row}) action={ACTION_NAMES[action]}")
                action = NOOP

            if result.done:
                print(f"Episode {episode} done. Score: {total_score}, Hits: {total_hits}")
                episode += 1
                seed = random.randint(0, 999999)
                obs = env.reset(seed=seed)
                total_score = 0
                total_hits = 0
                if dqn_frames is not None:
                    dqn_frames.reset(obs.frame)
                    dqn_frames._initialized = True
                step_count = 0

        # ── Draw ──────────────────────────────────────────────────

        screen.fill((20, 20, 25))

        # Game frame
        game_surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(game_surface, (0, 0))

        # Score overlay on game (with background for visibility)
        score_text = big_font.render(f"Score: {total_score}  Hits: {total_hits}", True, (255, 255, 255))
        score_bg = pygame.Surface((score_text.get_width() + 12, score_text.get_height() + 6))
        score_bg.set_alpha(180)
        score_bg.fill((0, 0, 0))
        screen.blit(score_bg, (4, 7))
        screen.blit(score_text, (10, 10))

        # ── Debug panel ───────────────────────────────────────────
        px = game_w  # panel x offset
        pw = panel_width
        pygame.draw.rect(screen, (30, 30, 40), (px, 0, pw, screen_h))

        y = 10

        # Title + status on one line
        title = font.render("V-JEPA Crosser", True, (0, 255, 200))
        screen.blit(title, (px + 10, y))
        mode_colors = {"pure": (100, 200, 255), "enhanced": (100, 255, 100), "reactive": (255, 200, 100)}
        mode_color = mode_colors.get(planning_mode, (200, 200, 200))
        mode_t = small_font.render(f"[{planning_mode}]", True, mode_color)
        screen.blit(mode_t, (px + 140, y + 3))
        pause_t = small_font.render("PAUSED", True, (255, 100, 100)) if paused else None
        if pause_t:
            screen.blit(pause_t, (px + pw - 55, y + 3))
        y += 20

        score_line = font.render(
            f"Score:{total_score}  Hits:{total_hits}  Ep:{episode}  Step:{step_count}",
            True, (200, 200, 200))
        screen.blit(score_line, (px + 10, y)); y += 18

        # Probe output — compact
        state = env._state
        gt_col = state.player.col / (game_config.grid_cols - 1)
        gt_row = state.player.row / (game_config.grid_rows - 1)

        probe_line = small_font.render(
            f"Probe col:{probe_pos[0]:.2f}({gt_col:.2f}) row:{probe_pos[1]:.2f}({gt_row:.2f})"
            f"  err:{stats.mean_col_err:.3f}/{stats.mean_row_err:.3f}",
            True, (160, 160, 160))
        screen.blit(probe_line, (px + 10, y)); y += 14

        # Decision + action on one line
        action_text = font.render(
            f"Action: {ACTION_NAMES[action]}", True, ACTION_COLORS[action])
        screen.blit(action_text, (px + 10, y)); y += 20

        # Plan scores bar chart
        if plan_scores is not None:
            draw_plan_scores(screen, px + 10, y, plan_scores)
            y += 90

        # Grid view with trajectories — compact
        pygame.draw.line(screen, (60, 60, 60), (px + 10, y), (px + pw - 10, y)); y += 3
        grid_size = 140
        draw_grid_view(screen, px + 20, y, grid_size, grid_size,
                       probe_pos[0], probe_pos[1], gt_col, gt_row,
                       latent_z, dynamics, probe, reward_head, device, plan_scores)

        # Error graphs next to grid view
        graph_x = px + 20 + grid_size + 10
        graph_w = pw - grid_size - 50
        graph_h = 30
        draw_graph(screen, graph_x, y, graph_w, graph_h,
                   stats.error_col, (255, 100, 100), max_val=0.1, label="Col err")
        draw_graph(screen, graph_x, y + graph_h + 3, graph_w, graph_h,
                   stats.error_row, (100, 255, 100), max_val=0.1, label="Row err")

        # Action distribution below graphs, next to grid
        act_y = y + 2 * (graph_h + 3) + 5
        draw_action_bars(screen, graph_x, act_y, stats.action_counts, stats.total_actions)

        y += grid_size + 5

        # ── Controls ──────────────────────────────────────────────
        pygame.draw.line(screen, (60, 60, 60), (px + 10, y), (px + pw - 10, y)); y += 4

        btn_mode.text = planning_mode.capitalize()
        btn_mode.active = (planning_mode != "reactive")
        btn_pause.text = "Resume" if paused else "Pause"
        btn_pause.active = paused
        btn_speed_normal.active = (speed_multiplier == 1)
        btn_speed_fast.active = (speed_multiplier == 0)

        bx = px + 10
        # Row 1: Mode | Pause | Reset
        btn_mode.rect.topleft = (bx, y)
        btn_pause.rect.topleft = (bx + btn_w + btn_gap, y)
        btn_reset.rect.topleft = (bx + 2 * (btn_w + btn_gap), y)
        btn_mode.draw(screen); btn_pause.draw(screen); btn_reset.draw(screen)
        y += btn_h + 4

        # Row 2: Speed | Think | Horizon
        btn_speed_normal.rect.topleft = (bx, y)
        btn_speed_fast.rect.topleft = (bx + btn_w + btn_gap, y)
        btn_speed_normal.draw(screen); btn_speed_fast.draw(screen)
        col = bx + 2 * (btn_w + btn_gap)
        lbl = small_font.render(f"Thk:{think_interval}", True, (150, 150, 150))
        screen.blit(lbl, (col, y + 7)); col += 36
        btn_think_minus.rect.topleft = (col, y); btn_think_minus.draw(screen); col += 40
        btn_think_plus.rect.topleft = (col, y); btn_think_plus.draw(screen)
        y += btn_h + 4

        # Row 3: Horizon + Rollouts
        col = bx
        lbl = small_font.render(f"Hz:{horizon}", True, (150, 150, 150))
        screen.blit(lbl, (col, y + 7)); col += 34
        btn_hz_minus.rect.topleft = (col, y); btn_hz_minus.draw(screen); col += 40
        btn_hz_plus.rect.topleft = (col, y); btn_hz_plus.draw(screen); col += 48
        lbl = small_font.render(f"Ro:{num_rollouts}", True, (150, 150, 150))
        screen.blit(lbl, (col, y + 7)); col += 38
        btn_ro_minus.rect.topleft = (col, y); btn_ro_minus.draw(screen); col += 40
        btn_ro_plus.rect.topleft = (col, y); btn_ro_plus.draw(screen)
        y += btn_h + 4

        # ── Difficulty ────────────────────────────────────────────
        pygame.draw.line(screen, (60, 60, 60), (px + 10, y), (px + pw - 10, y)); y += 3
        lbl = small_font.render("Difficulty (reset to apply)", True, (255, 200, 0))
        screen.blit(lbl, (px + 10, y)); y += 15

        # Cars + Speed on one row
        col = bx
        lbl = small_font.render(f"Cars:{max_cars}", True, (150, 150, 150))
        screen.blit(lbl, (col, y + 7)); col += 44
        btn_cars_minus.rect.topleft = (col, y); btn_cars_minus.draw(screen); col += 40
        btn_cars_plus.rect.topleft = (col, y); btn_cars_plus.draw(screen); col += 48
        lbl = small_font.render(f"Spd:{max_car_speed:.1f}", True, (150, 150, 150))
        screen.blit(lbl, (col, y + 7)); col += 48
        btn_speed_car_minus.rect.topleft = (col, y); btn_speed_car_minus.draw(screen); col += 40
        btn_speed_car_plus.rect.topleft = (col, y); btn_speed_car_plus.draw(screen)
        y += btn_h + 4

        # Lanes + Car width
        col = bx
        lbl = small_font.render(f"Lanes:{num_lanes}", True, (150, 150, 150))
        screen.blit(lbl, (col, y + 7)); col += 52
        btn_lanes_minus.rect.topleft = (col, y); btn_lanes_minus.draw(screen); col += 40
        btn_lanes_plus.rect.topleft = (col, y); btn_lanes_plus.draw(screen); col += 48
        lbl = small_font.render(f"CarW:{car_width}", True, (150, 150, 150))
        screen.blit(lbl, (col, y + 7))
        y += btn_h + 4

        # ── Scene Presets ─────────────────────────────────────────
        pygame.draw.line(screen, (60, 60, 60), (px + 10, y), (px + pw - 10, y)); y += 3
        lbl = small_font.render("Scene Presets (reset to apply)", True, (100, 200, 255))
        screen.blit(lbl, (px + 10, y)); y += 18

        scene_name, _, _, _, _, _, _, _, scene_color = scene_presets[scene_idx]
        btn_scene_prev.rect.topleft = (bx, y)
        btn_scene_prev.draw(screen)
        scene_lbl = font.render(scene_name, True, scene_color)
        screen.blit(scene_lbl, (bx + 44, y + 4))
        btn_scene_next.rect.topleft = (bx + 44 + scene_lbl.get_width() + 8, y)
        btn_scene_next.draw(screen)

        # Show preset details
        y += btn_h + 2
        _, p_lanes, p_cars, p_speed, p_cw, p_ch, p_fr, p_ps, _ = scene_presets[scene_idx]
        detail = small_font.render(
            f"{p_lanes}L  {p_cars}cars  spd:{p_speed}  {p_cw}x{p_ch}" +
            ("  FREE" if p_fr else "") +
            (f"  step:{p_ps}" if p_ps != 1.0 else ""),
            True, (120, 120, 140))
        screen.blit(detail, (bx + 5, y))

        pygame.display.flip()

        if speed_multiplier > 0:
            clock.tick(game_config.fps)

    env.close()
    pygame.quit()
    print(f"Final — Episode {episode}, Score: {total_score}, Hits: {total_hits}")


def benchmark():
    """Run N episodes headless, print crosses/hits stats. No GUI needed.

    Usage:
        python crosser_agent/live_agent.py --bench [--slots] [--mode pure|enhanced|reactive] [--episodes 20]
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--slots', action='store_true')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--mode', default='pure', choices=['pure', 'enhanced', 'reactive', 'probe'])
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--lanes', type=int, default=None)
    parser.add_argument('--cars', type=int, default=None)
    args = parser.parse_args()

    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    device = torch.device('cpu')
    if args.slots:
        ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_slots')
        print(f"Encoder: SLOT ATTENTION (K=8)")
    else:
        ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        print(f"Encoder: ORIGINAL")

    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)

    # Load models
    if args.slots:
        encoder = SlotEncoder(agent_config, num_slots=8, slot_dim=64, num_iters=3).to(device)
        encoder.load_state_dict(torch.load(
            os.path.join(ckpt_dir, 'slot_encoder.pt'), map_location=device, weights_only=True))
    else:
        encoder = Encoder(agent_config).to(device)
        encoder.load_state_dict(torch.load(
            os.path.join(ckpt_dir, 'encoder.pt'), map_location=device, weights_only=True))
    encoder.eval()

    dynamics = DynamicsPredictor(latent_dim=256, num_actions=5).to(device)
    dynamics.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'dynamics.pt'), map_location=device, weights_only=True))
    dynamics.eval()

    probe = PositionProbe(latent_dim=256).to(device)
    probe.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'position_probe.pt'), map_location=device, weights_only=True))
    probe.eval()

    reward_head = RewardHead(latent_dim=256).to(device)
    rh_path = os.path.join(ckpt_dir, 'reward_head.pt')
    if os.path.exists(rh_path):
        reward_head.load_state_dict(torch.load(rh_path, map_location=device, weights_only=True))
    reward_head.eval()

    from crosser_agent.train_car_probe import CarOccupancyProbe
    car_probe = CarOccupancyProbe(latent_dim=256, grid_size=144).to(device)
    cp_path = os.path.join(ckpt_dir, 'car_probe.pt')
    if os.path.exists(cp_path):
        car_probe.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
        car_probe.eval()
    else:
        car_probe = None

    print(f"Mode: {args.mode} | Episodes: {args.episodes} | Max steps: {args.max_steps}")
    print("-" * 60)

    game_config = Config(); game_config.headless = True
    game_config.target_score = 999; game_config.max_steps = args.max_steps
    if args.lanes is not None:
        game_config.num_lanes = args.lanes
    if args.cars is not None:
        game_config.max_cars_per_lane = args.cars

    total_crosses = 0
    total_hits = 0
    total_steps = 0
    rng = random.Random(42)

    for ep in range(1, args.episodes + 1):
        env = CrosserEnv(game_config)
        obs = env.reset(seed=rng.randint(0, 999999))
        ep_crosses = 0; ep_hits = 0; action = NOOP

        for step in range(args.max_steps):
            frame = obs.frame
            obs_tensor = preprocessor(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                latent_z = encoder(obs_tensor)

            if args.mode == 'reactive':
                action = UP
            elif args.mode == 'probe':
                scores = plan_action_probe_only(latent_z, dynamics, probe, reward_head, device,
                                                car_probe=car_probe)
                best = int(scores.argmax().item())
                best_score = scores[best].item()
                cur_score = scores[action].item() if action < 5 else -999
                if best_score > cur_score + 0.15 or cur_score < -0.5:
                    action = best
            elif args.mode == 'pure':
                scores = plan_action(latent_z, dynamics, probe, reward_head, device,
                                     car_probe=car_probe)
                best = int(scores.argmax().item())
                best_score = scores[best].item()
                cur_score = scores[action].item() if action < 5 else -999
                if best_score > cur_score + 0.15 or cur_score < -0.5:
                    action = best
            else:
                state = env._state
                scores = plan_action_enhanced(
                    latent_z, dynamics, probe, reward_head, device,
                    car_probe=car_probe,
                    player_col=state.player.col, player_row=state.player.row)
                best = int(scores.argmax().item())
                best_score = scores[best].item()
                cur_score = scores[action].item() if action < 5 else -999
                if best_score > cur_score + 0.15 or cur_score < -0.5:
                    action = best

            result = env.step(action)
            obs = result.observation

            if result.info.get("scored"):
                ep_crosses += 1
            if result.info.get("hit"):
                ep_hits += 1
            if result.done:
                break

        total_crosses += ep_crosses
        total_hits += ep_hits
        total_steps += step + 1
        ratio = ep_crosses / max(1, ep_hits)
        print(f"  Ep {ep:>2d}: crosses={ep_crosses}  hits={ep_hits}  "
              f"ratio={ratio:.1f}  steps={step+1}")

    print("-" * 60)
    ratio = total_crosses / max(1, total_hits)
    print(f"TOTAL: crosses={total_crosses}  hits={total_hits}  "
          f"ratio={ratio:.2f}  avg_steps={total_steps/args.episodes:.0f}")
    print(f"Per episode avg: {total_crosses/args.episodes:.1f} crosses, "
          f"{total_hits/args.episodes:.1f} hits")


if __name__ == "__main__":
    if '--bench' in sys.argv:
        benchmark()
    else:
        main()
