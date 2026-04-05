"""Train a car occupancy probe on pre-encoded latents.

Generates (latent, occupancy_grid) pairs by running the game,
then trains a probe to predict the 12x12 occupancy grid from the latent.

The occupancy grid has 1 where a car overlaps a cell, 0 otherwise.
Only car lanes (rows 2-7) have cars, so effective grid is 12x6 = 72 cells.
"""

import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'crosser'))

from crosser.config import Config
from crosser.env.crosser_env import CrosserEnv
from crosser.env.state import NOOP, UP, DOWN, LEFT, RIGHT

from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.preprocessing.transforms import Preprocessor


GRID_COLS = 12
GRID_ROWS = 12


def state_to_occupancy(state, config):
    """Convert game state to a flat occupancy vector (12*12 = 144)."""
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    for car in state.cars:
        row = car.row
        if row < 0 or row >= GRID_ROWS:
            continue
        # Car occupies [car.x, car.x + car.width) — may span fractional cells
        x_start = int(np.floor(car.x))
        x_end = int(np.ceil(car.x + car.width))
        for col in range(max(0, x_start), min(GRID_COLS, x_end)):
            grid[row, col] = 1.0
    return grid.flatten()


class CarOccupancyProbe(nn.Module):
    """Predicts 12x12 occupancy grid from 256-dim latent."""
    def __init__(self, latent_dim=256, grid_size=144):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, grid_size),
        )

    def forward(self, z):
        return self.net(z)


def generate_data(num_episodes=100, steps_per_ep=300):
    """Run game episodes, encode frames, collect (latent, occupancy) pairs."""
    device = torch.device('cpu')
    agent_config = AgentConfig()
    agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    encoder = Encoder(agent_config).to(device)
    encoder.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'encoder.pt'), map_location=device, weights_only=True))
    encoder.eval()

    game_config = Config()
    game_config.headless = True

    all_z = []
    all_occ = []

    rng = np.random.RandomState(42)
    actions = [NOOP, UP, DOWN, LEFT, RIGHT]

    print(f"Generating data: {num_episodes} episodes x {steps_per_ep} steps...")

    for ep in range(num_episodes):
        env = CrosserEnv(game_config)
        seed = rng.randint(0, 999999)
        obs = env.reset(seed=seed)

        for step in range(steps_per_ep):
            frame = obs.frame
            state = env._state

            # Encode frame
            with torch.no_grad():
                z = encoder(preprocessor(frame).unsqueeze(0))
            all_z.append(z)

            # Get occupancy
            occ = state_to_occupancy(state, game_config)
            all_occ.append(occ)

            # Random action for diversity
            action = rng.choice(actions)
            result = env.step(action)
            obs = result.observation

            if result.done:
                obs = env.reset(seed=rng.randint(0, 999999))

        if (ep + 1) % 20 == 0:
            print(f"  {ep + 1}/{num_episodes} episodes done ({len(all_z)} samples)")

    all_z = torch.cat(all_z, dim=0)
    all_occ = torch.tensor(np.array(all_occ), dtype=torch.float32)

    print(f"Total: {len(all_z)} samples")
    print(f"Occupancy stats: mean={all_occ.mean():.3f}, nonzero={(all_occ > 0).float().mean():.3f}")

    return all_z, all_occ


def train_probe(all_z, all_occ, epochs=200):
    """Train the car occupancy probe."""
    device = torch.device('cpu')

    # Train/val split
    n = len(all_z)
    perm = torch.randperm(n)
    train_n = int(n * 0.9)
    train_z, val_z = all_z[perm[:train_n]], all_z[perm[train_n:]]
    train_occ, val_occ = all_occ[perm[:train_n]], all_occ[perm[train_n:]]

    probe = CarOccupancyProbe(latent_dim=256, grid_size=GRID_COLS * GRID_ROWS).to(device)
    optimizer = AdamW(probe.parameters(), lr=1e-3)

    # Class weight: occupied cells are rare (~15%), weight them higher
    pos_weight = ((1 - all_occ.mean()) / all_occ.mean()).clamp(max=10.0)
    print(f"\nTraining car probe: {sum(p.numel() for p in probe.parameters()):,} params")
    print(f"Train: {train_n}, Val: {n - train_n}, pos_weight: {pos_weight:.1f}")

    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        probe.train()
        perm2 = torch.randperm(train_n)
        loss_sum = 0
        steps = 0

        for s in range(0, train_n, 256):
            idx = perm2[s:s+256]
            pred = probe(train_z[idx])
            # Weighted BCE — penalize missing occupied cells more
            loss = F.binary_cross_entropy_with_logits(
                pred, train_occ[idx],
                pos_weight=torch.tensor(pos_weight))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            steps += 1

        probe.eval()
        with torch.no_grad():
            val_pred = probe(val_z)
            val_loss = F.binary_cross_entropy_with_logits(val_pred, val_occ).item()

            # Accuracy: threshold at 0 (logits)
            val_binary = (val_pred > 0).float()
            accuracy = (val_binary == val_occ).float().mean().item()

            # Per-cell accuracy for occupied cells only
            occupied_mask = val_occ > 0
            if occupied_mask.any():
                occ_recall = val_binary[occupied_mask].mean().item()
            else:
                occ_recall = 0

            empty_mask = val_occ == 0
            if empty_mask.any():
                empty_acc = (1 - val_binary[empty_mask]).mean().item()
            else:
                empty_acc = 0

        if val_loss < best_val:
            best_val = val_loss

        if epoch % 10 == 0 or epoch <= 3:
            print(f"Epoch {epoch}/{epochs} | Train: {loss_sum/steps:.4f} | "
                  f"Val: {val_loss:.4f} | Acc: {accuracy:.3f} | "
                  f"Recall: {occ_recall:.3f} | Empty: {empty_acc:.3f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'car_probe.pt')
    torch.save(probe.state_dict(), out_path)
    print(f"\nSaved to {out_path}")

    return probe


def test_probe(probe, all_z, all_occ):
    """Quick test: visualize a few predictions."""
    probe.eval()
    with torch.no_grad():
        idx = torch.randint(0, len(all_z), (5,))
        preds = torch.sigmoid(probe(all_z[idx]))
        gts = all_occ[idx]

    for i in range(5):
        pred_grid = (preds[i] > 0.5).view(GRID_ROWS, GRID_COLS).numpy().astype(int)
        gt_grid = gts[i].view(GRID_ROWS, GRID_COLS).numpy().astype(int)
        correct = (pred_grid == gt_grid).sum()
        total = GRID_ROWS * GRID_COLS
        print(f"\nSample {i}: {correct}/{total} cells correct")
        print("GT:   ", gt_grid[2:8].tolist())  # car lanes only
        print("Pred: ", pred_grid[2:8].tolist())


if __name__ == "__main__":
    all_z, all_occ = generate_data(num_episodes=200, steps_per_ep=300)
    probe = train_probe(all_z, all_occ, epochs=500)
    test_probe(probe, all_z, all_occ)
