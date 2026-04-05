"""Retrain probes on LOCAL encoder latents.

The slot encoder was trained on GPU, but CPU torch produces slightly different
latents due to floating point differences. Probes trained on GPU latents
don't transfer well. Fix: re-encode data locally, retrain probes on those latents.

This runs on CPU — no GPU needed.

Usage:
    python retrain_probes_local.py [--episodes 100]
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import sys, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, '.'); sys.path.insert(0, 'crosser')

from crosser.config import Config
from crosser.env.crosser_env import CrosserEnv
from crosser.env.state import NOOP, UP, DOWN, LEFT, RIGHT
from game_agent.config import AgentConfig
from game_agent.models.slot_attention import SlotEncoder
from game_agent.preprocessing.transforms import Preprocessor

GRID_COLS, GRID_ROWS = 12, 12
CKPT = 'crosser_agent/checkpoints_slots'


class PositionProbe(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


class CarOccupancyProbe(nn.Module):
    def __init__(self, latent_dim=256, grid_size=144):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, grid_size),
        )
    def forward(self, z): return self.net(z)


class DynamicsPredictor(nn.Module):
    def __init__(self, latent_dim=256, num_actions=5):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 64)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 64, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
    def forward(self, z, action):
        a = self.action_embed(action.long())
        return self.norm(z + self.net(torch.cat([z, a], -1)))


def state_to_occupancy(state):
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    for car in state.cars:
        row = car.row
        if 0 <= row < GRID_ROWS:
            for col in range(max(0, int(np.floor(car.x))),
                             min(GRID_COLS, int(np.ceil(car.x + car.width)))):
                grid[row, col] = 1.0
    return grid.flatten()


def generate_and_encode(encoder, num_episodes=100, steps_per_ep=400, subsample=2):
    """Generate data AND encode with local encoder in one pass (memory efficient)."""
    print(f'\n=== Generating + Encoding: {num_episodes} eps ===', flush=True)
    game_config = Config(); game_config.headless = True
    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)
    rng = np.random.RandomState(42)
    actions_list = [NOOP, UP, DOWN, LEFT, RIGHT]

    total = num_episodes * (steps_per_ep // subsample)

    # Store latents (small) + metadata, NOT raw frames (saves ~24GB RAM)
    all_z = torch.empty(total, 256)
    all_z_next = torch.empty(total, 256)
    all_actions = torch.empty(total, dtype=torch.long)
    all_positions = torch.empty(total, 2)
    all_occupancy = torch.empty(total, GRID_ROWS * GRID_COLS)

    encoder.eval()
    idx = 0
    t0 = time.time()

    for ep in range(num_episodes):
        env = CrosserEnv(game_config)
        obs = env.reset(seed=rng.randint(0, 999999))

        for step in range(steps_per_ep):
            state = env._state
            action = rng.choice(actions_list)
            result = env.step(action)

            if step % subsample == 0 and idx < total:
                frame_t = preprocessor(obs.frame).unsqueeze(0)
                next_frame_t = preprocessor(result.observation.frame).unsqueeze(0)

                with torch.no_grad():
                    z = encoder(frame_t)
                    z_next = encoder(next_frame_t)

                all_z[idx] = z.squeeze(0)
                all_z_next[idx] = z_next.squeeze(0)
                all_actions[idx] = action
                all_positions[idx, 0] = state.player.col / (game_config.grid_cols - 1)
                all_positions[idx, 1] = state.player.row / (game_config.grid_rows - 1)
                all_occupancy[idx] = torch.from_numpy(state_to_occupancy(state))
                idx += 1

            obs = result.observation
            if result.done:
                obs = env.reset(seed=rng.randint(0, 999999))

        if (ep + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f'  {ep+1}/{num_episodes} ({idx} samples, {elapsed:.0f}s)', flush=True)

    print(f'Encoded {idx} samples with LOCAL encoder', flush=True)
    return {
        'z': all_z[:idx], 'z_next': all_z_next[:idx],
        'actions': all_actions[:idx], 'positions': all_positions[:idx],
        'occupancy': all_occupancy[:idx],
    }


def retrain_position_probe(data, epochs=200):
    print(f'\n=== Retraining Position Probe ({epochs} epochs) ===', flush=True)
    z, pos = data['z'], data['positions']
    n = len(z)
    perm = torch.randperm(n); train_n = int(n * 0.9)
    tz, vz = z[perm[:train_n]], z[perm[train_n:]]
    tp, vp_gt = pos[perm[:train_n]], pos[perm[train_n:]]

    probe = PositionProbe()
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float('inf'); best_state = None
    for epoch in range(1, epochs + 1):
        probe.train()
        pm = torch.randperm(train_n)
        ls = 0; st = 0
        for s in range(0, train_n, 256):
            idx = pm[s:s+256]
            pred = probe(tz[idx])
            loss = F.mse_loss(pred, tp[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ls += loss.item(); st += 1
        scheduler.step()

        if epoch % 20 == 0 or epoch <= 3:
            probe.eval()
            with torch.no_grad():
                vp = probe(vz)
                vl = F.mse_loss(vp, vp_gt).item()
            print(f'  Ep {epoch:>3d} | Train:{ls/st:.6f} Val:{vl:.6f}', flush=True)
            if vl < best_val:
                best_val = vl; best_state = copy.deepcopy(probe.state_dict())

    if best_state: probe.load_state_dict(best_state)
    torch.save(probe.state_dict(), f'{CKPT}/position_probe.pt')
    print(f'Position probe saved (best val: {best_val:.6f})', flush=True)
    return probe


def retrain_car_probe(data, epochs=500):
    print(f'\n=== Retraining Car Probe ({epochs} epochs) ===', flush=True)
    z, occ = data['z'], data['occupancy']
    n = len(z)
    perm = torch.randperm(n); train_n = int(n * 0.9)
    tz, vz = z[perm[:train_n]], z[perm[train_n:]]
    to, vo = occ[perm[:train_n]], occ[perm[train_n:]]

    pos_weight = torch.tensor(3.0)
    probe = CarOccupancyProbe()
    optimizer = AdamW(probe.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float('inf'); best_state = None
    for epoch in range(1, epochs + 1):
        probe.train()
        pm = torch.randperm(train_n)
        ls = 0; st = 0
        for s in range(0, train_n, 512):
            idx = pm[s:s+512]
            pred = probe(tz[idx])
            loss = F.binary_cross_entropy_with_logits(pred, to[idx], pos_weight=pos_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ls += loss.item(); st += 1
        scheduler.step()

        if epoch % 50 == 0 or epoch <= 5:
            probe.eval()
            with torch.no_grad():
                vp = probe(vz)
                vl = F.binary_cross_entropy_with_logits(vp, vo).item()
                vb = (vp > 0).float()
                acc = (vb == vo).float().mean().item()
                om = vo > 0; rec = vb[om].mean().item() if om.any() else 0
            print(f'  Ep {epoch:>3d} | T:{ls/st:.4f} V:{vl:.4f} '
                  f'Acc:{acc:.3f} Rec:{rec:.3f}', flush=True)
            if vl < best_val:
                best_val = vl; best_state = copy.deepcopy(probe.state_dict())

    if best_state: probe.load_state_dict(best_state)
    torch.save(probe.state_dict(), f'{CKPT}/car_probe.pt')
    print(f'Car probe saved!', flush=True)
    return probe


def retrain_dynamics(data, epochs=100):
    print(f'\n=== Retraining Dynamics ({epochs} epochs) ===', flush=True)
    z, z_next = data['z'], data['z_next']
    actions = data['actions']
    n = len(z)

    dynamics = DynamicsPredictor()
    optimizer = AdamW(dynamics.parameters(), lr=3e-4, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        dynamics.train()
        pm = torch.randperm(n)
        ls = 0; st = 0
        for s in range(0, n, 256):
            idx = pm[s:s+256]
            pred = dynamics(z[idx], actions[idx])
            loss = F.mse_loss(pred, z_next[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ls += loss.item(); st += 1

        if epoch % 10 == 0 or epoch <= 3:
            print(f'  Ep {epoch:>3d} | Loss:{ls/st:.6f}', flush=True)

    torch.save(dynamics.state_dict(), f'{CKPT}/dynamics.pt')
    print('Dynamics saved!', flush=True)
    return dynamics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    print('Loading slot encoder (frozen, for encoding only)...')
    config = AgentConfig(); config.num_actions = 5
    encoder = SlotEncoder(config, num_slots=8, slot_dim=64, num_iters=3)
    encoder.load_state_dict(torch.load(
        f'{CKPT}/slot_encoder.pt', map_location='cpu', weights_only=True))
    encoder.eval()

    data = generate_and_encode(encoder, num_episodes=args.episodes)

    retrain_dynamics(data, epochs=100)
    retrain_position_probe(data, epochs=200)
    retrain_car_probe(data, epochs=500)

    print('\n=== All probes retrained on LOCAL latents! ===')
    print('Run: python crosser_agent/live_agent.py --bench --slots --mode pure')
