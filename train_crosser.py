"""Train V-JEPA world model for Street Crosser.

Memory-efficient: processes one chunk at a time.
5 actions, 2D player position probe (col, row).
"""

import sys, os, glob, copy
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from game_agent.models.encoder import Encoder
from game_agent.config import AgentConfig
from game_agent.preprocessing.transforms import Preprocessor


class DynamicsPredictor(nn.Module):
    """Action-conditioned dynamics for 5 actions."""
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
    """Predicts player_col, player_row (normalized 0-1) from latent."""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # col, row
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class RewardHead(nn.Module):
    """Predicts reward from latent."""
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


def preprocess_batch(frames, preprocessor):
    return torch.stack([preprocessor(f) for f in frames])


def train():
    config = AgentConfig()
    config.num_actions = 5  # NOOP, UP, DOWN, LEFT, RIGHT
    device = torch.device('cpu')
    output_dir = 'crosser_agent/checkpoints'
    os.makedirs(output_dir, exist_ok=True)
    chunk_dir = 'crosser_agent/training_chunks'
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.npz')))

    if not chunk_files:
        print("No training chunks found! Run generate_training_data.py first.")
        return

    preprocessor = Preprocessor(config)

    # Models
    encoder = Encoder(config).to(device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    dynamics = DynamicsPredictor(latent_dim=config.latent_dim, num_actions=5).to(device)
    position_probe = PositionProbe(latent_dim=config.latent_dim).to(device)

    params = list(encoder.parameters()) + list(dynamics.parameters()) + list(position_probe.parameters())
    optimizer = AdamW(params, lr=3e-4, weight_decay=1e-4)

    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"Dynamics: {sum(p.numel() for p in dynamics.parameters()):,} params")
    print(f"Probe: {sum(p.numel() for p in position_probe.parameters()):,} params")
    print(f"Chunks: {len(chunk_files)}")

    # ========== Phase 1: World Model (30 epochs) ==========
    print("\n=== Phase 1: World Model (30 epochs) ===", flush=True)

    for epoch in range(1, 31):
        encoder.train(); dynamics.train(); position_probe.train()
        epoch_dyn = 0; epoch_pos = 0; steps = 0

        for cf in chunk_files:
            d = np.load(cf)
            step = 2  # subsample
            frames = d['obs'][::step]
            next_frames = d['next_obs'][::step]
            actions = d['action'][::step]
            positions = d['positions'][::step]

            for s in range(0, len(frames), 32):
                obs_t = preprocess_batch(frames[s:s+32], preprocessor).to(device)
                next_t = preprocess_batch(next_frames[s:s+32], preprocessor).to(device)
                act = torch.tensor(actions[s:s+32], dtype=torch.long, device=device)
                pos = torch.tensor(positions[s:s+32, :2], dtype=torch.float32, device=device)

                z_t = encoder(obs_t)
                with torch.no_grad():
                    z_next_target = target_encoder(next_t)

                z_next_pred = dynamics(z_t, act)
                dyn_loss = F.mse_loss(z_next_pred, z_next_target)

                pos_pred = position_probe(z_t)
                pos_loss = F.mse_loss(pos_pred, pos)

                loss = dyn_loss + pos_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for p_o, p_t in zip(encoder.parameters(), target_encoder.parameters()):
                        p_t.data.mul_(0.996).add_(p_o.data, alpha=0.004)

                epoch_dyn += dyn_loss.item()
                epoch_pos += pos_loss.item()
                steps += 1

            del d

        print(f"Epoch {epoch}/30 | Dyn: {epoch_dyn/steps:.6f} | Pos: {epoch_pos/steps:.6f}", flush=True)

    torch.save(encoder.state_dict(), os.path.join(output_dir, 'encoder.pt'))
    torch.save(dynamics.state_dict(), os.path.join(output_dir, 'dynamics.pt'))
    torch.save(target_encoder.state_dict(), os.path.join(output_dir, 'target_encoder.pt'))
    torch.save(position_probe.state_dict(), os.path.join(output_dir, 'position_probe.pt'))
    print("World model saved!", flush=True)

    # ========== Phase 2: Reward Head (50 epochs) ==========
    print("\n=== Phase 2: Reward Head (50 epochs) ===", flush=True)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    reward_head = RewardHead(latent_dim=config.latent_dim).to(device)
    rh_optimizer = AdamW(reward_head.parameters(), lr=3e-4)

    for epoch in range(1, 51):
        reward_head.train()
        epoch_loss = 0; steps = 0

        for cf in chunk_files:
            d = np.load(cf)
            step = 2
            frames = d['obs'][::step]
            rewards = d['reward'][::step]

            for s in range(0, len(frames), 32):
                obs_t = preprocess_batch(frames[s:s+32], preprocessor).to(device)
                rew = torch.tensor(rewards[s:s+32], dtype=torch.float32, device=device)

                with torch.no_grad():
                    z = encoder(obs_t)
                pred = reward_head(z)
                loss = F.mse_loss(pred, rew)
                rh_optimizer.zero_grad()
                loss.backward()
                rh_optimizer.step()

                epoch_loss += loss.item()
                steps += 1
            del d

        if epoch % 5 == 0 or epoch <= 3:
            print(f"Epoch {epoch}/50 | Loss: {epoch_loss/steps:.6f}", flush=True)

    torch.save(reward_head.state_dict(), os.path.join(output_dir, 'reward_head.pt'))
    print("Reward head saved!", flush=True)

    # ========== Phase 3: Probe refinement (100 epochs on pre-encoded latents) ==========
    print("\n=== Phase 3: Probe refinement (100 epochs) ===", flush=True)

    print("Pre-encoding...", flush=True)
    all_z = []; all_pos = []
    for i, cf in enumerate(chunk_files):
        d = np.load(cf)
        step = 2
        frames = d['obs'][::step]
        positions = d['positions'][::step, :2]
        print(f"  Chunk {i+1}", flush=True)
        for s in range(0, len(frames), 32):
            obs_t = preprocess_batch(frames[s:s+32], preprocessor).to(device)
            with torch.no_grad():
                all_z.append(encoder(obs_t))
            all_pos.append(torch.tensor(positions[s:s+32], dtype=torch.float32))
        del d; import gc; gc.collect()

    all_z = torch.cat(all_z)
    all_pos = torch.cat(all_pos)
    del preprocessor; import gc; gc.collect()
    print(f"Encoded {len(all_z)} samples", flush=True)

    probe = PositionProbe(latent_dim=config.latent_dim).to(device)
    probe_opt = AdamW(probe.parameters(), lr=1e-3)

    n = len(all_z)
    perm = torch.randperm(n)
    train_n = int(n * 0.9)
    train_z, val_z = all_z[perm[:train_n]], all_z[perm[train_n:]]
    train_p, val_p = all_pos[perm[:train_n]], all_pos[perm[train_n:]]

    for epoch in range(1, 101):
        probe.train()
        perm2 = torch.randperm(train_n)
        loss_sum = 0; steps = 0
        for s in range(0, train_n, 256):
            idx = perm2[s:s+256]
            pred = probe(train_z[idx])
            loss = F.mse_loss(pred, train_p[idx])
            probe_opt.zero_grad(); loss.backward(); probe_opt.step()
            loss_sum += loss.item(); steps += 1

        probe.eval()
        with torch.no_grad():
            vl = F.mse_loss(probe(val_z), val_p).item()

        if epoch % 10 == 0 or epoch <= 3:
            print(f"Probe {epoch}/100 | Train: {loss_sum/steps:.6f} | Val: {vl:.6f}", flush=True)

    torch.save(probe.state_dict(), os.path.join(output_dir, 'position_probe.pt'))
    print("\n=== All crosser training complete! ===", flush=True)

    # Quick test
    print("\nTesting dynamics spread...")
    dynamics.eval(); probe.eval()
    with torch.no_grad():
        z = all_z[0:1]
        for steps in [1, 5, 10]:
            results = {}
            for a, name in [(0, 'NOOP'), (1, 'UP'), (2, 'DOWN'), (3, 'LEFT'), (4, 'RIGHT')]:
                z_sim = z.clone()
                for _ in range(steps):
                    z_sim = dynamics(z_sim, torch.tensor([a]))
                p = probe(z_sim)[0]
                results[name] = (p[0].item(), p[1].item())
            print(f"  Step {steps}: " + " | ".join(f"{k}=({v[0]:.3f},{v[1]:.3f})" for k, v in results.items()))


if __name__ == "__main__":
    train()
