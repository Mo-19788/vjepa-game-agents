"""Local V2 training — memory efficient, no DataLoader shared memory issues.

Processes one chunk at a time, never loads all frames into RAM.
"""

import sys, os, glob, copy
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from game_agent.models.encoder import Encoder
from game_agent.models.dynamics import DynamicsPredictor
from game_agent.models.reward_head import RewardHead
from game_agent.config import AgentConfig
from game_agent.planning.shooting import PositionProbe
from game_agent.preprocessing.transforms import Preprocessor


def preprocess_batch(frames, preprocessor):
    """Preprocess a batch of numpy frames to tensor."""
    return torch.stack([preprocessor(f) for f in frames])


def train():
    config = AgentConfig()
    device = torch.device('cpu')
    output_dir = 'game_agent/checkpoints_v2'
    os.makedirs(output_dir, exist_ok=True)
    chunk_dir = 'kaggle_training/pong_v2_visual_chunks'
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.npz')))

    preprocessor = Preprocessor(config)

    # Models
    encoder = Encoder(config).to(device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    dynamics = DynamicsPredictor(config).to(device)
    position_probe = PositionProbe(config).to(device)

    params = list(encoder.parameters()) + list(dynamics.parameters()) + list(position_probe.parameters())
    optimizer = AdamW(params, lr=3e-4, weight_decay=1e-4)

    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"Dynamics: {sum(p.numel() for p in dynamics.parameters()):,} params")
    print(f"Probe: {sum(p.numel() for p in position_probe.parameters()):,} params")

    # ========== Phase 1: World Model (20 epochs) ==========
    print("\n=== Phase 1: World Model (20 epochs) ===", flush=True)

    for epoch in range(1, 21):
        encoder.train(); dynamics.train(); position_probe.train()
        epoch_dyn_loss = 0; epoch_pos_loss = 0; epoch_steps = 0

        for cf in chunk_files:
            d = np.load(cf)
            step = 4  # subsample to save memory
            frames = d['obs'][::step]
            next_frames = d['next_obs'][::step]
            actions = d['action'][::step]
            positions = d['positions'][::step]

            # Process in small batches
            for s in range(0, len(frames), 32):
                obs_t = preprocess_batch(frames[s:s+32], preprocessor).to(device)
                next_t = preprocess_batch(next_frames[s:s+32], preprocessor).to(device)
                act = torch.tensor(actions[s:s+32], dtype=torch.long, device=device)
                pos = torch.tensor(positions[s:s+32, :4], dtype=torch.float32, device=device)

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

                # EMA update
                with torch.no_grad():
                    for p_o, p_t in zip(encoder.parameters(), target_encoder.parameters()):
                        p_t.data.mul_(0.996).add_(p_o.data, alpha=0.004)

                epoch_dyn_loss += dyn_loss.item()
                epoch_pos_loss += pos_loss.item()
                epoch_steps += 1

            del d, frames, next_frames

        dyn = epoch_dyn_loss / epoch_steps
        pos = epoch_pos_loss / epoch_steps
        print(f"Epoch {epoch}/20 | Dyn: {dyn:.6f} | Pos: {pos:.6f}", flush=True)

    torch.save(encoder.state_dict(), os.path.join(output_dir, 'encoder.pt'))
    torch.save(dynamics.state_dict(), os.path.join(output_dir, 'dynamics.pt'))
    torch.save(target_encoder.state_dict(), os.path.join(output_dir, 'target_encoder.pt'))
    torch.save(position_probe.state_dict(), os.path.join(output_dir, 'position_probe.pt'))
    print("World model saved!", flush=True)

    # ========== Phase 2: Reward Head (50 epochs, frozen encoder) ==========
    print("\n=== Phase 2: Reward Head (50 epochs) ===", flush=True)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    reward_head = RewardHead(config).to(device)
    rh_optimizer = AdamW(reward_head.parameters(), lr=3e-4)

    for epoch in range(1, 51):
        reward_head.train()
        epoch_loss = 0; epoch_steps = 0

        for cf in chunk_files:
            d = np.load(cf)
            step = 4
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
                epoch_steps += 1

            del d, frames, rewards

        if epoch % 5 == 0 or epoch <= 3:
            print(f"Epoch {epoch}/50 | Loss: {epoch_loss/epoch_steps:.6f}", flush=True)

    torch.save(reward_head.state_dict(), os.path.join(output_dir, 'reward_head.pt'))
    print("Reward head saved!", flush=True)

    # ========== Phase 3: Bigger Probe (100 epochs, pre-encoded) ==========
    print("\n=== Phase 3: Probe refinement (100 epochs) ===", flush=True)

    # Pre-encode to latents
    print("Pre-encoding...", flush=True)
    all_z = []; all_pos = []
    for i, cf in enumerate(chunk_files):
        d = np.load(cf)
        step = 4
        frames = d['obs'][::step]
        positions = d['positions'][::step, :4]
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

    # Fresh bigger probe
    probe = PositionProbe(config).to(device)
    probe_optimizer = AdamW(probe.parameters(), lr=1e-3)

    n = len(all_z)
    train_n = int(n * 0.9)
    perm = torch.randperm(n)
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
            probe_optimizer.zero_grad(); loss.backward(); probe_optimizer.step()
            loss_sum += loss.item(); steps += 1

        probe.eval()
        with torch.no_grad():
            vl = F.mse_loss(probe(val_z), val_p).item()

        if epoch % 10 == 0 or epoch <= 3:
            print(f"Probe {epoch}/100 | Train: {loss_sum/steps:.6f} | Val: {vl:.6f}", flush=True)

    torch.save(probe.state_dict(), os.path.join(output_dir, 'position_probe.pt'))
    print("\n=== All V2 training complete! ===", flush=True)


if __name__ == "__main__":
    train()
