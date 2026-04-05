"""Slot Attention V-JEPA training pipeline on GPU.

1. Generate training data (reuses existing pipeline)
2. Train SlotEncoder + SlotDecoder with reconstruction loss
3. Train position probe on aggregated slot representation
4. Train car detection from slot attention masks (near-free)
5. Train dynamics on aggregated latent (backward compat)

Run on GPU server: python train_slots_gpu.py
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import sys, copy, gc, time
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
from game_agent.models.slot_attention import SlotEncoder, SlotDecoder
from game_agent.preprocessing.transforms import Preprocessor

GRID_COLS, GRID_ROWS = 12, 12
CKPT = 'crosser_agent/checkpoints_slots'
os.makedirs(CKPT, exist_ok=True)

NUM_SLOTS = 8
SLOT_DIM = 64
SLOT_ITERS = 3


# ---- Models (dynamics/probes, same as before) ----

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


def state_to_occupancy(state):
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    for car in state.cars:
        height = getattr(car, 'height', 1)
        for r in range(car.row, min(GRID_ROWS, car.row + height)):
            if 0 <= r < GRID_ROWS:
                for col in range(max(0, int(np.floor(car.x))),
                                 min(GRID_COLS, int(np.ceil(car.x + car.width)))):
                    grid[r, col] = 1.0
    return grid.flatten()


# ---- Step 1: Generate Data (same as train_full_gpu) ----

def generate_data(num_episodes=200, steps_per_ep=400, subsample=2):
    """Generate training data. subsample=2 keeps every other frame to save RAM."""
    total = num_episodes * (steps_per_ep // subsample)
    print(f'\n=== Generating Data: {num_episodes} eps x {steps_per_ep} steps '
          f'(subsample={subsample}, ~{total} samples) ===', flush=True)
    game_config = Config(); game_config.headless = True
    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)
    rng = np.random.RandomState(42)
    actions_list = [NOOP, UP, DOWN, LEFT, RIGHT]

    # Pre-allocate tensors to avoid list + stack OOM
    frames = torch.empty(total, 3, 224, 224)
    next_frames = torch.empty(total, 3, 224, 224)
    actions_t = torch.empty(total, dtype=torch.long)
    positions = torch.empty(total, 2)
    rewards = torch.empty(total)
    occupancy = torch.empty(total, GRID_ROWS * GRID_COLS)

    idx = 0
    for ep in range(num_episodes):
        env = CrosserEnv(game_config)
        obs = env.reset(seed=rng.randint(0, 999999))

        for step in range(steps_per_ep):
            state = env._state
            action = rng.choice(actions_list)
            result = env.step(action)

            if step % subsample == 0 and idx < total:
                frames[idx] = preprocessor(obs.frame)
                next_frames[idx] = preprocessor(result.observation.frame)
                actions_t[idx] = action
                positions[idx, 0] = state.player.col / (game_config.grid_cols - 1)
                positions[idx, 1] = state.player.row / (game_config.grid_rows - 1)
                rewards[idx] = result.reward
                occ = state_to_occupancy(state)
                occupancy[idx] = torch.from_numpy(occ)
                idx += 1

            obs = result.observation
            if result.done:
                obs = env.reset(seed=rng.randint(0, 999999))

        if (ep + 1) % 50 == 0:
            print(f'  {ep+1}/{num_episodes} ({idx} samples)', flush=True)

    # Trim if fewer samples than expected
    frames = frames[:idx]
    next_frames = next_frames[:idx]
    actions_t = actions_t[:idx]
    positions = positions[:idx]
    rewards = rewards[:idx]
    occupancy = occupancy[:idx]

    data = {
        'frames': frames,
        'next_frames': next_frames,
        'actions': actions_t,
        'positions': positions,
        'rewards': rewards,
        'occupancy': occupancy,
    }
    mem_gb = (frames.nbytes + next_frames.nbytes) / 1e9
    print(f'Total: {idx} samples ({mem_gb:.1f} GB for frames)', flush=True)
    return data


# ---- Step 2: Train Slot Encoder (reconstruction + V-JEPA) ----

def train_slot_encoder(data, device, epochs=100):
    print(f'\n=== Training Slot Encoder ({epochs} epochs) ===', flush=True)
    config = AgentConfig(); config.num_actions = 5

    encoder = SlotEncoder(config, num_slots=NUM_SLOTS, slot_dim=SLOT_DIM,
                          num_iters=SLOT_ITERS).to(device)
    decoder = SlotDecoder(slot_dim=SLOT_DIM, output_size=224,
                          output_channels=3).to(device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    dynamics = DynamicsPredictor().to(device)
    position_probe = PositionProbe().to(device)
    car_probe_head = CarOccupancyProbe().to(device)

    # Separate param groups — decoder gets higher LR
    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': 3e-4},
        {'params': decoder.parameters(), 'lr': 4e-4},
        {'params': dynamics.parameters(), 'lr': 3e-4},
        {'params': position_probe.parameters(), 'lr': 3e-4},
        {'params': car_probe_head.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(data['frames'])
    batch_size = 32  # CNN decoder uses more VRAM per slot
    pos_weight = torch.tensor(3.0, device=device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    dyn_params = sum(p.numel() for p in dynamics.parameters())
    car_params = sum(p.numel() for p in car_probe_head.parameters())
    print(f'SlotEncoder: {enc_params:,}  Decoder: {dec_params:,}  '
          f'Dynamics: {dyn_params:,}  CarProbe: {car_params:,}', flush=True)

    frames = data['frames']
    next_frames = data['next_frames']
    actions_t = data['actions']
    positions = data['positions']
    occupancy = data['occupancy']

    # Loss weights
    w_recon = 1.0
    w_dyn = 1.0
    w_pos = 1.0
    w_occ = 0.5  # occupancy auxiliary — forces encoder to represent cars

    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        encoder.train(); decoder.train(); dynamics.train()
        position_probe.train(); car_probe_head.train()
        perm = torch.randperm(n)
        ep_recon = 0; ep_dyn = 0; ep_pos = 0; ep_occ = 0; steps = 0

        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            f_batch = frames[idx].to(device)
            nf_batch = next_frames[idx].to(device)
            a_batch = actions_t[idx].to(device)
            p_batch = positions[idx].to(device)
            o_batch = occupancy[idx].to(device)

            # Forward through slot encoder
            slots, attn = encoder.forward_slots(f_batch)
            z_t = encoder.aggregate(slots.flatten(1))  # compat latent

            # Reconstruction loss
            recon, masks = decoder(slots)
            recon_loss = F.mse_loss(recon, f_batch)

            # V-JEPA dynamics loss
            with torch.no_grad():
                _, _ = target_encoder.forward_slots(nf_batch)
                z_next_target = target_encoder(nf_batch)
            z_next_pred = dynamics(z_t, a_batch)
            dyn_loss = F.mse_loss(z_next_pred, z_next_target)

            # Position probe loss
            pos_pred = position_probe(z_t)
            pos_loss = F.mse_loss(pos_pred, p_batch)

            # Car occupancy loss — forces encoder to represent car positions
            occ_logits = car_probe_head(z_t)
            occ_loss = F.binary_cross_entropy_with_logits(
                occ_logits, o_batch, pos_weight=pos_weight)

            loss = (w_recon * recon_loss + w_dyn * dyn_loss
                    + w_pos * pos_loss + w_occ * occ_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            # EMA update target encoder
            with torch.no_grad():
                for p_o, p_t in zip(encoder.parameters(), target_encoder.parameters()):
                    p_t.data.mul_(0.996).add_(p_o.data, alpha=0.004)

            ep_recon += recon_loss.item()
            ep_dyn += dyn_loss.item()
            ep_pos += pos_loss.item()
            ep_occ += occ_loss.item()
            steps += 1

        scheduler.step()

        avg_total = (ep_recon + ep_dyn + ep_pos + ep_occ) / steps
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(encoder.state_dict(), f'{CKPT}/slot_encoder_best.pt')
            torch.save(decoder.state_dict(), f'{CKPT}/slot_decoder_best.pt')

        if epoch % 5 == 0 or epoch <= 3:
            elapsed = time.time() - t0
            print(f'Ep {epoch:>3d}/{epochs} | Recon: {ep_recon/steps:.4f} | '
                  f'Dyn: {ep_dyn/steps:.6f} | Pos: {ep_pos/steps:.6f} | '
                  f'Occ: {ep_occ/steps:.4f} | {elapsed:.0f}s', flush=True)

    # Save final
    torch.save(encoder.state_dict(), f'{CKPT}/slot_encoder.pt')
    torch.save(decoder.state_dict(), f'{CKPT}/slot_decoder.pt')
    torch.save(target_encoder.state_dict(), f'{CKPT}/slot_target_encoder.pt')
    torch.save(dynamics.state_dict(), f'{CKPT}/dynamics.pt')
    torch.save(position_probe.state_dict(), f'{CKPT}/position_probe.pt')
    torch.save(car_probe_head.state_dict(), f'{CKPT}/car_probe.pt')
    print(f'Slot encoder training done! Best loss: {best_loss:.4f}', flush=True)
    return encoder, decoder, dynamics, position_probe


# ---- Step 3: Train Car Probe (on aggregated latent) ----

def train_car_probe(encoder, data, device, epochs=300):
    print(f'\n=== Training Car Occupancy Probe ({epochs} epochs) ===', flush=True)
    encoder.eval()

    frames = data['frames']
    occupancy = data['occupancy'].to(device)
    n = len(frames)

    # Pre-encode all frames
    print('Pre-encoding with slot encoder...', flush=True)
    all_z = []
    with torch.no_grad():
        for s in range(0, n, 256):
            batch = frames[s:s+256].to(device)
            z = encoder(batch)
            all_z.append(z)
    all_z = torch.cat(all_z)  # (N, 256)
    print(f'Encoded: {all_z.shape}', flush=True)

    # Train/val split
    perm = torch.randperm(n)
    train_n = int(n * 0.9)
    tz, vz = all_z[perm[:train_n]], all_z[perm[train_n:]]
    to, vo = occupancy[perm[:train_n]], occupancy[perm[train_n:]]

    pos_weight = torch.tensor(3.0, device=device)
    probe = CarOccupancyProbe().to(device)
    optimizer = AdamW(probe.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f'Car probe: {sum(p.numel() for p in probe.parameters()):,} params', flush=True)

    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        probe.train()
        pm = torch.randperm(train_n, device=device)
        ls = 0; st = 0
        for s in range(0, train_n, 512):
            idx = pm[s:s+512]
            pred = probe(tz[idx])
            loss = F.binary_cross_entropy_with_logits(pred, to[idx], pos_weight=pos_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ls += loss.item(); st += 1
        scheduler.step()

        if epoch % 25 == 0 or epoch <= 5:
            probe.eval()
            with torch.no_grad():
                vp = probe(vz)
                vl = F.binary_cross_entropy_with_logits(vp, vo).item()
                vb = (vp > 0).float()
                acc = (vb == vo).float().mean().item()
                om = vo > 0; rec = vb[om].mean().item() if om.any() else 0
                em = vo == 0; ea = (1 - vb[em]).mean().item()
            print(f'Ep {epoch:>3d} | T:{ls/st:.4f} V:{vl:.4f} '
                  f'Acc:{acc:.3f} Rec:{rec:.3f} Emp:{ea:.3f}', flush=True)
            if vl < best_val:
                best_val = vl
                best_state = copy.deepcopy(probe.state_dict())

    if best_state:
        probe.load_state_dict(best_state)
    torch.save(probe.state_dict(), f'{CKPT}/car_probe.pt')
    print('Car probe saved!', flush=True)
    return probe, all_z


# ---- Step 4: Evaluate Slot Masks for Car Detection ----

def evaluate_slot_masks(encoder, data, device, num_samples=1000):
    """Check if slot attention masks naturally segment cars vs player vs background."""
    print(f'\n=== Evaluating Slot Attention Masks ===', flush=True)
    encoder.eval()

    frames = data['frames']
    occupancy = data['occupancy']
    positions = data['positions']

    indices = torch.randperm(len(frames))[:num_samples]

    # Collect attention maps and ground truth
    all_attn = []
    all_occ = []
    all_pos = []

    with torch.no_grad():
        for s in range(0, num_samples, 128):
            idx = indices[s:s+128]
            batch = frames[idx].to(device)
            slots, attn = encoder.forward_slots(batch)  # attn: (B, K, 49)
            all_attn.append(attn.cpu())
            all_occ.append(occupancy[idx])
            all_pos.append(positions[idx])

    all_attn = torch.cat(all_attn)   # (N, 20, 49)
    all_occ = torch.cat(all_occ)     # (N, 144)
    all_pos = torch.cat(all_pos)     # (N, 2)

    # Reshape attention to 7x7 grid
    attn_7x7 = all_attn.reshape(-1, NUM_SLOTS, 7, 7)

    # Upsample to 12x12 for comparison with occupancy grid
    attn_12x12 = F.interpolate(attn_7x7, size=(12, 12), mode='bilinear',
                                align_corners=False)  # (N, 20, 12, 12)

    occ_grid = all_occ.reshape(-1, 12, 12)  # (N, 12, 12)

    # For each slot, compute correlation with car occupancy
    print('\nSlot-car correlations (top 5):')
    correlations = []
    for k in range(NUM_SLOTS):
        slot_map = attn_12x12[:, k]  # (N, 12, 12)
        # Correlation with car occupancy
        corr = torch.corrcoef(torch.stack([
            slot_map.flatten(), occ_grid.flatten()
        ]))[0, 1].item()
        correlations.append((k, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for k, corr in correlations[:5]:
        avg_attn = attn_12x12[:, k].mean(0)  # (12, 12)
        print(f'  Slot {k:>2d}: corr={corr:+.3f}  '
              f'mean_attn={avg_attn.mean():.3f}  '
              f'max_attn={avg_attn.max():.3f}')

    # Check if any slot tracks the player
    print('\nSlot-player correlations (top 3):')
    player_correlations = []
    for k in range(NUM_SLOTS):
        slot_map = attn_12x12[:, k]  # (N, 12, 12)
        # Player position as a heatmap
        player_grid = torch.zeros_like(occ_grid)
        for i in range(len(all_pos)):
            pc = int(all_pos[i, 0] * 11)
            pr = int(all_pos[i, 1] * 11)
            pc = max(0, min(11, pc))
            pr = max(0, min(11, pr))
            player_grid[i, pr, pc] = 1.0
        corr = torch.corrcoef(torch.stack([
            slot_map.flatten(), player_grid.flatten()
        ]))[0, 1].item()
        player_correlations.append((k, corr))

    player_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for k, corr in player_correlations[:3]:
        print(f'  Slot {k:>2d}: corr={corr:+.3f}')

    print('\nSlot mask evaluation done!', flush=True)


# ---- Main ----

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Device: {torch.cuda.get_device_name(0)}', flush=True)
    else:
        print('WARNING: Running on CPU — this will be slow!', flush=True)

    # Step 1: Generate data
    data = generate_data(num_episodes=200, steps_per_ep=400)

    # Step 2: Train slot encoder (reconstruction + V-JEPA + position + occupancy)
    encoder, decoder, dynamics, pos_probe = train_slot_encoder(
        data, device, epochs=100)

    # Step 3: Refine car probe on slot-based latents (already pretrained in step 2)
    car_probe, all_z = train_car_probe(encoder, data, device, epochs=200)

    # Step 4: Evaluate if slots naturally segment objects
    evaluate_slot_masks(encoder, data, device)

    # Quick dynamics test
    encoder.eval(); dynamics.eval(); pos_probe.eval()
    with torch.no_grad():
        z = all_z[0:1]
        for n_steps in [1, 5, 10]:
            results = {}
            for a, name in [(0,'NOOP'),(1,'UP'),(2,'DOWN'),(3,'LEFT'),(4,'RIGHT')]:
                z_sim = z.clone()
                for _ in range(n_steps):
                    z_sim = dynamics(z_sim, torch.tensor([a], device=device))
                p = pos_probe(z_sim)[0]
                results[name] = (p[0].item(), p[1].item())
            print(f'Step {n_steps}: ' + ' | '.join(
                f'{k}=({v[0]:.3f},{v[1]:.3f})' for k, v in results.items()),
                flush=True)

    print(f'\n=== All training complete! ===', flush=True)
    print(f'Checkpoints saved to: {CKPT}/', flush=True)
