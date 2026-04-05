"""V-JEPA 2.1 training pipeline.

Upgrades from v1:
1. Dense predictive loss — predict full 7x7 spatial features of next frame
   from masked current features + action
2. Deep self-supervision — position and car probes at both 14x14 and 7x7
   feature map resolutions

Run on GPU: python train_v21_gpu.py
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
from game_agent.models.encoder_v2 import EncoderV2
from game_agent.models.dynamics_dense import DenseDynamicsPredictor, generate_spatial_mask
from game_agent.models.probes_multiscale import SpatialPositionProbe, SpatialCarProbe
from game_agent.preprocessing.transforms import Preprocessor

GRID_COLS, GRID_ROWS = 12, 12
CKPT = 'crosser_agent/checkpoints_v21'
os.makedirs(CKPT, exist_ok=True)


# Global dynamics (backward compat, for inference)
class GlobalDynamics(nn.Module):
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


# MLP position probe (backward compat for inference)
class GlobalPositionProbe(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


# MLP car probe (backward compat for inference)
class GlobalCarProbe(nn.Module):
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
    return grid


# ---- Data Generation ----

def generate_data(num_episodes=200, steps_per_ep=400, subsample=2):
    total = num_episodes * (steps_per_ep // subsample)
    print(f'\n=== Generating Data: {num_episodes} eps x {steps_per_ep} steps '
          f'(subsample={subsample}, ~{total} samples) ===', flush=True)
    game_config = Config(); game_config.headless = True
    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)
    rng = np.random.RandomState(42)
    actions_list = [NOOP, UP, DOWN, LEFT, RIGHT]

    frames = torch.empty(total, 3, 224, 224)
    next_frames = torch.empty(total, 3, 224, 224)
    actions_t = torch.empty(total, dtype=torch.long)
    positions = torch.empty(total, 2)
    occupancy = torch.empty(total, GRID_ROWS, GRID_COLS)

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
                occupancy[idx] = torch.from_numpy(state_to_occupancy(state))
                idx += 1

            obs = result.observation
            if result.done:
                obs = env.reset(seed=rng.randint(0, 999999))

        if (ep + 1) % 50 == 0:
            print(f'  {ep+1}/{num_episodes} ({idx} samples)', flush=True)

    data = {
        'frames': frames[:idx], 'next_frames': next_frames[:idx],
        'actions': actions_t[:idx], 'positions': positions[:idx],
        'occupancy': occupancy[:idx],
    }
    mem_gb = (frames[:idx].nbytes + next_frames[:idx].nbytes) / 1e9
    print(f'Total: {idx} samples ({mem_gb:.1f} GB)', flush=True)
    return data


# ---- Training ----

def train(data, device, epochs=100, load_v1=None):
    print(f'\n=== V-JEPA 2.1 Training ({epochs} epochs) ===', flush=True)
    config = AgentConfig(); config.num_actions = 5

    # ── Models ──
    encoder = EncoderV2(config).to(device)
    if load_v1:
        print(f'Loading v1 weights from {load_v1}...', flush=True)
        v1_sd = torch.load(load_v1, map_location=device, weights_only=True)
        encoder.load_v1_weights(v1_sd)

    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    dense_dynamics = DenseDynamicsPredictor().to(device)
    global_dynamics = GlobalDynamics().to(device)

    # Spatial probes (deep supervision)
    pos_probe_7 = SpatialPositionProbe(in_channels=256).to(device)
    pos_probe_14 = SpatialPositionProbe(in_channels=256).to(device)
    car_probe_7 = SpatialCarProbe(in_channels=256, grid_size=12).to(device)
    car_probe_14 = SpatialCarProbe(in_channels=256, grid_size=12).to(device)

    # Global probes (backward compat, for inference)
    global_pos_probe = GlobalPositionProbe().to(device)
    global_car_probe = GlobalCarProbe().to(device)

    # ── Optimizer ──
    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': 3e-4},
        {'params': dense_dynamics.parameters(), 'lr': 3e-4},
        {'params': global_dynamics.parameters(), 'lr': 3e-4},
        {'params': pos_probe_7.parameters(), 'lr': 5e-4},
        {'params': pos_probe_14.parameters(), 'lr': 5e-4},
        {'params': car_probe_7.parameters(), 'lr': 5e-4},
        {'params': car_probe_14.parameters(), 'lr': 5e-4},
        {'params': global_pos_probe.parameters(), 'lr': 5e-4},
        {'params': global_car_probe.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Print param counts ──
    def pcount(m): return sum(p.numel() for p in m.parameters())
    print(f'EncoderV2: {pcount(encoder):,} | DenseDyn: {pcount(dense_dynamics):,} | '
          f'GlobalDyn: {pcount(global_dynamics):,}', flush=True)
    print(f'SpatialProbes: pos7={pcount(pos_probe_7):,} pos14={pcount(pos_probe_14):,} '
          f'car7={pcount(car_probe_7):,} car14={pcount(car_probe_14):,}', flush=True)
    print(f'GlobalProbes: pos={pcount(global_pos_probe):,} '
          f'car={pcount(global_car_probe):,}', flush=True)

    # ── Data ──
    n = len(data['frames'])
    batch_size = 64
    pos_weight = torch.tensor(3.0, device=device)
    mask_ratio = 0.4

    fr = data['frames']
    nf = data['next_frames']
    act = data['actions']
    pos = data['positions']
    occ = data['occupancy']

    # ── Loss weights ──
    W = {
        'dense_dyn': 2.0,    # core V-JEPA 2.1 signal
        'global_dyn': 0.5,   # backward compat
        'pos_7': 1.0,        # deep supervision
        'pos_14': 0.5,
        'car_7': 0.5,
        'car_14': 0.3,
        'global_pos': 0.5,   # backward compat probes
        'global_car': 0.3,
    }

    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        encoder.train(); dense_dynamics.train(); global_dynamics.train()
        pos_probe_7.train(); pos_probe_14.train()
        car_probe_7.train(); car_probe_14.train()
        global_pos_probe.train(); global_car_probe.train()

        perm = torch.randperm(n)
        ep_losses = {k: 0.0 for k in W}
        steps = 0

        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            f_b = fr[idx].to(device)
            nf_b = nf[idx].to(device)
            a_b = act[idx].to(device)
            p_b = pos[idx].to(device)
            o_b = occ[idx].to(device)  # (B, 12, 12)
            B = f_b.shape[0]

            # ── Forward: multi-scale features ──
            feats = encoder.forward_multiscale(f_b)
            f7 = feats['7x7']     # (B, 256, 7, 7)
            f14 = feats['14x14']  # (B, 256, 14, 14)
            z_global = encoder.pool(encoder.forward_backbone(f_b)['7x7']).flatten(1)
            z_global = encoder.norm(encoder.fc(z_global))  # (B, 256)

            # ── 1. Dense predictive loss ──
            mask = generate_spatial_mask(B, 7, 7, mask_ratio, device)
            pred_next_7 = dense_dynamics(f7, a_b, mask)  # (B, 256, 7, 7)

            with torch.no_grad():
                target_feats = target_encoder.forward_multiscale(nf_b)
                target_7 = target_feats['7x7']
                z_next_target = target_encoder(nf_b)

            dense_dyn_loss = F.mse_loss(pred_next_7, target_7)

            # ── 2. Global dynamics loss ──
            z_next_pred = global_dynamics(z_global, a_b)
            global_dyn_loss = F.mse_loss(z_next_pred, z_next_target)

            # ── 3. Deep self-supervision: position probes ──
            pos_loss_7 = F.mse_loss(pos_probe_7(f7), p_b)
            pos_loss_14 = F.mse_loss(pos_probe_14(f14), p_b)

            # ── 4. Deep self-supervision: car probes ──
            car_loss_7 = F.binary_cross_entropy_with_logits(
                car_probe_7(f7), o_b, pos_weight=pos_weight)
            car_loss_14 = F.binary_cross_entropy_with_logits(
                car_probe_14(f14), o_b, pos_weight=pos_weight)

            # ── 5. Global probes (backward compat) ──
            global_pos_loss = F.mse_loss(global_pos_probe(z_global), p_b)
            global_car_loss = F.binary_cross_entropy_with_logits(
                global_car_probe(z_global), o_b.flatten(1), pos_weight=pos_weight)

            # ── Total loss ──
            loss = (W['dense_dyn'] * dense_dyn_loss
                    + W['global_dyn'] * global_dyn_loss
                    + W['pos_7'] * pos_loss_7
                    + W['pos_14'] * pos_loss_14
                    + W['car_7'] * car_loss_7
                    + W['car_14'] * car_loss_14
                    + W['global_pos'] * global_pos_loss
                    + W['global_car'] * global_car_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for p_o, p_t in zip(encoder.parameters(), target_encoder.parameters()):
                    p_t.data.mul_(0.996).add_(p_o.data, alpha=0.004)

            # Track losses
            ep_losses['dense_dyn'] += dense_dyn_loss.item()
            ep_losses['global_dyn'] += global_dyn_loss.item()
            ep_losses['pos_7'] += pos_loss_7.item()
            ep_losses['pos_14'] += pos_loss_14.item()
            ep_losses['car_7'] += car_loss_7.item()
            ep_losses['car_14'] += car_loss_14.item()
            ep_losses['global_pos'] += global_pos_loss.item()
            ep_losses['global_car'] += global_car_loss.item()
            steps += 1

        scheduler.step()

        avg_total = sum(ep_losses[k] / steps for k in ep_losses)
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(encoder.state_dict(), f'{CKPT}/encoder_v21_best.pt')

        if epoch % 5 == 0 or epoch <= 3:
            elapsed = time.time() - t0
            dd = ep_losses['dense_dyn'] / steps
            gd = ep_losses['global_dyn'] / steps
            p7 = ep_losses['pos_7'] / steps
            p14 = ep_losses['pos_14'] / steps
            c7 = ep_losses['car_7'] / steps
            c14 = ep_losses['car_14'] / steps
            gp = ep_losses['global_pos'] / steps
            gc = ep_losses['global_car'] / steps
            print(f'Ep {epoch:>3d}/{epochs} | DDyn:{dd:.4f} GDyn:{gd:.5f} | '
                  f'Pos7:{p7:.5f} Pos14:{p14:.5f} | '
                  f'Car7:{c7:.3f} Car14:{c14:.3f} | '
                  f'GPos:{gp:.5f} GCar:{gc:.3f} | {elapsed:.0f}s', flush=True)

    # ── Save all models ──
    torch.save(encoder.state_dict(), f'{CKPT}/encoder_v21.pt')
    torch.save(target_encoder.state_dict(), f'{CKPT}/target_encoder_v21.pt')
    torch.save(dense_dynamics.state_dict(), f'{CKPT}/dense_dynamics.pt')
    torch.save(global_dynamics.state_dict(), f'{CKPT}/dynamics.pt')
    torch.save(pos_probe_7.state_dict(), f'{CKPT}/pos_probe_7x7.pt')
    torch.save(pos_probe_14.state_dict(), f'{CKPT}/pos_probe_14x14.pt')
    torch.save(car_probe_7.state_dict(), f'{CKPT}/car_probe_7x7.pt')
    torch.save(car_probe_14.state_dict(), f'{CKPT}/car_probe_14x14.pt')
    torch.save(global_pos_probe.state_dict(), f'{CKPT}/position_probe.pt')
    torch.save(global_car_probe.state_dict(), f'{CKPT}/car_probe.pt')
    print(f'\nAll models saved to {CKPT}/', flush=True)
    print(f'Best total loss: {best_loss:.4f}', flush=True)

    return encoder


# ---- Main ----

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--load-v1', type=str, default=None,
                        help='Path to v1 encoder.pt for fine-tuning')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Device: {torch.cuda.get_device_name(0)}', flush=True)
    else:
        print('WARNING: Running on CPU!', flush=True)

    data = generate_data(num_episodes=args.episodes, steps_per_ep=400)
    encoder = train(data, device, epochs=args.epochs, load_v1=args.load_v1)

    print('\n=== V-JEPA 2.1 Training Complete! ===', flush=True)
