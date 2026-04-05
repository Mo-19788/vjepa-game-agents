"""Full V-JEPA training pipeline on GPU.

1. Generate training data (game frames + positions + car occupancy)
2. Train world model (encoder + dynamics + position probe)
3. Train reward head
4. Train car occupancy probe

All on GPU for speed.
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import sys, copy, gc
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
from game_agent.models.encoder import Encoder
from game_agent.preprocessing.transforms import Preprocessor

GRID_COLS, GRID_ROWS = 12, 12
CKPT = 'crosser_agent/checkpoints'
os.makedirs(CKPT, exist_ok=True)


# ---- Models ----

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


class RewardHead(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
    def forward(self, z): return self.net(z).squeeze(-1)


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
        row = car.row
        if 0 <= row < GRID_ROWS:
            for col in range(max(0, int(np.floor(car.x))), min(GRID_COLS, int(np.ceil(car.x + car.width)))):
                grid[row, col] = 1.0
    return grid.flatten()


# ---- Step 1: Generate Data ----

def generate_data(num_episodes=200, steps_per_ep=400):
    print(f'\n=== Generating Data: {num_episodes} eps x {steps_per_ep} steps ===', flush=True)
    game_config = Config(); game_config.headless = True
    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)
    rng = np.random.RandomState(42)
    actions = [NOOP, UP, DOWN, LEFT, RIGHT]

    all_frames = []
    all_next_frames = []
    all_actions = []
    all_positions = []
    all_rewards = []
    all_occupancy = []

    for ep in range(num_episodes):
        env = CrosserEnv(game_config)
        obs = env.reset(seed=rng.randint(0, 999999))

        for step in range(steps_per_ep):
            state = env._state
            frame_t = preprocessor(obs.frame)
            pos = np.array([
                state.player.col / (game_config.grid_cols - 1),
                state.player.row / (game_config.grid_rows - 1)
            ], dtype=np.float32)
            occ = state_to_occupancy(state)

            action = rng.choice(actions)
            result = env.step(action)
            next_frame_t = preprocessor(result.observation.frame)

            all_frames.append(frame_t)
            all_next_frames.append(next_frame_t)
            all_actions.append(action)
            all_positions.append(pos)
            all_rewards.append(result.reward)
            all_occupancy.append(occ)

            obs = result.observation
            if result.done:
                obs = env.reset(seed=rng.randint(0, 999999))

        if (ep + 1) % 50 == 0:
            print(f'  {ep+1}/{num_episodes} ({len(all_frames)} samples)', flush=True)

    data = {
        'frames': torch.stack(all_frames),
        'next_frames': torch.stack(all_next_frames),
        'actions': torch.tensor(all_actions, dtype=torch.long),
        'positions': torch.tensor(np.array(all_positions)),
        'rewards': torch.tensor(all_rewards, dtype=torch.float32),
        'occupancy': torch.tensor(np.array(all_occupancy)),
    }
    print(f'Total: {len(all_frames)} samples', flush=True)
    return data


# ---- Step 2: Train World Model ----

def train_world_model(data, device, epochs=50):
    print(f'\n=== Training World Model ({epochs} epochs) ===', flush=True)
    config = AgentConfig(); config.num_actions = 5

    encoder = Encoder(config).to(device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters(): p.requires_grad = False
    dynamics = DynamicsPredictor().to(device)
    position_probe = PositionProbe().to(device)

    params = list(encoder.parameters()) + list(dynamics.parameters()) + list(position_probe.parameters())
    optimizer = AdamW(params, lr=3e-4, weight_decay=1e-4)

    n = len(data['frames'])
    batch_size = 128

    print(f'Encoder: {sum(p.numel() for p in encoder.parameters()):,}', flush=True)
    print(f'Dynamics: {sum(p.numel() for p in dynamics.parameters()):,}', flush=True)
    print(f'Probe: {sum(p.numel() for p in position_probe.parameters()):,}', flush=True)

    # Keep data on CPU, move batches to GPU to avoid OOM
    frames = data['frames']
    next_frames = data['next_frames']
    actions_t = data['actions']
    positions = data['positions']

    for epoch in range(1, epochs + 1):
        encoder.train(); dynamics.train(); position_probe.train()
        perm = torch.randperm(n)
        epoch_dyn = 0; epoch_pos = 0; steps = 0

        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            f_batch = frames[idx].to(device)
            nf_batch = next_frames[idx].to(device)
            a_batch = actions_t[idx].to(device)
            p_batch = positions[idx].to(device)

            z_t = encoder(f_batch)
            with torch.no_grad():
                z_next_target = target_encoder(nf_batch)

            z_next_pred = dynamics(z_t, a_batch)
            dyn_loss = F.mse_loss(z_next_pred, z_next_target)

            pos_pred = position_probe(z_t)
            pos_loss = F.mse_loss(pos_pred, p_batch)

            loss = dyn_loss + pos_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            with torch.no_grad():
                for p_o, p_t in zip(encoder.parameters(), target_encoder.parameters()):
                    p_t.data.mul_(0.996).add_(p_o.data, alpha=0.004)

            epoch_dyn += dyn_loss.item(); epoch_pos += pos_loss.item(); steps += 1

        if epoch % 5 == 0 or epoch <= 3:
            print(f'Ep {epoch}/{epochs} | Dyn: {epoch_dyn/steps:.6f} | Pos: {epoch_pos/steps:.6f}', flush=True)

    torch.save(encoder.state_dict(), f'{CKPT}/encoder.pt')
    torch.save(dynamics.state_dict(), f'{CKPT}/dynamics.pt')
    torch.save(target_encoder.state_dict(), f'{CKPT}/target_encoder.pt')
    torch.save(position_probe.state_dict(), f'{CKPT}/position_probe.pt')
    print('World model saved!', flush=True)
    return encoder, dynamics, position_probe


# ---- Step 3: Train Reward Head ----

def train_reward_head(data, encoder, device, epochs=50):
    print(f'\n=== Training Reward Head ({epochs} epochs) ===', flush=True)
    encoder.eval()

    reward_head = RewardHead().to(device)
    optimizer = AdamW(reward_head.parameters(), lr=3e-4)

    frames = data['frames']  # keep on CPU
    rewards = data['rewards'].to(device)
    n = len(frames)

    print('Pre-encoding...', flush=True)
    all_z = []
    with torch.no_grad():
        for s in range(0, n, 512):
            all_z.append(encoder(frames[s:s+512].to(device)))
    all_z = torch.cat(all_z)  # stays on GPU

    for epoch in range(1, epochs + 1):
        reward_head.train()
        perm = torch.randperm(n, device=device)
        loss_sum = 0; steps = 0
        for s in range(0, n, 256):
            idx = perm[s:s+256]
            pred = reward_head(all_z[idx])
            loss = F.mse_loss(pred, rewards[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item(); steps += 1
        if epoch % 10 == 0 or epoch <= 3:
            print(f'Ep {epoch}/{epochs} | Loss: {loss_sum/steps:.6f}', flush=True)

    torch.save(reward_head.state_dict(), f'{CKPT}/reward_head.pt')
    print('Reward head saved!', flush=True)
    return reward_head, all_z


# ---- Step 4: Train Car Probe ----

def train_car_probe(all_z, data, device, epochs=500):
    print(f'\n=== Training Car Probe ({epochs} epochs) ===', flush=True)

    occupancy = data['occupancy'].to(device)
    n = len(all_z)
    perm = torch.randperm(n); train_n = int(n * 0.9)
    tz = all_z[perm[:train_n]]; vz = all_z[perm[train_n:]]
    to = occupancy[perm[:train_n]]; vo = occupancy[perm[train_n:]]

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
            print(f'Ep {epoch:>3d} | T:{ls/st:.4f} V:{vl:.4f} Acc:{acc:.3f} Rec:{rec:.3f} Emp:{ea:.3f}', flush=True)
            if vl < best_val:
                best_val = vl
                best_state = copy.deepcopy(probe.state_dict())

    if best_state:
        probe.load_state_dict(best_state)
    torch.save(probe.state_dict(), f'{CKPT}/car_probe.pt')
    print('Car probe saved!', flush=True)
    return probe


# ---- Main ----

if __name__ == '__main__':
    device = torch.device('cuda')
    print(f'Device: {torch.cuda.get_device_name(0)}', flush=True)

    data = generate_data(num_episodes=100, steps_per_ep=400)
    encoder, dynamics, pos_probe = train_world_model(data, device, epochs=50)
    rh, all_z = train_reward_head(data, encoder, device, epochs=50)
    car_probe = train_car_probe(all_z, data, device, epochs=500)

    # Quick dynamics test
    encoder.eval(); dynamics.eval(); pos_probe.eval()
    with torch.no_grad():
        z = all_z[0:1]
        for steps in [1, 5, 10]:
            results = {}
            for a, name in [(0,'NOOP'),(1,'UP'),(2,'DOWN'),(3,'LEFT'),(4,'RIGHT')]:
                z_sim = z.clone()
                for _ in range(steps):
                    z_sim = dynamics(z_sim, torch.tensor([a], device=device))
                p = pos_probe(z_sim)[0]
                results[name] = (p[0].item(), p[1].item())
            print(f'Step {steps}: ' + ' | '.join(f'{k}=({v[0]:.3f},{v[1]:.3f})' for k,v in results.items()), flush=True)

    print('\n=== All training complete! ===', flush=True)
