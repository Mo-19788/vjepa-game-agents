"""
Kaggle GPU training script for V-JEPA Game Agent.
Trains encoder + dynamics, reward head, and policy.
Downloads results as checkpoints.
"""

import copy
import glob
import os
import zipfile
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import cv2


# ============================================================
# Config
# ============================================================

@dataclass
class AgentConfig:
    frame_size: Tuple[int, int] = (224, 224)
    grayscale: bool = False
    frame_stack: int = 1
    latent_dim: int = 256
    num_actions: int = 3
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    train_split: float = 0.9
    ema_tau: float = 0.996
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] < 7:
                    print(f"GPU compute capability {cap[0]}.{cap[1]} too low, using CPU")
                    return torch.device("cpu")
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)

    @property
    def input_channels(self) -> int:
        c = 1 if self.grayscale else 3
        return c * self.frame_stack


class Action(IntEnum):
    NOOP = 0
    UP = 1
    DOWN = 2


# ============================================================
# Preprocessing
# ============================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Preprocessor:
    def __init__(self, config: AgentConfig):
        self.size = config.frame_size
        self.grayscale = config.grayscale

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[:, :, np.newaxis]
        img = img.astype(np.float32) / 255.0
        if not self.grayscale:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        return tensor


# ============================================================
# Dataset
# ============================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def batch_resize_uint8(frames: np.ndarray, size=(224, 224)) -> torch.Tensor:
    """Resize frames and store as uint8: (N,H,W,3) -> (N,3,h,w) uint8."""
    # Permute to (N,3,H,W) as float for resize, then back to uint8
    t = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()
    if (t.shape[2], t.shape[3]) != size:
        t = torch.nn.functional.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return t.clamp(0, 255).byte()  # uint8, 8x less memory than float32


class TransitionDataset(Dataset):
    """Stores frames as uint8 in RAM, with positions for auxiliary loss."""

    def __init__(self, data_dir: str, config: AgentConfig):
        chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk_*.npz files found in {data_dir}")

        obs_list = []
        next_obs_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        positions_list = []

        import time
        step = 2  # Use every 2nd transition
        for i, cf in enumerate(chunk_files):
            t0 = time.time()
            print(f"Loading chunk {i+1}/{len(chunk_files)}...", end=" ", flush=True)
            d = np.load(cf)
            actions_list.append(d["action"][::step])
            rewards_list.append(d["reward"][::step])
            dones_list.append(d["done"][::step])

            # Positions: (ball_x, ball_y, left_paddle_y, right_paddle_y) x2 for current+next
            if "positions" in d:
                positions_list.append(d["positions"][::step])

            for start in range(0, len(d["obs"][::step]), 500):
                obs_list.append(batch_resize_uint8(d["obs"][::step][start:start+500], config.frame_size))
                next_obs_list.append(batch_resize_uint8(d["next_obs"][::step][start:start+500], config.frame_size))

            dt = time.time() - t0
            print(f"done ({dt:.1f}s)")
            del d
            import gc; gc.collect()

        self.obs = torch.cat(obs_list)
        self.next_obs = torch.cat(next_obs_list)
        del obs_list, next_obs_list
        import gc; gc.collect()

        self.actions = torch.tensor(np.concatenate(actions_list), dtype=torch.long)
        self.rewards = torch.tensor(np.concatenate(rewards_list), dtype=torch.float32)
        self.dones = torch.tensor(np.concatenate(dones_list), dtype=torch.bool)

        if positions_list:
            self.positions = torch.tensor(np.concatenate(positions_list), dtype=torch.float32)
            self.has_positions = True
        else:
            self.positions = torch.zeros(len(self.obs), 8)
            self.has_positions = False

        print(f"Dataset ready: {len(self.obs)} transitions, "
              f"obs shape: {self.obs.shape}, has_positions: {self.has_positions}, "
              f"RAM: ~{(self.obs.nelement() + self.next_obs.nelement()) / 1e9:.1f} GB")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = self.obs[idx].float().div_(255.0)
        obs = (obs - IMAGENET_MEAN) / IMAGENET_STD
        next_obs = self.next_obs[idx].float().div_(255.0)
        next_obs = (next_obs - IMAGENET_MEAN) / IMAGENET_STD
        return obs, self.actions[idx], next_obs, self.rewards[idx], self.dones[idx], self.positions[idx]


# ============================================================
# Models
# ============================================================

class Encoder(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        in_channels = config.input_channels
        ch = config.encoder_channels
        latent_dim = config.latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[1], ch[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[2], ch[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch[3], latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        h = self.convs(x)
        h = self.pool(h).flatten(1)
        h = self.fc(h)
        h = self.norm(h)
        return h


class DynamicsPredictor(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.num_actions = config.num_actions
        input_dim = config.latent_dim + config.num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, config.latent_dim),
        )
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(self, z, action):
        action_onehot = F.one_hot(action.long(), num_classes=self.num_actions).float()
        x = torch.cat([z, action_onehot], dim=-1)
        delta = self.net(x)
        return self.norm(z + delta)


class RewardHead(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class PositionProbe(nn.Module):
    """Auxiliary head: predict ball_x, ball_y, left_paddle_y, right_paddle_y from latent."""
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # ball_x, ball_y, left_paddle_y, right_paddle_y
            nn.Sigmoid(),  # positions are normalized 0-1
        )

    def forward(self, z):
        return self.net(z)


class PolicyNetwork(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.num_actions),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# Training functions
# ============================================================

@torch.no_grad()
def update_ema(online, target, tau):
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)


def train_world_model(config, dataset, device, output_dir):
    print("=== Training World Model (with position probe) ===")

    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Dataset: {len(dataset)} transitions ({train_size} train, {val_size} val)")
    print(f"Has positions: {dataset.has_positions}")

    encoder = Encoder(config).to(device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    dynamics = DynamicsPredictor(config).to(device)
    position_probe = PositionProbe(config).to(device)

    # Train encoder + dynamics + position probe jointly
    params = list(encoder.parameters()) + list(dynamics.parameters()) + list(position_probe.parameters())
    optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Dynamics params: {sum(p.numel() for p in dynamics.parameters()):,}")
    print(f"Position probe params: {sum(p.numel() for p in position_probe.parameters()):,}")

    for epoch in range(1, config.num_epochs + 1):
        encoder.train()
        dynamics.train()
        position_probe.train()
        train_loss_sum = 0.0
        train_pos_loss_sum = 0.0
        train_steps = 0

        for obs, action, next_obs, _, _, positions in train_loader:
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)
            positions = positions.to(device)

            z_t = encoder(obs)
            with torch.no_grad():
                z_next_target = target_encoder(next_obs)

            z_next_pred = dynamics(z_t, action)
            dynamics_loss = F.mse_loss(z_next_pred, z_next_target)

            # Auxiliary: predict current positions from current latent
            # positions[:, :4] = current (ball_x, ball_y, left_paddle_y, right_paddle_y)
            pos_pred = position_probe(z_t)
            pos_target = positions[:, :4]
            pos_loss = F.mse_loss(pos_pred, pos_target)

            # Combined loss: dynamics + position probe
            loss = dynamics_loss + 1.0 * pos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(encoder, target_encoder, config.ema_tau)

            train_loss_sum += dynamics_loss.item()
            train_pos_loss_sum += pos_loss.item()
            train_steps += 1

        scheduler.step()

        # Validation
        encoder.eval()
        dynamics.eval()
        position_probe.eval()
        val_loss_sum = 0.0
        val_pos_loss_sum = 0.0
        val_steps = 0
        latent_std = 0.0

        with torch.no_grad():
            for obs, action, next_obs, _, _, positions in val_loader:
                obs = obs.to(device)
                action = action.to(device)
                next_obs = next_obs.to(device)
                positions = positions.to(device)

                z_t = encoder(obs)
                z_next_target = target_encoder(next_obs)
                z_next_pred = dynamics(z_t, action)
                dynamics_loss = F.mse_loss(z_next_pred, z_next_target)

                pos_pred = position_probe(z_t)
                pos_loss = F.mse_loss(pos_pred, positions[:, :4])

                val_loss_sum += dynamics_loss.item()
                val_pos_loss_sum += pos_loss.item()
                val_steps += 1
                latent_std += z_t.std().item()

        train_loss = train_loss_sum / max(train_steps, 1)
        train_pos = train_pos_loss_sum / max(train_steps, 1)
        val_loss = val_loss_sum / max(val_steps, 1)
        val_pos = val_pos_loss_sum / max(val_steps, 1)
        latent_std = latent_std / max(val_steps, 1)

        print(f"Epoch {epoch}/{config.num_epochs} | Dyn: {train_loss:.6f}/{val_loss:.6f} | Pos: {train_pos:.6f}/{val_pos:.6f} | Std: {latent_std:.4f}")

        if latent_std < 0.01:
            print("WARNING: Possible representation collapse!")

    torch.save(encoder.state_dict(), os.path.join(output_dir, "encoder.pt"))
    torch.save(dynamics.state_dict(), os.path.join(output_dir, "dynamics.pt"))
    torch.save(target_encoder.state_dict(), os.path.join(output_dir, "target_encoder.pt"))
    torch.save(position_probe.state_dict(), os.path.join(output_dir, "position_probe.pt"))
    print("World model + position probe saved.")
    return encoder


def train_reward(config, dataset, encoder, device, output_dir, num_epochs=50):
    print(f"\n=== Training Reward Head ({num_epochs} epochs) ===")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    weights = []
    for idx in train_set.indices:
        _, _, _, reward, _ = dataset[idx]
        w = 10.0 if reward.abs().item() > 0 else 1.0
        weights.append(w)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    reward_head = RewardHead(config).to(device)
    optimizer = AdamW(reward_head.parameters(), lr=config.learning_rate)

    for epoch in range(1, num_epochs + 1):
        reward_head.train()
        train_loss_sum = 0.0
        train_steps = 0

        for obs, _, _, reward, _, _ in train_loader:
            obs, reward = obs.to(device), reward.to(device)
            with torch.no_grad():
                z = encoder(obs)
            pred = reward_head(z)
            loss = F.mse_loss(pred, reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_steps += 1

        reward_head.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for obs, _, _, reward, _, _ in val_loader:
                obs, reward = obs.to(device), reward.to(device)
                z = encoder(obs)
                pred = reward_head(z)
                val_loss_sum += F.mse_loss(pred, reward).item()
                val_steps += 1

        train_loss = train_loss_sum / max(train_steps, 1)
        val_loss = val_loss_sum / max(val_steps, 1)
        print(f"Epoch {epoch}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    torch.save(reward_head.state_dict(), os.path.join(output_dir, "reward_head.pt"))
    print("Reward head saved.")


def train_policy(config, dataset, encoder, device, output_dir):
    print("\n=== Training Policy ===")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    policy = PolicyNetwork(config).to(device)
    optimizer = AdamW(policy.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.num_epochs + 1):
        policy.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0
        train_steps = 0

        for obs, action, _, _, _, _ in train_loader:
            obs, action = obs.to(device), action.to(device)
            with torch.no_grad():
                z = encoder(obs)
            logits = policy(z)
            loss = F.cross_entropy(logits, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_steps += 1
            correct += (logits.argmax(dim=-1) == action).sum().item()
            total += action.size(0)

        policy.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_steps = 0
        with torch.no_grad():
            for obs, action, _, _, _, _ in val_loader:
                obs, action = obs.to(device), action.to(device)
                z = encoder(obs)
                logits = policy(z)
                val_loss_sum += F.cross_entropy(logits, action).item()
                val_steps += 1
                val_correct += (logits.argmax(dim=-1) == action).sum().item()
                val_total += action.size(0)

        train_loss = train_loss_sum / max(train_steps, 1)
        val_loss = val_loss_sum / max(val_steps, 1)
        train_acc = correct / max(total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"Epoch {epoch}/{config.num_epochs} | Train: {train_loss:.4f} acc:{train_acc:.3f} | Val: {val_loss:.4f} acc:{val_acc:.3f}")

    torch.save(policy.state_dict(), os.path.join(output_dir, "policy.pt"))
    print("Policy saved.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    config = AgentConfig()
    device = config.resolve_device()
    print(f"Device: {device}")

    # Find training data
    data_dir = "/kaggle/working/data"
    os.makedirs(data_dir, exist_ok=True)

    # Search all possible locations for .npz files
    npz_files = glob.glob("/kaggle/input/**/*.npz", recursive=True)
    zip_files = glob.glob("/kaggle/input/**/*.zip", recursive=True)

    if npz_files:
        data_dir = os.path.dirname(npz_files[0])
        print(f"Found {len(npz_files)} .npz files in {data_dir}")
    elif zip_files:
        print(f"Found zip: {zip_files[0]}, extracting...")
        with zipfile.ZipFile(zip_files[0], "r") as zf:
            zf.extractall(data_dir)
        print(f"Extracted {len(os.listdir(data_dir))} files to {data_dir}")
    else:
        # Download via kaggle API
        print("No data in /kaggle/input. Downloading via API...")
        os.system("kaggle datasets download mollmmes1010/pong-agent-training-data -p /kaggle/working/download")
        dl_zips = glob.glob("/kaggle/working/download/*.zip")
        if dl_zips:
            with zipfile.ZipFile(dl_zips[0], "r") as zf:
                zf.extractall(data_dir)
            print(f"Extracted {len(glob.glob(os.path.join(data_dir, '*.npz')))} .npz files")
        else:
            raise FileNotFoundError("Failed to download dataset")

    output_dir = "/kaggle/working/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    dataset = TransitionDataset(data_dir, config)
    print(f"Loaded {len(dataset)} transitions")

    # Train all models
    encoder = train_world_model(config, dataset, device, output_dir)
    train_reward(config, dataset, encoder, device, output_dir, num_epochs=100)
    train_policy(config, dataset, encoder, device, output_dir)

    # Zip checkpoints for easy download
    zip_path = "/kaggle/working/checkpoints.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in glob.glob(os.path.join(output_dir, "*.pt")):
            zf.write(f, os.path.basename(f))
    print(f"\nAll done! Checkpoints zipped to {zip_path}")
