"""Train the encoder + dynamics predictor jointly.

Uses MSE loss in latent space with an EMA target encoder to prevent
representation collapse.
"""

import argparse
import copy
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.dynamics import DynamicsPredictor
from game_agent.training.dataset import TransitionDataset
from game_agent.utils.logger import setup_logger

logger = setup_logger("train_world_model")


@torch.no_grad()
def update_ema(online: torch.nn.Module, target: torch.nn.Module, tau: float):
    """Exponential moving average update: target = tau*target + (1-tau)*online."""
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)


def train(config: AgentConfig):
    device = config.resolve_device()
    logger.info(f"Device: {device}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Data
    dataset = TransitionDataset(config)
    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    logger.info(f"Dataset: {len(dataset)} transitions ({train_size} train, {val_size} val)")

    # Models
    encoder = Encoder(config).to(device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    dynamics = DynamicsPredictor(config).to(device)

    # Optimizer
    params = list(encoder.parameters()) + list(dynamics.parameters())
    optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    logger.info(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    logger.info(f"Dynamics params: {sum(p.numel() for p in dynamics.parameters()):,}")

    best_val_loss = float("inf")

    for epoch in range(1, config.num_epochs + 1):
        # --- Training ---
        encoder.train()
        dynamics.train()
        train_loss_sum = 0.0
        train_steps = 0

        for obs, action, next_obs, _reward, _done in train_loader:
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)

            z_t = encoder(obs)
            with torch.no_grad():
                z_next_target = target_encoder(next_obs)

            z_next_pred = dynamics(z_t, action)
            loss = F.mse_loss(z_next_pred, z_next_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            update_ema(encoder, target_encoder, config.ema_tau)

            train_loss_sum += loss.item()
            train_steps += 1

        scheduler.step()
        train_loss = train_loss_sum / max(train_steps, 1)

        # --- Validation ---
        encoder.eval()
        dynamics.eval()
        val_loss_sum = 0.0
        val_steps = 0
        latent_std = 0.0

        with torch.no_grad():
            for obs, action, next_obs, _reward, _done in val_loader:
                obs = obs.to(device)
                action = action.to(device)
                next_obs = next_obs.to(device)

                z_t = encoder(obs)
                z_next_target = target_encoder(next_obs)
                z_next_pred = dynamics(z_t, action)
                loss = F.mse_loss(z_next_pred, z_next_target)

                val_loss_sum += loss.item()
                val_steps += 1
                latent_std += z_t.std().item()

        val_loss = val_loss_sum / max(val_steps, 1)
        latent_std = latent_std / max(val_steps, 1)

        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | "
            f"Latent std: {latent_std:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Collapse warning
        if latent_std < 0.01:
            logger.warning("Latent std very low — possible representation collapse!")

        # Checkpoint
        if epoch % 5 == 0 or val_loss < best_val_loss:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            torch.save(encoder.state_dict(), os.path.join(config.checkpoint_dir, "encoder.pt"))
            torch.save(dynamics.state_dict(), os.path.join(config.checkpoint_dir, "dynamics.pt"))
            torch.save(target_encoder.state_dict(), os.path.join(config.checkpoint_dir, "target_encoder.pt"))
            logger.info(f"Checkpoint saved (val_loss={val_loss:.6f})")

    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train world model (encoder + dynamics)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    config = AgentConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.data_dir:
        config.data_dir = args.data_dir

    train(config)


if __name__ == "__main__":
    main()
