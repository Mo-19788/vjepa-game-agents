"""Train the reward head on recorded transitions with reward labels.

The encoder is frozen (loaded from checkpoint). Only the reward head is trained.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.reward_head import RewardHead
from game_agent.training.dataset import TransitionDataset
from game_agent.utils.logger import setup_logger

logger = setup_logger("train_reward")


def train(config: AgentConfig):
    device = config.resolve_device()
    logger.info(f"Device: {device}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Load frozen encoder
    encoder = Encoder(config).to(device)
    encoder_path = os.path.join(config.checkpoint_dir, "encoder.pt")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    logger.info("Encoder loaded and frozen.")

    # Data
    dataset = TransitionDataset(config)
    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Build oversampling weights: upweight non-zero rewards
    weights = []
    for idx in train_set.indices:
        _, _, _, reward, _ = dataset[idx]
        w = 10.0 if reward.abs().item() > 0 else 1.0
        weights.append(w)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    logger.info(f"Dataset: {len(dataset)} transitions ({train_size} train, {val_size} val)")

    # Reward head
    reward_head = RewardHead(config).to(device)
    optimizer = AdamW(reward_head.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.num_epochs + 1):
        reward_head.train()
        train_loss_sum = 0.0
        train_steps = 0

        for obs, _action, _next_obs, reward, _done in train_loader:
            obs = obs.to(device)
            reward = reward.to(device)

            with torch.no_grad():
                z = encoder(obs)

            pred = reward_head(z)
            loss = F.mse_loss(pred, reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        # Validation
        reward_head.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for obs, _action, _next_obs, reward, _done in val_loader:
                obs = obs.to(device)
                reward = reward.to(device)
                z = encoder(obs)
                pred = reward_head(z)
                loss = F.mse_loss(pred, reward)
                val_loss_sum += loss.item()
                val_steps += 1

        train_loss = train_loss_sum / max(train_steps, 1)
        val_loss = val_loss_sum / max(val_steps, 1)
        logger.info(f"Epoch {epoch}/{config.num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if epoch % 5 == 0:
            torch.save(reward_head.state_dict(), os.path.join(config.checkpoint_dir, "reward_head.pt"))
            logger.info("Checkpoint saved.")

    torch.save(reward_head.state_dict(), os.path.join(config.checkpoint_dir, "reward_head.pt"))
    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train reward head")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    config = AgentConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.data_dir:
        config.data_dir = args.data_dir

    train(config)


if __name__ == "__main__":
    main()
