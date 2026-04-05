"""Train the policy network via behavioral cloning.

The encoder is frozen. The policy learns to imitate recorded human actions.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.policy import PolicyNetwork
from game_agent.training.dataset import TransitionDataset
from game_agent.utils.logger import setup_logger

logger = setup_logger("train_policy")


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

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    logger.info(f"Dataset: {len(dataset)} transitions ({train_size} train, {val_size} val)")

    # Policy
    policy = PolicyNetwork(config).to(device)
    optimizer = AdamW(policy.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.num_epochs + 1):
        policy.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0
        train_steps = 0

        for obs, action, _next_obs, _reward, _done in train_loader:
            obs = obs.to(device)
            action = action.to(device)

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

        # Validation
        policy.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_steps = 0

        with torch.no_grad():
            for obs, action, _next_obs, _reward, _done in val_loader:
                obs = obs.to(device)
                action = action.to(device)
                z = encoder(obs)
                logits = policy(z)
                loss = F.cross_entropy(logits, action)
                val_loss_sum += loss.item()
                val_steps += 1
                val_correct += (logits.argmax(dim=-1) == action).sum().item()
                val_total += action.size(0)

        train_loss = train_loss_sum / max(train_steps, 1)
        val_loss = val_loss_sum / max(val_steps, 1)
        train_acc = correct / max(total, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}"
        )

        if epoch % 5 == 0:
            torch.save(policy.state_dict(), os.path.join(config.checkpoint_dir, "policy.pt"))
            logger.info("Checkpoint saved.")

    torch.save(policy.state_dict(), os.path.join(config.checkpoint_dir, "policy.pt"))
    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train policy (behavioral cloning)")
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
