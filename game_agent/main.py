"""V-JEPA Game Agent - Main entry point.

Usage:
    python -m game_agent.main --mode play    # Agent plays the game
    python -m game_agent.main --mode record  # Record human gameplay
    python -m game_agent.main --mode train   # Train all models
"""

import argparse
import os
import time

import torch

from game_agent.actions import Action
from game_agent.capture.frame_grabber import FrameGrabber
from game_agent.capture.window_utils import find_window_rect, set_foreground
from game_agent.config import AgentConfig
from game_agent.control.controller import GameController
from game_agent.models.dynamics import DynamicsPredictor
from game_agent.models.policy import PolicyNetwork
from game_agent.models.encoder import Encoder
from game_agent.models.reward_head import RewardHead
from game_agent.planning.shooting import ShootingPlanner
from game_agent.preprocessing.transforms import Preprocessor
from game_agent.utils.logger import setup_logger
from game_agent.utils.video_recorder import VideoRecorder

logger = setup_logger("agent")


def load_model(model: torch.nn.Module, path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def play(config: AgentConfig):
    """Run the agent to play the game autonomously."""
    device = config.resolve_device()
    logger.info(f"Device: {device}")

    # Capture
    rect = find_window_rect(config.window_title)
    grabber = FrameGrabber(region=rect)
    preprocessor = Preprocessor(config)
    logger.info(f"Capturing window '{config.window_title}' at {rect}")

    # Load models
    encoder = load_model(Encoder(config).to(device),
                         os.path.join(config.checkpoint_dir, "encoder.pt"), device)
    dynamics = load_model(DynamicsPredictor(config).to(device),
                          os.path.join(config.checkpoint_dir, "dynamics.pt"), device)

    from game_agent.planning.shooting import PositionProbe
    position_probe = load_model(PositionProbe(config).to(device),
                                os.path.join(config.checkpoint_dir, "position_probe.pt"), device)
    logger.info("Models loaded (encoder + dynamics + position probe).")

    # Planner: shooting with position-based scoring
    planner = ShootingPlanner(encoder, dynamics, position_probe, config, num_samples=512)

    # Controller
    controller = GameController()

    # Video recorder
    video = VideoRecorder(fps=config.capture_fps)

    # Bring game to foreground
    set_foreground(config.window_title)
    time.sleep(0.5)

    dt = 1.0 / config.capture_fps
    step_count = 0

    logger.info(f"Playing at {config.capture_fps} FPS with policy network. Ctrl+C to stop.")

    try:
        while True:
            t0 = time.time()

            frame = grabber.grab()
            obs = preprocessor(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                z = encoder(obs)

                # Use position probe to read ball and paddle positions
                pos = position_probe(z)
                ball_y = pos[0, 1].item()
                paddle_y = pos[0, 2].item()

                # Move paddle toward ball
                # ball_y < paddle_y means ball is higher on screen (lower y value)
                # UP action decreases paddle.y (moves paddle up on screen)
                diff = ball_y - paddle_y
                if diff < -0.02:
                    action = Action.UP    # ball is above, move up
                elif diff > 0.02:
                    action = Action.DOWN  # ball is below, move down
                else:
                    action = Action.NOOP

                if step_count < 20:
                    logger.info(f"ball_y={ball_y:.3f} paddle_y={paddle_y:.3f} diff={diff:.3f} -> {action.name}")

            controller.hold(action)
            video.add_frame(frame)

            step_count += 1
            if step_count % 100 == 0:
                logger.info(f"Step {step_count} | Last action: {action.name}")

            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        controller.release_all()
        if len(video) > 0:
            video_path = "gameplay.mp4"
            video.save(video_path)
            logger.info(f"Saved {len(video)} frames to {video_path}")
        logger.info(f"Stopped after {step_count} steps.")


def train_all(config: AgentConfig):
    """Train all models in sequence."""
    from game_agent.training.train_world_model import train as train_wm
    from game_agent.training.train_reward import train as train_rw
    from game_agent.training.train_policy import train as train_pol

    logger.info("=== Training world model (encoder + dynamics) ===")
    train_wm(config)

    logger.info("=== Training reward head ===")
    train_rw(config)

    logger.info("=== Training policy (behavioral cloning) ===")
    train_pol(config)

    logger.info("=== All training complete ===")


def main():
    parser = argparse.ArgumentParser(description="V-JEPA Game Agent")
    parser.add_argument("--mode", choices=["play", "record", "train"], required=True,
                        help="play: agent plays | record: record human gameplay | train: train models")
    parser.add_argument("--window", type=str, default=None, help="Game window title")
    parser.add_argument("--fps", type=int, default=None, help="Capture FPS")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--horizon", type=int, default=None, help="Planning horizon")
    parser.add_argument("--device", type=str, default=None, help="Device (auto/cuda/cpu)")
    args = parser.parse_args()

    config = AgentConfig()
    if args.window:
        config.window_title = args.window
    if args.fps:
        config.capture_fps = args.fps
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.horizon:
        config.planning_horizon = args.horizon
    if args.device:
        config.device = args.device

    if args.mode == "play":
        play(config)
    elif args.mode == "record":
        from game_agent.record import record
        record(config)
    elif args.mode == "train":
        train_all(config)


if __name__ == "__main__":
    main()
