"""V-JEPA agent plays Pong through the environment API.

Uses the trained encoder + position probe to read ball/paddle positions
from rendered frames, then moves the paddle toward the ball.
"""

import sys
import os
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from env.pong_env import PongEnv
from policies.bot_policy import BotPolicy
from utils.seeding import make_rng

from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.planning.shooting import PositionProbe
from game_agent.preprocessing.transforms import Preprocessor


def main():
    # Game config
    game_config = Config()
    game_config.target_score = 999
    game_config.headless = False
    game_config.policy_right = "bot"
    game_config.bot_difficulty = "easy"

    # Agent config
    agent_config = AgentConfig()
    device = agent_config.resolve_device()
    preprocessor = Preprocessor(agent_config)

    # Load V-JEPA models
    ckpt = os.path.join(os.path.dirname(__file__), "..", agent_config.checkpoint_dir)
    encoder = Encoder(agent_config).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ckpt, "encoder.pt"),
                                        map_location=device, weights_only=True))
    encoder.eval()

    probe = PositionProbe(agent_config).to(device)
    probe.load_state_dict(torch.load(os.path.join(ckpt, "position_probe.pt"),
                                      map_location=device, weights_only=True))
    probe.eval()

    print(f"V-JEPA agent on {device}")
    print("Encoder + position probe loaded")

    # Bot opponent
    rng = make_rng(42)
    bot = BotPolicy(difficulty=game_config.bot_difficulty, rng=rng)

    # Environment
    env = PongEnv(game_config)
    obs = env.reset(seed=42)

    score_left = 0
    score_right = 0
    step_count = 0
    action_counts = {0: 0, 1: 0, 2: 0}

    action_left = 0
    think_counter = 0

    # Pre-compute encoder once to warm up
    dummy = preprocessor(np.zeros((256, 256, 3), dtype=np.uint8)).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder(dummy)
    print("Encoder warmed up.")

    print("Playing! Close window or Ctrl+C to stop.\n")

    try:
        while True:
            frame = obs.frame

            # Run V-JEPA every 10 game steps
            think_counter += 1
            if think_counter >= 6:
                think_counter = 0
                obs_tensor = preprocessor(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    z = encoder(obs_tensor)
                    pos = probe(z)[0]

                ball_y = pos[1].item()
                paddle_y = pos[2].item()

                # Pure V-JEPA: positions from pixels only
                diff = ball_y - paddle_y
                if diff > 0.01:
                    action_left = 2  # DOWN
                elif diff < -0.01:
                    action_left = 1  # UP
                else:
                    action_left = 0  # NOOP

            # No sleep - let the game loop run as fast as it can
            # The env.step + render handles timing

            action_counts[action_left] += 1

            # Bot plays right paddle using internal state object
            action_right = bot.get_action(env._state, "right")

            # Step
            result = env.step(action_left, action_right)
            obs = result.observation

            # Track score
            new_state = env.get_state()
            if new_state["score_left"] > score_left:
                score_left = new_state["score_left"]
                print(f"  Agent scored! {score_left} - {score_right}")
            if new_state["score_right"] > score_right:
                score_right = new_state["score_right"]
                print(f"  Bot scored!   {score_left} - {score_right}")

            step_count += 1
            if step_count % 500 == 0:
                print(f"  Step {step_count} | Score: {score_left}-{score_right} | "
                      f"Actions: NOOP={action_counts[0]} UP={action_counts[1]} DOWN={action_counts[2]}")

            if result.done:
                print(f"\nEpisode done. Final: {score_left} - {score_right}")
                print(f"Actions: NOOP={action_counts[0]} UP={action_counts[1]} DOWN={action_counts[2]}")
                obs = env.reset()
                score_left = 0
                score_right = 0
                action_counts = {0: 0, 1: 0, 2: 0}

            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            time.sleep(1.0 / game_config.fps)

    except KeyboardInterrupt:
        print(f"\nStopped. Score: {score_left} - {score_right}")
        print(f"Actions: NOOP={action_counts[0]} UP={action_counts[1]} DOWN={action_counts[2]}")


if __name__ == "__main__":
    main()
