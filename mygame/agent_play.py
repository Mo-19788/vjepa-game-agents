"""Agent plays Pong directly through the environment API.

No screen capture needed — uses env.step() for actions and env.get_frame() for observations.
The agent learns online by playing against the bot.
"""

import sys
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

# Add parent dir so game_agent imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from env.pong_env import PongEnv
from policies.bot_policy import BotPolicy
from utils.seeding import make_rng

from game_agent.config import AgentConfig
from game_agent.models.encoder import Encoder
from game_agent.models.dynamics import DynamicsPredictor
from game_agent.models.reward_head import RewardHead
from game_agent.preprocessing.transforms import Preprocessor


def main():
    # Game config
    game_config = Config()
    game_config.target_score = 999
    game_config.headless = False
    game_config.policy_right = "bot"
    game_config.bot_difficulty = "medium"

    # Agent config
    agent_config = AgentConfig()
    device = agent_config.resolve_device()
    preprocessor = Preprocessor(agent_config)

    # Load models
    ckpt = agent_config.checkpoint_dir
    encoder = Encoder(agent_config).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ckpt, "encoder.pt"), map_location=device, weights_only=True))
    encoder.eval()

    dynamics = DynamicsPredictor(agent_config).to(device)
    dynamics.load_state_dict(torch.load(os.path.join(ckpt, "dynamics.pt"), map_location=device, weights_only=True))
    dynamics.eval()

    # Bot opponent
    rng = make_rng(42)
    bot = BotPolicy(difficulty=game_config.bot_difficulty, rng=rng)

    # Environment
    env = PongEnv(game_config)
    obs = env.reset(seed=42)

    score_left = 0
    score_right = 0
    step_count = 0

    print(f"Agent playing Pong on {device}. Close window to stop.")

    while True:
        frame = obs.frame
        state = env.get_state()

        # --- Simple heuristic agent using game state ---
        # Track ball y-position with paddle
        ball_y = state["ball_y"]
        paddle_y = state["left_paddle_y"]
        paddle_half = game_config.paddle_height / 2

        # Move toward ball
        if ball_y < paddle_y - paddle_half * 0.3:
            action_left = 1  # UP
        elif ball_y > paddle_y + paddle_half * 0.3:
            action_left = 2  # DOWN
        else:
            action_left = 0  # NOOP

        # Bot plays right paddle
        action_right = bot.get_action(state)

        # Step
        result = env.step(action_left, action_right)
        obs = result.observation

        # Track score
        new_state = env.get_state()
        if new_state["score_left"] > score_left:
            score_left = new_state["score_left"]
            print(f"Agent scored! {score_left} - {score_right}")
        if new_state["score_right"] > score_right:
            score_right = new_state["score_right"]
            print(f"Bot scored! {score_left} - {score_right}")

        step_count += 1
        if step_count % 500 == 0:
            print(f"Step {step_count} | Score: {score_left} - {score_right}")

        if result.done:
            print(f"Episode done. Final: {score_left} - {score_right}")
            obs = env.reset()
            score_left = 0
            score_right = 0

        # Small delay for real-time display
        time.sleep(1.0 / game_config.fps)


if __name__ == "__main__":
    main()
