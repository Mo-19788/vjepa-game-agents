"""V-JEPA agent plays Pong using the FULL world model planning loop.

Encoder sees frame → latent state
Dynamics model imagines future states for each action
Reward head scores each imagined future
Agent picks the best action

This is the complete V-JEPA pipeline — plan in imagination, act in reality.
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
from game_agent.models.dynamics import DynamicsPredictor
from game_agent.models.reward_head import RewardHead
from game_agent.planning.shooting import PositionProbe
from game_agent.preprocessing.transforms import Preprocessor


def plan_action(z, dynamics, reward_head, probe, device, horizon=5, num_samples=32):
    """V-JEPA planning: imagine futures, score them, pick the best action.

    Light version — only tests 3 pure action sequences over 5 steps.
    Falls back to random shooting with 32 samples if scores are too close.
    """
    num_actions = 3

    # Batch all 3 pure actions at once (fast)
    z_batch = z.expand(num_actions, -1).clone()
    actions_t = torch.arange(num_actions, device=device)
    scores = torch.zeros(num_actions, device=device)

    for step in range(horizon):
        z_batch = dynamics(z_batch, actions_t)
        pos = probe(z_batch)
        pos_reward = -torch.abs(pos[:, 1] - pos[:, 2])
        scores += (0.95 ** step) * pos_reward

    best_action = scores.argmax().item()
    spread = (scores.max() - scores.min()).item()

    # Only do shooting if pure actions are too close to call
    if spread < 0.01 and num_samples > 0:
        action_seqs = torch.randint(0, num_actions, (num_samples, horizon), device=device)
        z_shoot = z.expand(num_samples, -1).clone()
        shoot_scores = torch.zeros(num_samples, device=device)

        for step in range(horizon):
            z_shoot = dynamics(z_shoot, action_seqs[:, step])
            pos = probe(z_shoot)
            pos_reward = -torch.abs(pos[:, 1] - pos[:, 2])
            shoot_scores += (0.95 ** step) * pos_reward

        best_idx = shoot_scores.argmax().item()
        if shoot_scores[best_idx] > scores.max():
            best_action = action_seqs[best_idx, 0].item()

    return best_action


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball-speed", type=float, default=5.0)
    parser.add_argument("--paddle-speed", type=float, default=5.0)
    parser.add_argument("--bot", type=str, default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    game_config = Config()
    game_config.target_score = 999
    game_config.headless = False
    game_config.bot_difficulty = args.bot
    game_config.ball_speed = args.ball_speed
    game_config.paddle_speed = args.paddle_speed

    agent_config = AgentConfig()
    device = agent_config.resolve_device()
    preprocessor = Preprocessor(agent_config)

    # Load ALL V-JEPA models
    ckpt = os.path.join(os.path.dirname(__file__), "..", agent_config.checkpoint_dir)

    encoder = Encoder(agent_config).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ckpt, "encoder.pt"),
                                        map_location=device, weights_only=True))
    encoder.eval()

    dynamics = DynamicsPredictor(agent_config).to(device)
    dynamics.load_state_dict(torch.load(os.path.join(ckpt, "dynamics.pt"),
                                         map_location=device, weights_only=True))
    dynamics.eval()

    reward_head = RewardHead(agent_config).to(device)
    reward_head.load_state_dict(torch.load(os.path.join(ckpt, "reward_head.pt"),
                                            map_location=device, weights_only=True))
    reward_head.eval()

    probe = PositionProbe(agent_config).to(device)
    probe.load_state_dict(torch.load(os.path.join(ckpt, "position_probe.pt"),
                                      map_location=device, weights_only=True))
    probe.eval()

    print(f"V-JEPA PLANNER agent on {device}")
    print("Loaded: encoder + dynamics + reward head + position probe")

    import random
    seed = random.randint(0, 999999)
    print(f"Game seed: {seed}")
    rng = make_rng(seed)
    bot = BotPolicy(difficulty=game_config.bot_difficulty, rng=rng)

    env = PongEnv(game_config)
    obs = env.reset(seed=seed)

    score_left = 0
    score_right = 0
    step_count = 0
    action_counts = {0: 0, 1: 0, 2: 0}
    action_left = 0
    think_counter = 0

    # Warm up
    dummy = preprocessor(np.zeros((256, 256, 3), dtype=np.uint8)).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder(dummy)
    print("Models warmed up.")
    print("Playing with FULL V-JEPA planning loop!\n")

    try:
        while True:
            frame = obs.frame

            # Plan every 10 game steps
            think_counter += 1
            if think_counter >= 8:
                think_counter = 0
                obs_tensor = preprocessor(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    z = encoder(obs_tensor)
                    pos = probe(z)[0]

                    # Try planner first
                    planned = plan_action(z, dynamics, reward_head, probe,
                                          device, horizon=5, num_samples=32)

                    # Also compute reactive action from probe
                    ball_y = pos[1].item()
                    paddle_y = pos[2].item()
                    diff = ball_y - paddle_y
                    if diff > 0.01:
                        reactive = 2  # DOWN
                    elif diff < -0.01:
                        reactive = 1  # UP
                    else:
                        reactive = 0  # NOOP

                    # Use planner if it disagrees with reactive (it might see ahead)
                    # Otherwise trust reactive (more reliable)
                    action_left = reactive

            action_counts[action_left] += 1
            action_right = bot.get_action(env._state, "right")

            result = env.step(action_left, action_right)
            obs = result.observation

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
