"""CLI entry point for the Retro Pong environment."""

import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config
from env.pong_env import PongEnv
from env.state import NOOP
from data_logging.dataset_writer import DatasetWriter
from data_logging.replay_loader import ReplayLoader
from policies.base import Policy
from policies.human_policy import HumanPolicy
from policies.bot_policy import (
    BotPolicy, PerfectTrackingPolicy, DelayedTrackingPolicy,
    OscillatoryPolicy, WeakDefensePolicy,
)
from policies.random_policy import RandomPolicy, StickyRandomPolicy, ConstrainedRandomPolicy
from utils.seeding import make_rng


def make_policy(name: str, rng: np.random.RandomState, difficulty: str = "medium") -> Policy:
    """Create a policy by name."""
    policies = {
        "human": lambda: HumanPolicy(),
        "bot": lambda: BotPolicy(difficulty=difficulty, rng=rng),
        "random": lambda: RandomPolicy(rng=rng),
        "sticky_random": lambda: StickyRandomPolicy(rng=rng),
        "constrained_random": lambda: ConstrainedRandomPolicy(rng=rng),
        "perfect": lambda: PerfectTrackingPolicy(),
        "delayed": lambda: DelayedTrackingPolicy(),
        "oscillatory": lambda: OscillatoryPolicy(),
        "weak": lambda: WeakDefensePolicy(rng=rng),
    }
    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Options: {list(policies.keys())}")
    return policies[name]()


def run_play(config: Config):
    """Interactive play mode."""
    import pygame

    env = PongEnv(config)
    clock = pygame.time.Clock()

    policy_left = make_policy(config.policy_left, make_rng(0), config.bot_difficulty)
    policy_right = make_policy(config.policy_right, make_rng(1), config.bot_difficulty)

    obs = env.reset(seed=config.seed)
    running = True

    while running and not env.state.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action_left = policy_left.get_action(env.state, "left")
        action_right = policy_right.get_action(env.state, "right")
        result = env.step(action_left, action_right)
        clock.tick(config.fps)

    s = env.state
    print(f"Game over! Score: {s.score_left} - {s.score_right} ({s.step_count} steps)")
    env.close()


def run_episode(env: PongEnv, policy_left: Policy, policy_right: Policy,
                config: Config, seed: int,
                writer: Optional[DatasetWriter] = None,
                render: bool = False) -> dict:
    """Run a single episode, optionally logging data."""
    import pygame

    policy_left.reset()
    policy_right.reset()
    obs = env.reset(seed=seed)

    if writer:
        writer.begin()
        # Log initial frame/state
        writer.log_step(
            t=0,
            action_left=NOOP,
            action_right=NOOP,
            frame=obs.frame if config.save_frames else None,
            state_dict=env.get_flat_state() if config.save_states else None,
        )

    while not env.state.done:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    sys.exit(0)

        action_left = policy_left.get_action(env.state, "left")
        action_right = policy_right.get_action(env.state, "right")
        result = env.step(action_left, action_right)

        if writer:
            writer.log_step(
                t=env.state.step_count,
                action_left=action_left,
                action_right=action_right,
                frame=result.observation.frame if config.save_frames else None,
                state_dict=env.get_flat_state() if config.save_states else None,
            )

        if render:
            import pygame
            pygame.time.Clock().tick(config.fps)

    s = env.state
    return {
        "score_left": s.score_left,
        "score_right": s.score_right,
        "steps": s.step_count,
    }


def run_generate(config: Config, episodes: int, out_dir: str,
                 split: str = "train"):
    """Generate multiple episodes automatically."""
    config.save_frames = True
    config.save_actions = True
    config.save_states = True

    base_dir = Path(out_dir) / f"split_{split}"
    base_seed = config.seed if config.seed is not None else 42
    rng = make_rng(base_seed)

    env = PongEnv(config)

    interrupted = False

    def handle_signal(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nInterrupted — finishing current episode...")

    signal.signal(signal.SIGINT, handle_signal)

    for i in range(episodes):
        if interrupted:
            break

        ep_seed = int(rng.randint(0, 2**31 - 1))
        ep_id = f"ep_{i:06d}"
        ep_dir = base_dir / ep_id

        policy_left = make_policy(config.policy_left, make_rng(ep_seed), config.bot_difficulty)
        policy_right = make_policy(config.policy_right, make_rng(ep_seed + 1), config.bot_difficulty)

        writer = DatasetWriter(ep_dir, config)

        info = run_episode(env, policy_left, policy_right, config, seed=ep_seed, writer=writer)

        writer.write_metadata(
            episode_id=ep_id,
            seed=ep_seed,
            mode="generate",
            policy_left=config.policy_left,
            policy_right=config.policy_right,
            total_steps=info["steps"],
            score_left=info["score_left"],
            score_right=info["score_right"],
        )
        writer.close()

        print(f"[{i+1}/{episodes}] {ep_id}: {info['score_left']}-{info['score_right']} "
              f"({info['steps']} steps, seed={ep_seed})")

    env.close()
    print(f"Done. Episodes saved to {base_dir}")


def run_replay(episode_dir: str, config: Config):
    """Replay a saved episode."""
    import pygame

    loader = ReplayLoader(Path(episode_dir))
    replay_config = loader.load_config()
    replay_config.headless = False
    seed = loader.load_seed()
    actions = loader.load_actions()

    if not actions:
        print("No actions found to replay.")
        return

    env = PongEnv(replay_config)
    obs = env.reset(seed=seed)
    clock = pygame.time.Clock()

    saved_states = loader.load_states()

    mismatches = 0
    for t, action_left, action_right in actions:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                return

        result = env.step(action_left, action_right)

        # Determinism check
        if saved_states and t < len(saved_states):
            expected = saved_states[t]
            actual = env.get_flat_state()
            if (abs(actual.get("ball_x", 0) - expected.get("ball_x", 0)) > 0.01 or
                abs(actual.get("ball_y", 0) - expected.get("ball_y", 0)) > 0.01):
                mismatches += 1

        clock.tick(replay_config.fps)

    s = env.state
    print(f"Replay done! Score: {s.score_left} - {s.score_right} ({s.step_count} steps)")
    if saved_states:
        print(f"Determinism check: {mismatches} mismatches out of {len(actions)} steps")
    env.close()


def run_record_human(config: Config, out_dir: str):
    """Human play with recording."""
    import pygame

    config.policy_left = "human"
    config.save_frames = True
    config.save_actions = True
    config.save_states = True

    seed = config.seed if config.seed is not None else 42
    ep_dir = Path(out_dir)

    env = PongEnv(config)
    policy_left = HumanPolicy()
    policy_right = make_policy(config.policy_right, make_rng(seed + 1), config.bot_difficulty)

    writer = DatasetWriter(ep_dir, config)
    info = run_episode(env, policy_left, policy_right, config, seed=seed,
                       writer=writer, render=True)

    writer.write_metadata(
        episode_id=ep_dir.name,
        seed=seed,
        mode="record_human",
        policy_left="human",
        policy_right=config.policy_right,
        total_steps=info["steps"],
        score_left=info["score_left"],
        score_right=info["score_right"],
    )
    writer.close()
    env.close()
    print(f"Recorded to {ep_dir}. Score: {info['score_left']}-{info['score_right']}")


def main():
    parser = argparse.ArgumentParser(description="Retro Pong Environment")
    parser.add_argument("--mode", required=True,
                        choices=["play", "bot_vs_bot", "record_human", "generate", "replay"],
                        help="Mode of operation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to generate")
    parser.add_argument("--out", type=str, default="dataset", help="Output directory")
    parser.add_argument("--episode", type=str, help="Episode dir for replay")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument("--target-score", type=int, default=5, help="Score to win")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--policy-left", type=str, default=None, help="Left policy")
    parser.add_argument("--policy-right", type=str, default=None, help="Right policy")
    parser.add_argument("--bot-difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--resolution", type=int, default=256, help="Render resolution (square)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.load(Path(args.config))
    else:
        config = Config()

    # Override from CLI
    if args.seed is not None:
        config.seed = args.seed
    config.fps = args.fps
    config.target_score = args.target_score
    config.max_steps = args.max_steps
    config.bot_difficulty = args.bot_difficulty
    config.render_width = args.resolution
    config.render_height = args.resolution
    if args.headless:
        config.headless = True

    # Mode-specific defaults
    if args.mode == "play":
        config.policy_left = args.policy_left or "human"
        config.policy_right = args.policy_right or "bot"
        run_play(config)

    elif args.mode == "bot_vs_bot":
        config.policy_left = args.policy_left or "bot"
        config.policy_right = args.policy_right or "bot"
        run_play(config)

    elif args.mode == "record_human":
        config.policy_right = args.policy_right or "bot"
        run_record_human(config, args.out)

    elif args.mode == "generate":
        config.headless = True
        config.policy_left = args.policy_left or "bot"
        config.policy_right = args.policy_right or "bot"
        run_generate(config, args.episodes, args.out, args.split)

    elif args.mode == "replay":
        if not args.episode:
            parser.error("--episode is required for replay mode")
        run_replay(args.episode, config)


if __name__ == "__main__":
    main()
