"""Generate diverse training data for V-JEPA.

Runs multiple bot strategies to get varied player behavior:
- Cautious bot: waits for gaps, mostly goes up
- Aggressive bot: rushes forward, gets hit often
- Random: explores all positions and actions
- Dodger: moves left/right a lot to avoid cars
"""

import subprocess
import sys
import json
import os
import numpy as np
from pathlib import Path

import cv2

from config import Config
from env.crosser_env import CrosserEnv
from env.state import NOOP, UP, DOWN, LEFT, RIGHT


def cautious_bot(state, config, rng):
    """Waits for clear path, then moves up."""
    player = state.player
    target_row = player.row - 1

    if target_row < config.safe_rows:
        return UP  # safe zone, just go

    # Check if target row is clear
    for car in state.cars:
        if car.row == target_row:
            car_left = car.x
            car_right = car.x + car.width
            if car_left - 2.0 <= player.col <= car_right + 1.0:
                # Car nearby — wait or dodge
                return rng.choice([NOOP, NOOP, NOOP, LEFT, RIGHT])

    return UP


def aggressive_bot(state, config, rng):
    """Rushes up, doesn't dodge much."""
    if rng.random() < 0.7:
        return UP
    return rng.choice([NOOP, LEFT, RIGHT])


def random_bot(state, config, rng):
    """Pure random actions for exploration."""
    return rng.choice([NOOP, UP, DOWN, LEFT, RIGHT])


def dodger_bot(state, config, rng):
    """Moves laterally a lot, goes up when safe."""
    player = state.player
    target_row = player.row - 1

    # Check danger in current row
    in_danger = False
    for car in state.cars:
        if car.row == player.row:
            dist = abs(car.x + car.width / 2 - player.col)
            if dist < 3:
                in_danger = True
                # Dodge away from car
                if car.x > player.col:
                    return LEFT
                else:
                    return RIGHT

    # Check if safe to go up
    safe_ahead = True
    if target_row >= config.safe_rows:
        for car in state.cars:
            if car.row == target_row:
                if abs(car.x + car.width / 2 - player.col) < 2.5:
                    safe_ahead = False

    if safe_ahead and rng.random() < 0.4:
        return UP
    return rng.choice([NOOP, LEFT, RIGHT, LEFT, RIGHT])  # heavy lateral movement


BOTS = {
    "cautious": cautious_bot,
    "aggressive": aggressive_bot,
    "random": random_bot,
    "dodger": dodger_bot,
}


def generate_episodes(config, bot_name, bot_fn, num_episodes, out_dir):
    """Generate episodes with a specific bot."""
    split_dir = Path(out_dir) / "split_train"
    split_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(split_dir.glob("ep_*")))

    config.headless = True

    print(f"\n{'='*50}")
    print(f"Generating {num_episodes} episodes with {bot_name} bot")
    print(f"{'='*50}")

    for ep_idx in range(num_episodes):
        ep_id = f"ep_{existing + ep_idx:06d}"
        ep_dir = split_dir / ep_id
        ep_dir.mkdir(exist_ok=True)

        seed = np.random.randint(0, 999999)
        rng = np.random.RandomState(seed)

        env = CrosserEnv(config)
        obs = env.reset(seed=seed)
        state = env._state

        actions_file = open(ep_dir / "actions.jsonl", "w")
        states_file = open(ep_dir / "states.jsonl", "w")

        frame_idx = 0
        cv2.imwrite(str(ep_dir / f"frame_{frame_idx:06d}.png"),
                     cv2.cvtColor(obs.frame, cv2.COLOR_RGB2BGR))
        states_file.write(json.dumps(state.flat_dict()) + "\n")

        while not state.done:
            action = int(bot_fn(state, config, rng))
            actions_file.write(json.dumps({"t": frame_idx, "action": action}) + "\n")

            result = env.step(action)
            state = env._state
            frame_idx += 1

            cv2.imwrite(str(ep_dir / f"frame_{frame_idx:06d}.png"),
                         cv2.cvtColor(result.observation.frame, cv2.COLOR_RGB2BGR))
            states_file.write(json.dumps(state.flat_dict()) + "\n")

        actions_file.close()
        states_file.close()

        metadata = {
            "episode_id": ep_id,
            "seed": seed,
            "bot": bot_name,
            "total_steps": frame_idx,
            "final_score": state.score,
            "frame_count": frame_idx + 1,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  {ep_id} ({bot_name}): {frame_idx} steps, score={state.score}")


def main():
    config = Config()
    config.max_steps = 500  # shorter episodes for more diversity
    out_dir = "dataset"

    PLAN = [
        ("cautious", 30),
        ("aggressive", 30),
        ("random", 30),
        ("dodger", 30),
    ]

    for bot_name, num_eps in PLAN:
        generate_episodes(config, bot_name, BOTS[bot_name], num_eps, out_dir)

    # Count total
    total = len(list(Path(out_dir, "split_train").glob("ep_*")))
    print(f"\nDone! Total episodes: {total}")


if __name__ == "__main__":
    main()
