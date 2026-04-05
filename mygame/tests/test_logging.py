"""Tests for dataset logging and replay loading."""

import sys
import os
import json
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
from pathlib import Path
from config import Config
from env.pong_env import PongEnv
from env.state import NOOP
from data_logging.dataset_writer import DatasetWriter
from data_logging.replay_loader import ReplayLoader
from policies.bot_policy import BotPolicy
from utils.seeding import make_rng


def test_dataset_writer():
    """Test that writer produces correct file structure."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        config = Config(headless=True, save_frames=True, save_actions=True,
                        save_states=True, target_score=1, max_steps=500)
        ep_dir = tmpdir / "ep_000000"
        writer = DatasetWriter(ep_dir, config)
        writer.begin()

        env = PongEnv(config)
        seed = 42
        rng = make_rng(seed)
        bot = BotPolicy(rng=make_rng(seed + 1))
        env.reset(seed=seed)

        step = 0
        writer.log_step(step, NOOP, NOOP, frame=env.get_frame(),
                        state_dict=env.get_flat_state())

        while not env.state.done and step < 200:
            al = bot.get_action(env.state, "left")
            ar = bot.get_action(env.state, "right")
            env.step(al, ar)
            step += 1
            writer.log_step(step, al, ar, frame=env.get_frame(),
                            state_dict=env.get_flat_state())

        writer.write_metadata("ep_000000", seed, "test", "bot", "bot",
                              step, env.state.score_left, env.state.score_right)
        writer.close()
        env.close()

        # Verify files
        assert (ep_dir / "metadata.json").exists()
        assert (ep_dir / "actions.jsonl").exists()
        assert (ep_dir / "states.jsonl").exists()
        assert (ep_dir / "frame_000000.png").exists()

        # Verify counts match
        meta = json.loads((ep_dir / "metadata.json").read_text())
        actions_count = sum(1 for _ in open(ep_dir / "actions.jsonl"))
        states_count = sum(1 for _ in open(ep_dir / "states.jsonl"))
        frame_count = len(list(ep_dir.glob("frame_*.png")))

        assert frame_count == meta["frame_count"], \
            f"Frame count mismatch: {frame_count} vs {meta['frame_count']}"
        assert actions_count == meta["action_count"]

        print(f"PASS: test_dataset_writer ({frame_count} frames, {actions_count} actions)")
    finally:
        shutil.rmtree(tmpdir)


def test_replay_loader():
    """Test that replay loader reads saved data correctly."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        ep_dir = tmpdir / "ep_test"
        ep_dir.mkdir()

        config = Config(headless=True, target_score=3)
        meta = {
            "episode_id": "ep_test",
            "seed": 42,
            "config": config.to_dict(),
            "mode": "test",
            "policy_left": "bot",
            "policy_right": "bot",
            "total_steps": 100,
            "final_score": {"left": 3, "right": 1},
            "timestamp": "2025-01-01T00:00:00",
            "version": "0.1.0",
            "frame_count": 100,
            "action_count": 100,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        with open(ep_dir / "actions.jsonl", "w") as f:
            for t in range(5):
                f.write(json.dumps({"t": t, "action_left": 1, "action_right": 2}) + "\n")

        loader = ReplayLoader(ep_dir)
        loaded_meta = loader.load_metadata()
        assert loaded_meta["seed"] == 42
        assert loaded_meta["episode_id"] == "ep_test"

        actions = loader.load_actions()
        assert len(actions) == 5
        assert actions[0] == (0, 1, 2)

        loaded_config = loader.load_config()
        assert loaded_config.target_score == 3

        print("PASS: test_replay_loader")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_dataset_writer()
    test_replay_loader()
    print("\nAll logging tests passed!")
