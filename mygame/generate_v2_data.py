"""Generate V2 training data with bigger, more colorful visuals.

Bigger ball, bigger paddles, colors — more visual signal for the encoder
to differentiate actions in the dynamics model.
"""

import subprocess
import sys
import json
from pathlib import Path

# Create V2 config with bigger, colorful visuals
from config import Config

v2 = Config()
v2.ball_size = 24
v2.paddle_height = 100
v2.paddle_width = 16
v2.ball_color = (255, 255, 0)       # yellow ball
v2.paddle_color = (0, 200, 255)     # cyan paddles
v2.bg_color = (20, 20, 40)          # dark blue background
v2.line_color = (60, 60, 100)       # subtle center line
v2.ball_speed = 5.0
v2.paddle_speed = 5.0
v2.headless = True
v2.save_frames = True
v2.save_actions = True
v2.save_states = True

config_path = Path("config_v2.json")
v2.save(config_path)
print(f"V2 config saved to {config_path}")

OUT_DIR = "dataset_v2_visual"

CONFIGS = [
    # (episodes, policy_left, policy_right, difficulty, target_score)
    (30, "bot", "bot", "easy", 10),
    (30, "bot", "bot", "medium", 10),
    (30, "bot", "bot", "hard", 10),
    (20, "random", "bot", "medium", 10),
    (20, "sticky_random", "bot", "medium", 10),
]

for episodes, pl, pr, diff, ts in CONFIGS:
    label = f"{pl}_vs_{pr}_{diff}"
    print(f"\n{'='*50}")
    print(f"Generating: {label} ({episodes} episodes)")
    print(f"{'='*50}")

    cmd = [
        sys.executable, "main.py",
        "--mode", "generate",
        "--episodes", str(episodes),
        "--policy-left", pl,
        "--policy-right", pr,
        "--bot-difficulty", diff,
        "--target-score", str(ts),
        "--max-steps", "10000",
        "--headless",
        "--out", OUT_DIR,
        "--config", str(config_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed: {e}")
        # Try without --config flag if not supported
        cmd_no_config = [c for c in cmd if c != "--config" and c != str(config_path)]
        print("Retrying without --config...")
        subprocess.run(cmd_no_config, check=True)

print(f"\nDone! Data saved to {OUT_DIR}/")
