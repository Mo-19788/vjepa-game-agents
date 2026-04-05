"""Generate diverse training data for the V-JEPA agent.

Runs multiple bot configurations to get varied paddle behavior and more scoring events.
"""

import subprocess
import sys

CONFIGS = [
    # (episodes, policy_left, policy_right, difficulty, target_score, label)
    (30, "bot", "bot", "easy", 10, "easy_vs_easy"),
    (30, "bot", "bot", "medium", 10, "med_vs_med"),
    (30, "bot", "bot", "hard", 10, "hard_vs_hard"),
    (20, "bot", "bot", "easy", 10, "easy_left_vs_hard_right"),
    (20, "bot", "bot", "hard", 10, "hard_left_vs_easy_right"),
    (20, "random", "bot", "medium", 10, "random_vs_med"),
    (20, "sticky_random", "bot", "medium", 10, "sticky_vs_med"),
]

OUT_DIR = "dataset_v2"

def run(episodes, policy_left, policy_right, difficulty, target_score, label):
    cmd = [
        sys.executable, "main.py",
        "--mode", "generate",
        "--episodes", str(episodes),
        "--policy-left", policy_left,
        "--policy-right", policy_right,
        "--bot-difficulty", difficulty,
        "--target-score", str(target_score),
        "--max-steps", "10000",
        "--headless",
        "--out", OUT_DIR,
    ]
    print(f"\n{'='*60}")
    print(f"Generating: {label} ({episodes} episodes)")
    print(f"  Left: {policy_left} | Right: {policy_right} | Difficulty: {difficulty}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    for cfg in CONFIGS:
        run(*cfg)
    print(f"\nDone! All data saved to {OUT_DIR}/")
