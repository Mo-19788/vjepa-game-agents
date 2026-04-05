"""Manual gameplay recorder.

Run this script while playing an emulator game. It captures frames and
detects your keyboard input to build a training dataset.

Hotkeys:
    Arrow keys           - gameplay (mapped to Action enum)
    F5                   - mark episode end (done=True)
    F6                   - mark positive reward (+1)
    F7                   - mark negative reward (-1)
    ESC                  - stop recording
"""

import argparse
import time

import keyboard
import numpy as np

from game_agent.actions import Action
from game_agent.capture.frame_grabber import FrameGrabber
from game_agent.config import AgentConfig
from game_agent.training.dataset import TransitionBuffer
from game_agent.utils.logger import setup_logger

logger = setup_logger("recorder")

# Keyboard key -> Action mapping for detecting human input
# Adjust these to match your emulator's key bindings
KEY_ACTION_MAP = [
    (["left"], Action.LEFT),
    (["right"], Action.RIGHT),
    (["up"], Action.UP),
    (["down"], Action.DOWN),
]


def detect_action() -> Action:
    """Detect the current action from keyboard state."""
    for keys, action in KEY_ACTION_MAP:
        if all(keyboard.is_pressed(k) for k in keys):
            return action
    return Action.NOOP


def record(config: AgentConfig):
    grabber = FrameGrabber(window_title=config.window_title)
    buffer = TransitionBuffer(config.data_dir)

    dt = 1.0 / config.capture_fps
    prev_frame = None
    current_reward = 0.0
    current_done = False
    total_transitions = 0

    logger.info("Recording started. Press ESC to stop.")
    logger.info(f"Window: {config.window_title} | FPS: {config.capture_fps}")
    logger.info("Hotkeys: F5=done, F6=+reward, F7=-reward")

    try:
        while True:
            t0 = time.time()

            if keyboard.is_pressed("esc"):
                break

            # Check hotkeys
            if keyboard.is_pressed("f5"):
                current_done = True
                logger.info("Episode end marked")
            if keyboard.is_pressed("f6"):
                current_reward = 1.0
                logger.info("Positive reward marked")
            if keyboard.is_pressed("f7"):
                current_reward = -1.0
                logger.info("Negative reward marked")

            frame = grabber.grab()
            action = detect_action()

            if prev_frame is not None:
                buffer.add(
                    obs=prev_frame,
                    action=int(action),
                    next_obs=frame,
                    reward=current_reward,
                    done=current_done,
                )
                total_transitions += 1

                # Reset per-transition signals
                current_reward = 0.0
                current_done = False

            prev_frame = frame

            # Flush every 100 transitions
            if len(buffer.buffer) >= 100:
                n = buffer.flush()
                logger.info(f"Flushed {n} transitions to disk (total: {total_transitions})")

            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        pass

    # Final flush
    n = buffer.flush()
    logger.info(f"Recording stopped. Final flush: {n} transitions. Total: {total_transitions}")


def main():
    parser = argparse.ArgumentParser(description="Record gameplay for training")
    parser.add_argument("--window", type=str, default=None, help="Game window title")
    parser.add_argument("--fps", type=int, default=None, help="Capture FPS")
    parser.add_argument("--data-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = AgentConfig()
    if args.window:
        config.window_title = args.window
    if args.fps:
        config.capture_fps = args.fps
    if args.data_dir:
        config.data_dir = args.data_dir

    record(config)


if __name__ == "__main__":
    main()
