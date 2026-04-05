"""Calibrate the position probe by comparing predictions to ground truth on live frames."""

import sys, os, time
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
    game_config = Config()
    game_config.headless = True  # no window needed

    agent_config = AgentConfig()
    device = torch.device('cpu')

    ckpt = os.path.join(os.path.dirname(__file__), "..", agent_config.checkpoint_dir)
    encoder = Encoder(agent_config).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ckpt, "encoder.pt"), map_location=device, weights_only=True))
    encoder.eval()

    probe = PositionProbe(agent_config).to(device)
    probe.load_state_dict(torch.load(os.path.join(ckpt, "position_probe.pt"), map_location=device, weights_only=True))
    probe.eval()

    preprocessor = Preprocessor(agent_config)

    rng = make_rng(42)
    bot_l = BotPolicy(difficulty="medium", rng=rng)
    bot_r = BotPolicy(difficulty="medium", rng=rng)

    env = PongEnv(game_config)
    obs = env.reset(seed=42)

    errors_ball_y = []
    errors_paddle_y = []

    for step in range(500):
        frame = obs.frame
        state = env._state

        gt_ball_y = state.ball.y / 480.0
        gt_paddle_y = state.left_paddle.y / 480.0

        obs_tensor = preprocessor(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(obs_tensor)
            pos = probe(z)[0]

        pred_ball_y = pos[1].item()
        pred_paddle_y = pos[2].item()

        errors_ball_y.append(abs(pred_ball_y - gt_ball_y))
        errors_paddle_y.append(abs(pred_paddle_y - gt_paddle_y))

        if step < 20 or step % 100 == 0:
            print(f"Step {step:3d} | GT ball_y={gt_ball_y:.3f} pred={pred_ball_y:.3f} err={abs(pred_ball_y-gt_ball_y):.3f} | "
                  f"GT pad_y={gt_paddle_y:.3f} pred={pred_paddle_y:.3f} err={abs(pred_paddle_y-gt_paddle_y):.3f}")

        action_l = bot_l.get_action(state, "left")
        action_r = bot_r.get_action(state, "right")
        result = env.step(action_l, action_r)
        obs = result.observation

    print(f"\nMean error ball_y:   {np.mean(errors_ball_y):.4f} (RMSE {np.sqrt(np.mean(np.array(errors_ball_y)**2)):.4f})")
    print(f"Mean error paddle_y: {np.mean(errors_paddle_y):.4f} (RMSE {np.sqrt(np.mean(np.array(errors_paddle_y)**2)):.4f})")
    print(f"Probe ball_y range:  {min(pos[1].item() for _ in [1]):.3f} to {max(pos[1].item() for _ in [1]):.3f}")


if __name__ == "__main__":
    main()
