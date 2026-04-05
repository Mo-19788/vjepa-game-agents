"""End-to-end DQN from pixels. No frozen encoder — CNN learns its own features.

Standard Atari-style DQN:
- CNN processes 84x84 grayscale frames (4 stacked for motion)
- Outputs Q-values for 5 actions
- Trains with experience replay + target network

Usage:
    python train_dqn.py [--steps 100000] [--lanes 3] [--cars 2]
"""
import os
import sys, time, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, '.'); sys.path.insert(0, 'crosser')

from crosser.config import Config
from crosser.env.crosser_env import CrosserEnv
from crosser.env.state import NOOP, UP, DOWN, LEFT, RIGHT

import cv2


# ---- DQN Network ----

class DQN(nn.Module):
    """Atari-style CNN: 84x84x4 → Q-values for 5 actions."""
    def __init__(self, in_channels=4, num_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 84 → 20 → 9 → 7, so 7*7*64 = 3136
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        """x: (B, 4, 84, 84) float32 in [0, 1]."""
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


# ---- Frame preprocessing ----

def preprocess_frame(frame):
    """RGB 512x512 → grayscale 84x84 float32 [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    """Stack 4 frames for motion detection."""
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.k):
            self.frames.append(processed)
        return self._get()

    def push(self, frame):
        self.frames.append(preprocess_frame(frame))
        return self._get()

    def _get(self):
        return np.stack(list(self.frames), axis=0)  # (4, 84, 84)


# ---- Replay Buffer ----

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---- Training ----

def train(args):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    device = torch.device('cpu')

    game_config = Config(); game_config.headless = True
    game_config.max_steps = 500; game_config.target_score = 999
    if args.lanes: game_config.num_lanes = args.lanes
    if args.cars: game_config.max_cars_per_lane = args.cars
    print(f'Game: {game_config.num_lanes} lanes, {game_config.max_cars_per_lane} cars')

    # Networks
    q_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    params = sum(p.numel() for p in q_net.parameters())
    print(f'DQN: {params:,} params')

    optimizer = Adam(q_net.parameters(), lr=1e-4)
    replay = ReplayBuffer(capacity=args.buffer_size)
    frame_stack = FrameStack(k=4)

    # Hyperparams
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = args.steps // 2  # linear decay over half of training
    batch_size = 32
    learn_start = 1000  # start learning after this many steps
    target_update = 1000  # sync target net every N steps

    # Reward shaping
    CROSS_REWARD = 10.0
    HIT_PENALTY = -5.0

    env = CrosserEnv(game_config)
    obs = env.reset(seed=random.randint(0, 999999))
    state = frame_stack.reset(obs.frame)
    prev_row = env._state.player.row

    total_crosses = 0; total_hits = 0
    ep_crosses = 0; ep_hits = 0
    recent_crosses = deque(maxlen=50)
    recent_hits = deque(maxlen=50)
    ep_count = 0
    best_ratio = 0
    t0 = time.time()

    print(f'\nTraining DQN for {args.steps} steps...\n', flush=True)

    for step in range(1, args.steps + 1):
        # Epsilon-greedy action selection
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay)
        if random.random() < eps:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_vals = q_net(state_t)
                action = q_vals.argmax(dim=1).item()

        # Step environment
        result = env.step(action)
        next_state = frame_stack.push(result.observation.frame)

        # Shaped reward
        cur_row = env._state.player.row
        reward = (prev_row - cur_row) * 0.5  # row progress
        prev_row = cur_row

        if result.info.get("scored"):
            reward = CROSS_REWARD
            ep_crosses += 1
            prev_row = env._state.player.row
        if result.info.get("hit"):
            reward = HIT_PENALTY
            ep_hits += 1
            prev_row = env._state.player.row

        done = result.done
        replay.push(state, action, reward, next_state, float(done))
        state = next_state

        if done:
            recent_crosses.append(ep_crosses)
            recent_hits.append(ep_hits)
            ep_count += 1
            total_crosses += ep_crosses
            total_hits += ep_hits
            ep_crosses = 0; ep_hits = 0
            obs = env.reset(seed=random.randint(0, 999999))
            state = frame_stack.reset(obs.frame)
            prev_row = env._state.player.row

        # Learn from replay
        if len(replay) >= learn_start and step % 4 == 0:
            states_b, actions_b, rewards_b, next_states_b, dones_b = \
                replay.sample(batch_size)

            # Current Q values
            q_values = q_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

            # Target Q values (Double DQN)
            with torch.no_grad():
                next_actions = q_net(next_states_b).argmax(dim=1)
                next_q = target_net(next_states_b).gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_b + gamma * next_q * (1 - dones_b)

            loss = F.smooth_l1_loss(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        # Update target network
        if step % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Logging
        if step % 5000 == 0:
            avg_c = np.mean(recent_crosses) if recent_crosses else 0
            avg_h = np.mean(recent_hits) if recent_hits else 0
            ratio = avg_c / max(0.1, avg_h)
            elapsed = time.time() - t0
            print(f'Step {step:>6d} | Eps: {eps:.2f} | Eps_done: {ep_count} | '
                  f'Crosses: {avg_c:.1f} | Hits: {avg_h:.1f} | '
                  f'Ratio: {ratio:.2f} | {elapsed:.0f}s', flush=True)

            if ratio > best_ratio and ep_count >= 20:
                best_ratio = ratio
                torch.save(q_net.state_dict(), 'crosser_agent/dqn_best.pt')

    torch.save(q_net.state_dict(), 'crosser_agent/dqn.pt')
    avg_c = np.mean(recent_crosses) if recent_crosses else 0
    avg_h = np.mean(recent_hits) if recent_hits else 0
    print(f'\nDone! {ep_count} episodes, {step} steps')
    print(f'Final: {avg_c:.1f} crosses, {avg_h:.1f} hits, ratio {avg_c/max(0.1,avg_h):.2f}')
    print(f'Best ratio: {best_ratio:.2f}')


# ---- Evaluate ----

def evaluate(args):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    game_config = Config(); game_config.headless = True
    game_config.max_steps = 500; game_config.target_score = 999
    if args.lanes: game_config.num_lanes = args.lanes
    if args.cars: game_config.max_cars_per_lane = args.cars

    q_net = DQN()
    path = 'crosser_agent/dqn_best.pt'
    if not os.path.exists(path):
        path = 'crosser_agent/dqn.pt'
    q_net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    q_net.eval()
    print(f'Loaded {path}')

    frame_stack = FrameStack(k=4)
    total_c = 0; total_h = 0

    for ep in range(1, args.episodes + 1):
        env = CrosserEnv(game_config)
        obs = env.reset(seed=random.randint(0, 999999))
        state = frame_stack.reset(obs.frame)
        ec = 0; eh = 0

        for step in range(500):
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = q_net(state_t).argmax(dim=1).item()
            result = env.step(action)
            state = frame_stack.push(result.observation.frame)
            if result.info.get("scored"): ec += 1
            if result.info.get("hit"): eh += 1
            if result.done: break

        total_c += ec; total_h += eh
        print(f'  Ep {ep:>2d}: crosses={ec}  hits={eh}  ratio={ec/max(1,eh):.1f}')

    ratio = total_c / max(1, total_h)
    print(f'TOTAL: crosses={total_c}  hits={total_h}  ratio={ratio:.2f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lanes', type=int, default=None)
    parser.add_argument('--cars', type=int, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
