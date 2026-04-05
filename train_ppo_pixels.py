"""PPO policy on top of frozen slot encoder — pure pixels, no game state.

The slot encoder is frozen (pretrained). We only train a small policy + value
network that maps the 256-dim latent to actions.

Runs on CPU. No GPU needed.

Usage:
    python train_ppo_pixels.py [--episodes 2000] [--lanes 3] [--cars 2]
"""
import os
import sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque

sys.path.insert(0, '.'); sys.path.insert(0, 'crosser')

from crosser.config import Config
from crosser.env.crosser_env import CrosserEnv
from crosser.env.state import NOOP, UP, DOWN, LEFT, RIGHT
from game_agent.config import AgentConfig
from game_agent.models.slot_attention import SlotEncoder
from game_agent.models.encoder import Encoder
from game_agent.preprocessing.transforms import Preprocessor


class PolicyValueNet(nn.Module):
    """Small MLP: latent → policy logits + value estimate."""
    def __init__(self, latent_dim=256, num_actions=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, z):
        h = self.shared(z)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def get_action(self, z):
        """Sample action from policy, return action + log_prob + value."""
        logits, value = self.forward(z)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


def ppo_update(policy_net, optimizer, states, actions, old_log_probs,
               advantages, returns, epochs=4, clip_eps=0.2, batch_size=64):
    """PPO clipped objective update."""
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long)
    old_log_probs = torch.stack(old_log_probs).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(states)
    total_loss = 0; steps = 0

    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            logits, values = policy_net(states[idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy().mean()

            # PPO clipped ratio
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns[idx])

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item(); steps += 1

    return total_loss / max(1, steps)


def train(args):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    device = torch.device('cpu')

    # Load frozen encoder
    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)

    if args.slots:
        encoder = SlotEncoder(agent_config, num_slots=8, slot_dim=64, num_iters=3)
        encoder.load_state_dict(torch.load(
            'crosser_agent/checkpoints_slots/slot_encoder.pt',
            map_location='cpu', weights_only=True))
        ckpt_dir = 'crosser_agent/checkpoints_slots'
        print(f'Slot encoder loaded (frozen)')
    else:
        encoder = Encoder(agent_config)
        encoder.load_state_dict(torch.load(
            'crosser_agent/checkpoints/encoder.pt',
            map_location='cpu', weights_only=True))
        ckpt_dir = 'crosser_agent/checkpoints'
        print(f'Original encoder loaded (frozen)')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Policy network (tiny — only this gets trained)
    policy_net = PolicyValueNet(latent_dim=256, num_actions=5)
    optimizer = Adam(policy_net.parameters(), lr=3e-4)
    print(f'Policy net: {sum(p.numel() for p in policy_net.parameters()):,} params')

    # Game env
    game_config = Config(); game_config.headless = True
    game_config.max_steps = args.max_steps; game_config.target_score = 999
    if args.lanes is not None:
        game_config.num_lanes = args.lanes
    if args.cars is not None:
        game_config.max_cars_per_lane = args.cars
    print(f'Game: {game_config.num_lanes} lanes, {game_config.max_cars_per_lane} cars/lane, '
          f'speed {game_config.max_car_speed}')

    # Training loop
    rollout_steps = args.rollout  # collect this many steps before updating
    total_episodes = 0
    total_steps = 0
    recent_crosses = deque(maxlen=50)
    recent_hits = deque(maxlen=50)
    best_ratio = 0

    # Reward shaping
    CROSS_REWARD = 10.0
    HIT_PENALTY = -3.0

    t0 = time.time()
    env = CrosserEnv(game_config)
    obs = env.reset(seed=random.randint(0, 999999))
    ep_crosses = 0; ep_hits = 0
    prev_row = env._state.player.row

    print(f'\nTraining PPO for {args.episodes} episodes (rollout={rollout_steps})...\n')

    while total_episodes < args.episodes:
        # Collect rollout
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        for _ in range(rollout_steps):
            frame = preprocessor(obs.frame).unsqueeze(0)
            with torch.no_grad():
                z = encoder(frame)

            action, log_prob, value = policy_net.get_action(z)

            result = env.step(action)

            # Shaped reward — dense signal based on row progress
            cur_row = env._state.player.row
            row_progress = (prev_row - cur_row) * 0.3  # reward moving up
            prev_row = cur_row
            reward = row_progress

            if result.info.get("scored"):
                reward = CROSS_REWARD
                ep_crosses += 1
                prev_row = env._state.player.row  # reset after respawn
            if result.info.get("hit"):
                reward = HIT_PENALTY
                ep_hits += 1
                prev_row = env._state.player.row  # reset after respawn

            states.append(z.squeeze(0))
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(float(result.done))

            obs = result.observation
            total_steps += 1

            if result.done:
                recent_crosses.append(ep_crosses)
                recent_hits.append(ep_hits)
                total_episodes += 1
                ep_crosses = 0; ep_hits = 0
                obs = env.reset(seed=random.randint(0, 999999))

                if total_episodes % 50 == 0:
                    avg_c = np.mean(recent_crosses) if recent_crosses else 0
                    avg_h = np.mean(recent_hits) if recent_hits else 0
                    ratio = avg_c / max(1, avg_h)
                    elapsed = time.time() - t0
                    print(f'Ep {total_episodes:>5d} | Crosses: {avg_c:.1f} | '
                          f'Hits: {avg_h:.1f} | Ratio: {ratio:.2f} | '
                          f'{elapsed:.0f}s', flush=True)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        torch.save(policy_net.state_dict(),
                                   f'{ckpt_dir}/policy_best.pt')

        # PPO update
        advantages, returns = compute_gae(rewards, values, dones)
        loss = ppo_update(policy_net, optimizer, states, actions, log_probs,
                          advantages, returns)

    # Save final
    torch.save(policy_net.state_dict(), f'{ckpt_dir}/policy.pt')
    avg_c = np.mean(recent_crosses) if recent_crosses else 0
    avg_h = np.mean(recent_hits) if recent_hits else 0
    print(f'\nTraining done! Final avg: {avg_c:.1f} crosses, {avg_h:.1f} hits, '
          f'ratio {avg_c/max(1,avg_h):.2f}')
    print(f'Best ratio: {best_ratio:.2f}')
    print(f'Saved to: crosser_agent/checkpoints_slots/policy.pt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--rollout', type=int, default=256)
    parser.add_argument('--lanes', type=int, default=None)
    parser.add_argument('--cars', type=int, default=None)
    parser.add_argument('--slots', action='store_true')
    args = parser.parse_args()
    train(args)
