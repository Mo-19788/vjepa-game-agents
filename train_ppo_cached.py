"""PPO with cached encoder — collect real game experience, then train fast.

Phase 1: Play N episodes with current policy, encode each frame ONCE, store (z, action, reward, done)
Phase 2: Run PPO updates on the cached latents (no encoder needed — instant)
Repeat.

This is much faster than encoding every frame during PPO updates.

Usage:
    python train_ppo_cached.py [--rounds 30] [--lanes 3] [--cars 2] [--slots]
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

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
from train_ppo_pixels import PolicyValueNet


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_val = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


def ppo_update(policy, optimizer, states, actions, old_log_probs,
               advantages, returns, epochs=6, clip_eps=0.2, batch_size=128):
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long)
    old_log_probs = torch.stack(old_log_probs).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = len(states)

    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            logits, values = policy(states[idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs[idx]).exp()
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns[idx])
            loss = policy_loss + 0.5 * value_loss - 0.02 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


def collect_episodes(encoder, preprocessor, policy, game_config,
                     num_episodes=20, max_steps=300):
    """Play episodes with current policy, return cached (z, action, reward, done)."""
    all_z, all_actions, all_log_probs, all_rewards, all_values, all_dones = \
        [], [], [], [], [], []

    total_crosses = 0; total_hits = 0

    for ep in range(num_episodes):
        env = CrosserEnv(game_config)
        obs = env.reset(seed=random.randint(0, 999999))
        prev_row = env._state.player.row

        for step in range(max_steps):
            frame = preprocessor(obs.frame).unsqueeze(0)
            with torch.no_grad():
                z = encoder(frame)
                action, log_prob, value = policy.get_action(z)

            result = env.step(action)

            # Dense reward
            cur_row = env._state.player.row
            reward = (prev_row - cur_row) * 0.3  # row progress
            prev_row = cur_row

            if result.info.get("scored"):
                reward = 10.0
                total_crosses += 1
                prev_row = env._state.player.row
            if result.info.get("hit"):
                reward = -3.0
                total_hits += 1
                prev_row = env._state.player.row

            all_z.append(z.squeeze(0))
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_rewards.append(reward)
            all_values.append(value.item())
            all_dones.append(float(result.done))

            obs = result.observation
            if result.done:
                break

    return (all_z, all_actions, all_log_probs, all_rewards, all_values, all_dones,
            total_crosses, total_hits)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=50,
                        help='Number of collect-then-train rounds')
    parser.add_argument('--eps-per-round', type=int, default=20,
                        help='Episodes to collect per round')
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--lanes', type=int, default=None)
    parser.add_argument('--cars', type=int, default=None)
    parser.add_argument('--slots', action='store_true')
    args = parser.parse_args()

    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)

    if args.slots:
        ckpt = 'crosser_agent/checkpoints_slots'
        encoder = SlotEncoder(agent_config, num_slots=8, slot_dim=64, num_iters=3)
        encoder.load_state_dict(torch.load(f'{ckpt}/slot_encoder.pt',
                                           map_location='cpu', weights_only=True))
        print('Slot encoder (frozen)')
    else:
        ckpt = 'crosser_agent/checkpoints'
        encoder = Encoder(agent_config)
        encoder.load_state_dict(torch.load(f'{ckpt}/encoder.pt',
                                           map_location='cpu', weights_only=True))
        print('Original encoder (frozen)')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    policy = PolicyValueNet(latent_dim=256, num_actions=5)
    optimizer = Adam(policy.parameters(), lr=3e-4)
    print(f'Policy: {sum(p.numel() for p in policy.parameters()):,} params')

    game_config = Config(); game_config.headless = True
    game_config.max_steps = args.max_steps; game_config.target_score = 999
    if args.lanes: game_config.num_lanes = args.lanes
    if args.cars: game_config.max_cars_per_lane = args.cars
    print(f'Game: {game_config.num_lanes} lanes, {game_config.max_cars_per_lane} cars')

    recent_crosses = deque(maxlen=10)
    recent_hits = deque(maxlen=10)
    best_ratio = 0
    t0 = time.time()

    print(f'\n{args.rounds} rounds x {args.eps_per_round} eps/round = '
          f'{args.rounds * args.eps_per_round} total episodes\n', flush=True)

    for rnd in range(1, args.rounds + 1):
        # Phase 1: Collect experience with current policy
        data = collect_episodes(encoder, preprocessor, policy, game_config,
                                num_episodes=args.eps_per_round,
                                max_steps=args.max_steps)
        z_list, actions, log_probs, rewards, values, dones, crosses, hits = data

        recent_crosses.append(crosses / args.eps_per_round)
        recent_hits.append(hits / args.eps_per_round)

        # Phase 2: PPO update on cached latents (fast — no encoder)
        advantages, returns = compute_gae(rewards, values, dones)
        ppo_update(policy, optimizer, z_list, actions, log_probs,
                   advantages, returns)

        avg_c = np.mean(recent_crosses)
        avg_h = np.mean(recent_hits)
        ratio = avg_c / max(0.1, avg_h)
        elapsed = time.time() - t0

        total_eps = rnd * args.eps_per_round
        print(f'Round {rnd:>3d} ({total_eps:>4d} eps) | '
              f'Crosses: {avg_c:.1f} | Hits: {avg_h:.1f} | '
              f'Ratio: {ratio:.2f} | Steps: {len(z_list)} | '
              f'{elapsed:.0f}s', flush=True)

        if ratio > best_ratio and rnd >= 3:
            best_ratio = ratio
            torch.save(policy.state_dict(), f'{ckpt}/policy_best.pt')

    torch.save(policy.state_dict(), f'{ckpt}/policy.pt')
    print(f'\nDone! Best ratio: {best_ratio:.2f}')
    print(f'Saved to: {ckpt}/policy.pt')


if __name__ == '__main__':
    main()
