"""Dreamer-style actor-critic trained on imagined rollouts in latent space.

All frozen: slot encoder, dynamics, position probe, car probe.
Only trains: actor (policy) + critic (value) — two small MLPs.

The trick: we never render frames during training. We:
1. Collect a pool of real latent states (encode a batch of game frames once)
2. From those starting states, imagine H-step rollouts using the dynamics model
3. Score imagined trajectories using position probe (progress) + car probe (danger)
4. Train actor-critic on those imagined rewards via backprop through the dream

CPU-friendly — millions of latent steps in minutes.

Usage:
    python train_dreamer.py [--steps 50000] [--lanes 3] [--cars 2]
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import sys, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, '.'); sys.path.insert(0, 'crosser')

from crosser.config import Config
from crosser.env.crosser_env import CrosserEnv
from crosser.env.state import NOOP, UP, DOWN, LEFT, RIGHT
from game_agent.config import AgentConfig
from game_agent.models.slot_attention import SlotEncoder
from game_agent.models.encoder import Encoder
from game_agent.preprocessing.transforms import Preprocessor

GRID_ROWS = 12
GOAL_ROW_NORM = 2 / 11  # ~0.182
CAR_LANE_START_NORM = 2 / 11
CAR_LANE_END_NORM = 7 / 11


# ---- Actor-Critic ----

class Actor(nn.Module):
    """Latent → action distribution."""
    def __init__(self, latent_dim=256, num_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, z):
        return self.net(z)

    def get_dist(self, z):
        logits = self.forward(z)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    """Latent → value estimate."""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ---- Dynamics + Probes (frozen, from training) ----

class DynamicsPredictor(nn.Module):
    def __init__(self, latent_dim=256, num_actions=5):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 64)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 64, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
    def forward(self, z, action):
        a = self.action_embed(action.long())
        return self.norm(z + self.net(torch.cat([z, a], -1)))


class PositionProbe(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


class CarOccupancyProbe(nn.Module):
    def __init__(self, latent_dim=256, grid_size=144):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, grid_size),
        )
    def forward(self, z): return self.net(z)


# ---- Reward function in latent space ----

def compute_dream_reward(z, pos_probe, car_probe, prev_row):
    """Compute reward from latent state using frozen probes.

    Returns (reward, current_row, done) per batch element.
    When danger > threshold, the episode is "done" (simulated hit).
    """
    HIT_THRESHOLD = 0.5  # car probe confidence above this = hit

    with torch.no_grad():
        pos = pos_probe(z)          # (B, 2) — col, row in [0,1]
        row = pos[:, 1]             # (B,)
        col = pos[:, 0]             # (B,)

        occ_logits = car_probe(z)   # (B, 144)
        occ_probs = torch.sigmoid(occ_logits).view(-1, 12, 12)  # (B, 12, 12)

    # Row progress reward (moving up = positive)
    row_progress = (prev_row - row) * 2.0

    # Crossing bonus
    crossed = (row < GOAL_ROW_NORM).float() * 10.0

    # Car danger at current position
    grid_row = (row * 11).long().clamp(0, 11)
    grid_col = (col * 11).long().clamp(0, 11)
    danger = torch.zeros_like(row)
    for b in range(z.shape[0]):
        r, c = grid_row[b].item(), grid_col[b].item()
        if CAR_LANE_START_NORM <= row[b].item() <= CAR_LANE_END_NORM:
            d = occ_probs[b, r, c].item()
            d = max(d, 0.5 * occ_probs[b, r, max(0, c-1)].item())
            d = max(d, 0.5 * occ_probs[b, r, min(11, c+1)].item())
            danger[b] = d

    # Simulated hit: high danger = episode done with big penalty
    hit = (danger > HIT_THRESHOLD).float()
    hit_penalty = -hit * 10.0

    # Alive penalty for being in danger (even below threshold)
    danger_penalty = -danger * 2.0

    # Edge penalty
    edge = ((col < 0.05) | (col > 0.95)).float() * -0.5

    reward = row_progress + crossed + hit_penalty + danger_penalty + edge
    return reward, row, hit


# ---- Collect real starting states ----

def collect_starting_states(encoder, preprocessor, num_states=2000,
                            num_episodes=50, lanes=None, cars=None):
    """Play random episodes, encode frames, return pool of real latent states."""
    print(f'Collecting {num_states} real latent states...', flush=True)
    game_config = Config(); game_config.headless = True
    if lanes: game_config.num_lanes = lanes
    if cars: game_config.max_cars_per_lane = cars

    rng = np.random.RandomState(42)
    actions = [NOOP, UP, DOWN, LEFT, RIGHT]
    all_z = []

    for ep in range(num_episodes):
        env = CrosserEnv(game_config)
        obs = env.reset(seed=rng.randint(0, 999999))

        for step in range(200):
            if len(all_z) >= num_states:
                break
            frame = preprocessor(obs.frame).unsqueeze(0)
            with torch.no_grad():
                z = encoder(frame).squeeze(0)
            all_z.append(z)

            action = rng.choice(actions)
            result = env.step(action)
            obs = result.observation
            if result.done:
                obs = env.reset(seed=rng.randint(0, 999999))

        if len(all_z) >= num_states:
            break

    all_z = torch.stack(all_z[:num_states])
    print(f'Collected {len(all_z)} latent states', flush=True)
    return all_z


# ---- Dreamer training loop ----

def train_dreamer(args):
    device = torch.device('cpu')

    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)

    # Load frozen models
    if args.slots:
        ckpt = 'crosser_agent/checkpoints_slots'
        encoder = SlotEncoder(agent_config, num_slots=8, slot_dim=64, num_iters=3)
        encoder.load_state_dict(torch.load(f'{ckpt}/slot_encoder.pt', map_location='cpu', weights_only=True))
        print('Slot encoder loaded (frozen)')
    else:
        ckpt = 'crosser_agent/checkpoints'
        encoder = Encoder(agent_config)
        encoder.load_state_dict(torch.load(f'{ckpt}/encoder.pt', map_location='cpu', weights_only=True))
        print('Original encoder loaded (frozen)')
    encoder.eval()

    dynamics = DynamicsPredictor()
    dynamics.load_state_dict(torch.load(f'{ckpt}/dynamics.pt', map_location='cpu', weights_only=True))
    dynamics.eval()

    pos_probe = PositionProbe()
    pos_probe.load_state_dict(torch.load(f'{ckpt}/position_probe.pt', map_location='cpu', weights_only=True))
    pos_probe.eval()

    car_probe = CarOccupancyProbe()
    car_probe.load_state_dict(torch.load(f'{ckpt}/car_probe.pt', map_location='cpu', weights_only=True))
    car_probe.eval()

    # Actor-Critic (only these get trained)
    actor = Actor()
    critic = Critic()
    target_critic = copy.deepcopy(critic)
    actor_opt = Adam(actor.parameters(), lr=3e-4)
    critic_opt = Adam(critic.parameters(), lr=3e-4)

    a_params = sum(p.numel() for p in actor.parameters())
    c_params = sum(p.numel() for p in critic.parameters())
    print(f'Actor: {a_params:,} params | Critic: {c_params:,} params')

    # Collect pool of real starting states (one-time cost)
    z_pool = collect_starting_states(
        encoder, preprocessor, num_states=args.pool_size,
        num_episodes=100, lanes=args.lanes, cars=args.cars)

    # Training
    horizon = args.horizon
    batch_size = args.batch_size
    gamma = 0.99
    lam = 0.95
    target_update = 100  # update target critic every N steps

    print(f'\nDreamer training: {args.steps} steps, horizon={horizon}, '
          f'batch={batch_size}', flush=True)
    print(f'All in latent space — no frames rendered during training\n', flush=True)

    t0 = time.time()
    best_avg_reward = -float('inf')

    for step in range(1, args.steps + 1):
        # Sample batch of real starting states
        idx = torch.randint(0, len(z_pool), (batch_size,))
        z = z_pool[idx].clone()  # (B, 256)

        # ── Imagine H-step rollout ──
        imagined_z = []
        imagined_actions = []
        imagined_log_probs = []
        imagined_rewards = []
        imagined_values = []

        prev_row = pos_probe(z)[:, 1].detach()
        done_mask = torch.zeros(batch_size)  # tracks which episodes are done

        for h in range(horizon):
            # Actor picks action
            dist = actor.get_dist(z)
            action = dist.sample()       # (B,)
            log_prob = dist.log_prob(action)

            # Critic evaluates current state
            value = critic(z)

            # Store
            imagined_z.append(z)
            imagined_actions.append(action)
            imagined_log_probs.append(log_prob)
            imagined_values.append(value)

            # Step dynamics (frozen)
            with torch.no_grad():
                z_next = dynamics(z, action)

            # Compute reward from probes (frozen)
            reward, cur_row, hit = compute_dream_reward(
                z_next, pos_probe, car_probe, prev_row)

            # Zero out reward for already-done episodes
            reward = reward * (1 - done_mask)
            imagined_rewards.append(reward)

            # Update done mask
            done_mask = torch.clamp(done_mask + hit, 0, 1)

            # Reset hit episodes to random starting states
            hit_idx = (hit > 0).nonzero(as_tuple=True)[0]
            if len(hit_idx) > 0:
                reset_idx = torch.randint(0, len(z_pool), (len(hit_idx),))
                z_next[hit_idx] = z_pool[reset_idx]
                prev_row_new = pos_probe(z_next)[:, 1].detach()
                prev_row = prev_row.clone()
                prev_row[hit_idx] = prev_row_new[hit_idx]
                done_mask[hit_idx] = 0  # allow new episode to continue
            else:
                prev_row = cur_row.detach()

            z = z_next.detach()

        # ── Compute returns with GAE ──
        with torch.no_grad():
            bootstrap_value = target_critic(z)  # value of final state

        values = torch.stack(imagined_values)    # (H, B)
        rewards = torch.stack(imagined_rewards)  # (H, B)
        log_probs = torch.stack(imagined_log_probs)  # (H, B)

        # GAE
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size)
        for t in reversed(range(horizon)):
            if t == horizon - 1:
                next_val = bootstrap_value
            else:
                next_val = values[t + 1].detach()
            delta = rewards[t] + gamma * next_val - values[t].detach()
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        returns = advantages + values.detach()

        # ── Update Critic ──
        critic_loss = F.mse_loss(values, returns)
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()

        # ── Update Actor (policy gradient with advantages) ──
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = -(log_probs * adv_norm.detach()).mean()
        entropy = torch.stack([actor.get_dist(z_i).entropy().mean()
                               for z_i in imagined_z]).mean()
        actor_total = actor_loss - 0.01 * entropy

        actor_opt.zero_grad()
        actor_total.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step()

        # Target critic soft update
        if step % target_update == 0:
            with torch.no_grad():
                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.mul_(0.995).add_(p.data, alpha=0.005)

        # ── Logging ──
        avg_reward = rewards.mean().item()
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(actor.state_dict(), f'{ckpt}/dreamer_actor_best.pt')
            torch.save(critic.state_dict(), f'{ckpt}/dreamer_critic_best.pt')

        if step % 500 == 0 or step <= 10:
            elapsed = time.time() - t0
            avg_r = rewards.sum(0).mean().item()  # total reward per episode
            with torch.no_grad():
                # Check action distribution
                test_dist = actor.get_dist(z_pool[:100])
                action_probs = test_dist.probs.mean(0)
            print(f'Step {step:>5d} | R/ep: {avg_r:+.2f} | '
                  f'A_loss: {actor_loss.item():.4f} | C_loss: {critic_loss.item():.4f} | '
                  f'Actions: [{" ".join(f"{p:.2f}" for p in action_probs.tolist())}] | '
                  f'{elapsed:.0f}s', flush=True)

    # Save final
    torch.save(actor.state_dict(), f'{ckpt}/dreamer_actor.pt')
    torch.save(critic.state_dict(), f'{ckpt}/dreamer_critic.pt')
    print(f'\nDreamer training done! Best avg reward: {best_avg_reward:.3f}')
    print(f'Saved to: {ckpt}/dreamer_actor.pt')


# ---- Evaluate in real game ----

def evaluate(args):
    """Run the trained dreamer actor in the real game."""
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    device = torch.device('cpu')
    agent_config = AgentConfig(); agent_config.num_actions = 5
    preprocessor = Preprocessor(agent_config)

    if args.slots:
        ckpt = 'crosser_agent/checkpoints_slots'
        encoder = SlotEncoder(agent_config, num_slots=8, slot_dim=64, num_iters=3)
        encoder.load_state_dict(torch.load(f'{ckpt}/slot_encoder.pt', map_location='cpu', weights_only=True))
    else:
        ckpt = 'crosser_agent/checkpoints'
        encoder = Encoder(agent_config)
        encoder.load_state_dict(torch.load(f'{ckpt}/encoder.pt', map_location='cpu', weights_only=True))
    encoder.eval()

    actor = Actor()
    actor_path = f'{ckpt}/dreamer_actor_best.pt'
    if not os.path.exists(actor_path):
        actor_path = f'{ckpt}/dreamer_actor.pt'
    actor.load_state_dict(torch.load(actor_path, map_location='cpu', weights_only=True))
    actor.eval()

    game_config = Config(); game_config.headless = True
    game_config.max_steps = 500; game_config.target_score = 999
    if args.lanes: game_config.num_lanes = args.lanes
    if args.cars: game_config.max_cars_per_lane = args.cars

    print(f'Evaluating dreamer actor ({args.episodes} episodes, '
          f'{game_config.num_lanes} lanes, {game_config.max_cars_per_lane} cars)')

    total_crosses = 0; total_hits = 0
    for ep in range(1, args.episodes + 1):
        env = CrosserEnv(game_config)
        obs = env.reset(seed=random.randint(0, 999999))
        ep_c = 0; ep_h = 0

        for step in range(500):
            frame = preprocessor(obs.frame).unsqueeze(0)
            with torch.no_grad():
                z = encoder(frame)
                logits = actor(z)
                action = int(logits.argmax(dim=-1).item())

            result = env.step(action)
            obs = result.observation
            if result.info.get("scored"): ep_c += 1
            if result.info.get("hit"): ep_h += 1
            if result.done: break

        total_crosses += ep_c; total_hits += ep_h
        ratio = ep_c / max(1, ep_h)
        print(f'  Ep {ep:>2d}: crosses={ep_c}  hits={ep_h}  ratio={ratio:.1f}')

    ratio = total_crosses / max(1, total_hits)
    print(f'TOTAL: crosses={total_crosses}  hits={total_hits}  ratio={ratio:.2f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--pool-size', type=int, default=2000)
    parser.add_argument('--lanes', type=int, default=None)
    parser.add_argument('--cars', type=int, default=None)
    parser.add_argument('--slots', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train_dreamer(args)
