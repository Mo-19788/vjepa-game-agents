# V-JEPA Game Agents

World model agents that learn to play games from pixels using V-JEPA (Video Joint Embedding Predictive Architecture), Slot Attention, and DQN.

**Demo:** https://youtu.be/CmaSodNJw6k

Two games included:
- **Street Crosser** — Frogger-style lane crossing game
- **Pong** — Classic Pong with V-JEPA agent

## Quick Start

### Requirements

```bash
pip install torch pygame numpy opencv-python
```

### Play the Games

```bash
# Street Crosser (human play with settings panel)
cd crosser
python main.py

# Pong
python mygame/main.py
```

### Watch AI Agents Play

```bash
# Crosser — V-JEPA agent with dashboard
# Cycle planner mode: Pure / Enhanced / DQN / PPO
python crosser_agent/live_agent.py

# Crosser — with slot attention encoder
python crosser_agent/live_agent.py --slots

# Crosser — easy mode (3 lanes, 2 cars)
python crosser_agent/live_agent.py --lanes 3 --cars 2

# Pong — V-JEPA agent with dashboard
python mygame/vjepa_dashboard.py
```

### Benchmark Agents

```bash
# Compare all modes on easy
python crosser_agent/live_agent.py --bench --mode pure --lanes 3 --cars 2
python crosser_agent/live_agent.py --bench --slots --mode enhanced --lanes 3 --cars 2

# DQN benchmark
python train_dqn.py --eval --episodes 20 --lanes 3 --cars 2
```

## Architecture

### V-JEPA World Model

The agent learns a world model from pixels:

1. **CNN Encoder** — compresses 224x224 RGB frames to 256-dim latent vectors
2. **EMA Target Encoder** — slowly-updated copy prevents representation collapse
3. **Dynamics Predictor** — predicts next latent from current latent + action
4. **Position Probe** — reads player position from the latent
5. **Car Occupancy Probe** — detects car positions on a 12x12 grid
6. **Reward Head** — predicts reward signal

The agent plans by simulating actions in latent space and scoring outcomes.

### Slot Attention Encoder

An alternative encoder that decomposes the scene into K=8 object slots:

- CNN backbone -> 7x7 feature map -> Slot Attention -> 8 slots of 64 dims
- CNN decoder reconstructs the image from slots (training only)
- Occupancy auxiliary loss forces encoder to represent car positions
- Aggregation layer maps slots back to 256-dim for backward compatibility

### DQN (End-to-End)

Standard Atari-style DQN that learns directly from pixels:

- 84x84 grayscale, 4 stacked frames for motion detection
- CNN -> Q-values for 5 actions
- Double DQN with experience replay
- No world model, no probes — learns action values end-to-end

## Results (Street Crosser, 3 lanes / 2 cars)

| Approach | Cross:Hit Ratio | Pure Pixels? |
|----------|----------------|-------------|
| DQN (end-to-end) | **1.80** | Yes |
| V-JEPA Slot Enhanced | 1.47 | No (uses game state for position) |
| V-JEPA Original Pure | 0.93 | Yes |
| Reactive (always UP) | 0.15 | N/A |

The V-JEPA approach generalizes across different game configurations (car sizes, speeds, lane layouts, free-roam mode) because it works in latent space. The DQN achieves higher scores but is specific to the training configuration.

## Training

### Train DQN (CPU, ~15 min)

```bash
python train_dqn.py --steps 100000 --lanes 3 --cars 2
```

### Train V-JEPA World Model (GPU recommended)

```bash
# Generate training data
cd crosser
python main.py --mode generate --episodes 120

# Train full pipeline on GPU
python train_full_gpu.py

# Or train slot attention encoder
python train_slots_gpu.py

# Retrain probes locally (fix GPU/CPU mismatch)
python retrain_probes_local.py
```

### Train PPO Policy on Frozen Encoder (CPU)

```bash
python train_ppo_cached.py --rounds 50 --slots
```

## Project Structure

```
crosser/                    # Frogger-style game
  config.py                 # Game configuration
  main.py                   # Human play with settings panel
  env/                      # Game environment (physics, renderer, state)

crosser_agent/              # Crosser AI agents
  live_agent.py             # Dashboard with all agent modes
  checkpoints/              # Original V-JEPA models
  checkpoints_slots/        # Slot attention models
  dqn_best.pt              # Trained DQN model

game_agent/                 # Shared V-JEPA components
  models/
    encoder.py              # CNN encoder (256-dim latent)
    slot_attention.py       # Slot Attention encoder + decoder
    dynamics.py             # Action-conditioned dynamics
  preprocessing/
    transforms.py           # Frame preprocessing

mygame/                     # Pong game + V-JEPA agent
  main.py                   # Pong game
  vjepa_dashboard.py        # Pong V-JEPA agent dashboard

train_dqn.py                # End-to-end DQN training
train_slots_gpu.py          # Slot attention training (GPU)
train_full_gpu.py           # Original V-JEPA training (GPU)
train_ppo_pixels.py         # PPO on frozen encoder
train_ppo_cached.py         # PPO with cached latents
train_dreamer.py            # Dreamer-style imagined rollouts
retrain_probes_local.py     # Fix GPU/CPU probe mismatch
```

## Scene Presets (Street Crosser)

The live agent dashboard includes scene presets to test generalization:

- **Default** — 6 lanes, 3 cars, standard difficulty
- **Easy** — 3 lanes, 2 cars
- **Wide Cars** — cars 3 cells wide
- **Tall Vans** — cars spanning 2 rows
- **Buses** — 4 wide, 2 tall
- **Trucks** — 5 wide, 3 tall, slow
- **Smooth** — continuous movement (0.3 cells/step)
- **Animal** — smooth + free-roam cars
- **Scatter/Roam** — cars on random rows, not fixed lanes
- **Chaos** — 6 lanes, 4 cars, fast

## Key Learnings

1. **V-JEPA generalizes, DQN memorizes** — V-JEPA works across different game configs because it learns abstract representations; DQN is tied to its training layout
2. **256-dim bottleneck limits spatial precision** — position probe has ~0.5 cell error, too imprecise for fine-grained car avoidance
3. **Slot attention masks don't specialize at 7x7 resolution** — too coarse for 8 slots to compete for individual objects
4. **Occupancy auxiliary loss is critical** — without it, the encoder doesn't represent car positions
5. **GPU/CPU torch mismatch** — models trained on GPU produce slightly different latents on CPU; probes must be retrained locally
6. **Action hysteresis prevents jitter** — only switch actions when the new one is significantly better
7. **Dreamer fails when dynamics can't simulate collisions** — the dream is too optimistic, actor collapses to always-UP
8. **End-to-end RL (DQN) outperforms world-model planning** for this task, but doesn't generalize
