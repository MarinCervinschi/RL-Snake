# 🐍 RL-Snake: Reinforcement Learning "Hello World"

## Project Overview

This project is a modular implementation of a **Reinforcement Learning (RL) Agent** that learns to play the classic game "Snake" from scratch.

## Getting Started

1. Set up environment:

```bash
uv sync
```

2. Run the training loop:

```bash
# Default: PyGame UI and Q-Learning agent
uv run main.py

# Or use terminal UI
uv run main.py --ui terminal

# Customize agent type and show plots
uv run main.py --agent_type dqn --show-plots
```

### ⚠️ note on `dqn` agent:

The first time you run the DQN agent, it trains for 3000 episodes and saves the model to `dqn_model.pth`. This operation take ~10 minutes. Suggest to update the `RENDER_INTERVAL` in `config.py` to a large number to avoid rendering during this initial training. Subsequent runs will load the saved model for evaluation. Then update `RENDER_INTERVAL` back to a smaller number for visual feedback.

### Command Line Options

- `--ui [pygame|terminal]` - Choose UI renderer (default: pygame)
- `--agent_type [q_learning|dqn]` - Choose agent type (default: q_learning)
- `--show-plots/--no-show-plots` - Show training metrics plots after training (default: no-show-plots)

### Rendering configuration

Settings for rendering can be adjusted in [config.py](config.py):

- Modify `RENDER_SPEED` to adjust the speed of rendering when using PyGame UI. (default: 0.1 seconds per frame)
- Modify `RENDER_INTERVAL` to adjust how often the game is visually rendered during training. (default: every 100 episodes)

### UI Options

- **PyGame UI (Default):** Modern graphical interface with grid visualization
- **Terminal UI:** Text-based visualization in console

Initially, the snake will move randomly. Over time, it will learn to navigate the grid, avoid collisions, and seek out food.

## AI Architectures (Reinforcement Learning)

This project implements two different RL approaches to demonstrate the evolution from classical to modern deep learning methods:

### 1. Tabular Q-Learning (Classical RL)

A lightweight, interpretable algorithm that uses a mathematical lookup table to store the value of every state-action pair. Perfect for small state spaces and understanding RL fundamentals.

**Key Features:**

- 📊 Q-Table: 2048 × 3 matrix (states × actions)
- ⚡ Fast training: Converges in ~1000 episodes
- 🔍 Interpretable: Can inspect exact Q-values
- 💾 Minimal memory: ~6KB

**[→ Full Documentation: Tabular Q-Learning](docs/tabular_q_learning.md)**

### 2. Deep Q-Network (DQN) (Modern Deep RL)

A neural network-based approach that approximates Q-values through function approximation. Scales to large state spaces and is the foundation of modern RL systems.

**Key Features:**

- 🧠 Neural Network: 11 → 128 → 128 → 3 architecture
- 🎯 Experience Replay: Learns from past experiences
- 🎚️ Target Network: Stabilizes training
- 📈 Scalable: Handles millions of states

**[→ Full Documentation: Deep Q-Network (DQN)](docs/deep_q_network.md)**

## Comparison: Q-Learning vs DQN

### Memory Comparison

**Q-Learning:**

```python
q_table = np.zeros((2048, 3))
# Memory: 2048 × 3 × 8 bytes (float64) = 49,152 bytes ≈ 48 KB
```

**DQN:**

```python
q_network = QNetwork()  # 18,435 parameters
replay_buffer = ReplayBuffer(100_000)  # 100k experiences
# Network: 18,435 × 4 bytes (float32) ≈ 74 KB
# Buffer: 100,000 × ~50 bytes ≈ 5 MB
# Total: ~5 MB
```

### Speed Comparison

**Q-Learning:**

- Update time: **0.0001 ms** (array lookup + arithmetic)
- Training 1000 episodes: ~0.02 minutes (on CPU)

**DQN:**

- Update time: **1-2 ms** (forward pass + backward pass)
- Training 3000 episodes: ~10 minutes (on CPU)

### Performance Comparison

After equivalent training:

| Metric        | Q-Learning (1000 ep) | DQN (3000 ep)                   |
| ------------- | -------------------- | ------------------------------- |
| Average Score | TODO apples          | TODO apples                     |
| Max Score     | TODO apples          | TODO apples                     |
| Consistency   | Moderate variance    | Lower variance (when converged) |
| Pathfinding   | Good                 | Excellent                       |
