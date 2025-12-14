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
uv run main.py --agent_type q_learning --show-plots
```

### Command Line Options

- `--ui [pygame|terminal]` - Choose UI renderer (default: pygame)
- `--agent_type [q_learning]` - Choose agent type (default: q_learning)
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