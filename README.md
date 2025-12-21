# 🐍 RL-Snake: Reinforcement Learning "Hello World"

## My Introduction to Deep Reinforcement Learning

This is my **"Hello World" project for Reinforcement Learning** – a complete implementation of Deep Q-Network (DQN) applied to the classic Snake game. As an AI student, I built this to deeply understand how modern RL algorithms work in practice, moving beyond theory into hands-on implementation.

Instead of printing "Hello World" to a console, my RL agent learns to navigate a 10×10 grid, collect food, and avoid obstacles – all through trial and error with neural networks.

## 🎯 Project Goals

### Technical Understanding

- Implement a complete **Markov Decision Process (MDP)** formulation
- Build a **CNN-based DQN** that generalizes across states
- Master the three key DQN innovations: **replay buffer, target network, double DQN**
- Understand training dynamics through empirical observation

### Practical Experience

- Train an agent that learns intelligent behavior (not hand-coded rules)
- Debug common RL issues (instability, poor exploration, slow learning)
- Visualize and interpret what neural networks learn
- Achieve satisfying results on a non-trivial problem

## 📚 Documentation

I've written detailed documentation to explain every aspect:

### Core Formulation

**[→ Complete MDP Formulation](docs/mdp.md)**

The mathematical foundation – how I formalized Snake as an MDP:

- **State Space:** 4-channel grid representation (Head, Body gradient, Food, Time)
- **Action Space:** Four directional moves with collision handling
- **Reward Function:** Sparse rewards with milestone bonuses
- **Markov Property:** Why this state representation is complete

### Algorithm Deep-Dive

**[→ Deep Q-Network Documentation](docs/dqn_cnn.md)**

Everything about my DQN implementation:

- **Architecture:** Why I chose this CNN design (~1.7M parameters)
- **Training Process:** The four phases from random to mastery
- **Hyperparameters:** How I tuned them and why they matter
- **Common Issues:** Problems I encountered and how I solved them

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (I use this for fast dependency management)

### Installation

Clone and set up the environment:

```bash
git clone https://github.com/MarinCervinschi/rl-snake.git
cd rl-snake
uv sync
```

That's it! `uv sync` creates a virtual environment and installs everything you need.

### Training the Agent

The training notebook contains **all the information and instructions** you need, including explanations of each step.

#### Option 1: Local Training (CPU)

Open the [dqn_agent.ipynb](dqn_agent.ipynb) notebook and follow along.

#### Option 2: Google Colab (GPU) - Recommended

For faster training with GPU acceleration:

1. Open `dqn_agent.ipynb` in Google Colab
2. The notebook includes instructions for Colab setup
3. Follow the step-by-step cells – everything is explained inside

**Note:** The notebook is self-contained with detailed markdown cells explaining each code block.

### Watch Your Agent Play

After training, see your agent in action:

```bash
uv run play.py
```

**Options:**

```bash
# Play multiple episodes
uv run play.py --episodes 10

# Adjust speed (lower = faster)
uv run play.py --speed 0.05

# Use a custom model
uv run play.py --model_path models/my_model.pkl
```

Close the window or press Ctrl+C to stop.

## 📊 What to Expect

### Training Results (10×10 Grid)

| Metric                      | Value        |
| --------------------------- | ------------ |
| **Episodes to Convergence** | 5,000-10,000 |
| **Final Average Score**     | 18-25 apples |
| **Best Score**              | 30-40 apples |

## 🎓 What I Learned

### Theoretical Insights

1. **The Curse of Dimensionality is Real:**  
   10×10 grid = ~10^10 states. Tabular methods need 320 GB just to store the Q-table!

2. **Function Approximation is Powerful:**  
   My network uses only 1.7M parameters (~7 MB) and generalizes across all states.

3. **Experience Replay Works:**  
   Random sampling breaks correlation and dramatically improves learning stability.

4. **Target Networks Matter:**  
   Without them, training oscillates or diverges. They provide stable TD targets.

5. **CNNs Understand Space:**  
   Convolutional layers naturally extract spatial features – "food right", "wall ahead", etc.

### Practical Lessons

1. **Hyperparameters are Crucial:**  
   Small changes in learning rate or epsilon decay significantly affect convergence.

2. **Reward Shaping Helps:**  
   Milestone bonuses accelerate learning by providing intermediate goals.

3. **Training Takes Time:**  
   Deep RL needs thousands of episodes. Patience and monitoring are essential.

4. **Debugging is Non-trivial:**  
   RL failures are subtle – agent might not crash but never improve.

5. **Visualization Helps:**  
   Watching the agent play reveals what it learned (and didn't learn).

## 💡 Why This Project Matters (To Me)

This isn't just a homework assignment – it's my foundation for understanding deep RL:

- **Hands-on Learning:** Theory is great, but implementation reveals the details
- **Complete Pipeline:** From MDP formulation to trained agent
- **Debugging Experience:** I encountered and solved real RL problems
- **Foundation for More:** Now I can understand papers on A3C, PPO, Rainbow, etc.

**Most importantly:** I can now explain DQN to others because I built it from scratch.
