# 🐍 RL-Snake: Reinforcement Learning "Hello World"

## Project Overview

This project implements a **rigorous comparative study** of reinforcement learning algorithms applied to the Snake game, formulated as a **complete Markov Decision Process (MDP)**. This project uses **full grid state representation** to create a MDP, enabling direct comparison of classical and modern RL approaches across varying problem complexities.

**Key Innovation:** By using complete state observation (3-channel grid representation), we can:

1. Demonstrate the **curse of dimensionality** empirically
2. Show where tabular methods fail and why
3. Validate the necessity of function approximation
4. Compare value-based (DQN) vs. policy-based (PPO) methods

## 🎯 Project Goals

### Academic Goals

- **Rigorous MDP Formulation:** Complete mathematical specification satisfying the Markov property
- **Empirical Analysis:** Demonstrate theoretical limits through systematic experimentation
- **Algorithm Comparison:** Direct performance comparison across multiple RL paradigms
- **Scalability Study:** Show how different methods handle increasing state space complexity

### Educational Goals

- Understand the curse of dimensionality through hands-on experience
- See why function approximation becomes necessary
- Compare classical RL (Q-Learning) with modern deep RL (DQN, PPO)
- Learn implementation best practices for each algorithm type

## 📚 Documentation Structure

### Core MDP Formulation

**[→ Complete MDP Formulation](docs/mdp.md)

This document provides the mathematical foundation:

- **State Space ($\mathcal{S}$):** 3-channel grid representation (Head, Body, Food)
- **Action Space ($\mathcal{A}$):** Four absolute directions (UP, RIGHT, DOWN, LEFT)
- **Transition Function ($\mathcal{P}$):** Deterministic dynamics with explicit rules
- **Reward Function ($\mathcal{R}$):** +10 food, -10 collision, 0 survival
- **Discount Factor ($\gamma$):** 0.99 for long-term planning
- **Markov Property Proof:** Formal verification of MDP properties
- **Complexity Analysis:** State space growth and feasibility analysis

### Algorithm Documentation

#### 1. Tabular Q-Learning (Classical RL)

**[→ Tabular Q-Learning Documentation](docs/tabular_q_learning.md)

A classical value-based method using lookup tables.

**Scope:**

- Implementation for small grids (5×5, 7×7)
- State hashing strategy
- Expected behavior and limitations
- Empirical demonstration of curse of dimensionality

**Key Learning Points:**

- How Q-Learning works in discrete state spaces
- Why tabular methods fail on larger problems
- Memory and sample complexity in practice

---

#### 2. Deep Q-Network with CNN (Modern Value-Based RL)

**[→ DQN-CNN Documentation](docs/dqn_cnn.md)

A modern value-based method using convolutional neural networks.

**Scope:**

- CNN architecture for grid state inputs
- Experience replay and target networks
- Implementation for medium/large grids (10×10, 20×20)
- Comparison with tabular methods

**Key Learning Points:**

- Why CNNs work well for spatial data
- How function approximation enables generalization
- DQN's three key innovations (replay, target network, CNN)

---

#### 3. Proximal Policy Optimization (Modern Policy-Based RL)

**[→ PPO Documentation](docs/ppo.md)

A state-of-the-art policy gradient method.

**Scope:**

- Actor-Critic architecture with shared CNN backbone
- PPO clipped objective
- Implementation for all grid sizes
- Comparison with value-based methods (DQN)

**Key Learning Points:**

- Policy gradient methods vs. value-based methods
- Why PPO is more stable and sample-efficient
- Actor-Critic architecture benefits

## 🧪 Experimental Design

### Phase 1: Small Grid (5×5)

**Purpose:** Baseline - all algorithms can learn

| Algorithm          | Expected Performance | Training Time | Key Insight                                   |
| ------------------ | -------------------- | ------------- | --------------------------------------------- |
| Tabular Q-Learning | ✅ Excellent         | ~1 min        | Works perfectly on small spaces               |
| DQN-CNN            | ✅ Good              | ~5 min        | Slight overkill but demonstrates architecture |
| PPO                | ✅ Good              | ~5 min        | Baseline for policy methods                   |

**Outcome:** "Even simple tabular methods work when state space is manageable."

---

### Phase 2: Medium Grid (10×10)

**Purpose:** Show where tabular methods break down

| Algorithm          | Expected Performance | Training Time           | Key Insight                             |
| ------------------ | -------------------- | ----------------------- | --------------------------------------- |
| Tabular Q-Learning | ❌ Poor/Fails        | Hours (never converges) | **Curse of dimensionality in action**   |
| DQN-CNN            | ✅ Good              | ~15 min                 | Function approximation enables learning |
| PPO                | ✅ Better            | ~15 min                 | More sample-efficient than DQN          |

**Outcome:** "State space explosion (~10⁹ states) makes tabular methods impractical."

---

### Phase 3: Large Grid (20×20)

**Purpose:** Validate modern methods at scale

| Algorithm          | Expected Performance | Training Time | Key Insight                          |
| ------------------ | -------------------- | ------------- | ------------------------------------ |
| Tabular Q-Learning | N/A                  | Not attempted | Completely infeasible (~10⁴⁰ states) |
| DQN-CNN            | ✅ Moderate          | ~30 min       | Works but may show instability       |
| PPO                | ✅ Good              | ~25 min       | More stable than DQN                 |

**Outcome:** "Modern deep RL scales to complex problems that tabular methods cannot touch."

---

## 📊 Evaluation Metrics

### Performance Metrics

1. **Average Score:** Mean number of apples eaten per episode
2. **Max Score:** Best performance achieved
3. **Success Rate:** Percentage of episodes reaching score ≥ 5
4. **Steps to Food:** Efficiency metric (lower is better)

### Learning Metrics

1. **Learning Curve:** Score vs. episode number
2. **Sample Efficiency:** Episodes needed to reach threshold performance
3. **Convergence Stability:** Variance in late-training performance

### Computational Metrics

1. **Training Time:** Wall-clock time to convergence
2. **Memory Usage:** RAM consumption
3. **States Visited:** (Tabular only) Unique states encountered
4. **Parameter Count:** Network size (DQN, PPO)

## 🔬 Key Research Questions

1. **At what grid size does tabular Q-Learning become impractical?**

   - Hypothesis: 7×7 is marginal, 10×10 fails

2. **How much better is function approximation?**

   - Compare sample efficiency and final performance

3. **Do CNNs provide meaningful advantages over fully-connected networks?**

   - Ablation study: CNN vs. FC with same parameter count

4. **Is PPO superior to DQN for this problem?**

   - Compare stability, sample efficiency, and final performance

5. **How do algorithms transfer across grid sizes?**
   - Train on 10×10, test on 12×12 or 15×15

### Suitable For

- ✅ Reinforcement Learning course projects
- ✅ Machine Learning exam demonstrations
- ✅ Comparative algorithm studies
- ✅ Deep Learning practical assignments
- ✅ Research methodology courses (experimental design)

## 🚀 Getting Started

🚧 **TODO:** Installation and setup instructions

```bash
# Coming soon:
# 1. Environment setup with uv/pip
# 2. Running individual experiments
# 3. Reproducing results
# 4. Hyperparameter configuration
```

## 📈 Expected Results

### Performance Comparison (Predicted)

**Average Score after Training:**

| Grid Size | Tabular            | DQN-CNN      | PPO          |
| --------- | ------------------ | ------------ | ------------ |
| 5×5       | 8-12 apples        | 10-15 apples | 10-15 apples |
| 10×10     | 1-3 apples (fails) | 15-25 apples | 20-30 apples |
| 20×20     | N/A                | 25-40 apples | 30-50 apples |

**Training Time (to convergence):**

| Grid Size | Tabular | DQN-CNN | PPO    |
| --------- | ------- | ------- | ------ |
| 5×5       | 1 min   | 5 min   | 5 min  |
| 10×10     | Never   | 15 min  | 15 min |
| 20×20     | N/A     | 30 min  | 25 min |

### Learning Curves

We expect to see:

1. **Tabular (5×5):** Smooth improvement, convergence by episode 1000
2. **Tabular (10×10):** Flat/noisy, no meaningful improvement
3. **DQN (10×10):** Initial noise, then steady improvement
4. **PPO (10×10):** More stable than DQN, potentially faster convergence
5. **DQN/PPO (20×20):** Slower but steady improvement