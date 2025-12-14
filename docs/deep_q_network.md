# Deep Q-Network (DQN): From Tables to Neural Networks

## Overview

Deep Q-Network (DQN) is a breakthrough algorithm that combines **Q-Learning** with **Deep Neural Networks** to solve reinforcement learning problems with large or continuous state spaces. Introduced by DeepMind in 2015, it was the first algorithm to achieve human-level performance on Atari games using only raw pixel inputs.

Unlike Tabular Q-Learning which stores Q-values in a lookup table, DQN uses a neural network as a **function approximator** to estimate Q-values for any state.

## 1. The Core Problem: Why Not Just Use Q-Learning?

### The Curse of Dimensionality

Tabular Q-Learning works great for our compressed 11-sensor state space (2048 states), but consider:

| State Representation | State Space Size | Q-Table Memory |
|---------------------|------------------|----------------|
| 11 binary sensors | 2^11 = 2,048 | 6 KB ✅ |
| 20 binary sensors | 2^20 = 1,048,576 | 3 MB ✅ |
| 30 binary sensors | 2^30 = 1,073,741,824 | 3 GB ❌ |
| 20×20 raw grid | (400)^100 ≈ ∞ | Impossible ❌ |
| Raw pixels (84×84×4) | (256)^28,224 ≈ ∞ | Impossible ❌ |

**The Problem:** Real-world problems have massive state spaces that cannot fit in memory as tables.

**The Solution:** Use a neural network to **approximate** the Q-function instead of storing it.

## 2. The Core Concept: Function Approximation

Instead of storing Q-values in a table:

```
Q_table[state][action] = value
```

We use a neural network to compute them:

```
Q_network(state) = [Q(s, a₀), Q(s, a₁), Q(s, a₂)]
```

### Mathematical Formulation

**Tabular Q-Learning:**
$$Q: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$
(A lookup table mapping state-action pairs to values)

**Deep Q-Learning:**
$$Q_\theta: \mathcal{S} \rightarrow \mathbb{R}^{|\mathcal{A}|}$$
(A neural network with parameters $\theta$ that maps states to Q-values for all actions)

## 3. Neural Network Architecture

Our DQN uses a simple fully-connected architecture:

```
Input Layer (11 neurons)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (128 neurons, ReLU)
    ↓
Output Layer (3 neurons, Linear)
    ↓
Q-values: [Q(s, Straight), Q(s, Right), Q(s, Left)]
```

### Layer-by-Layer Breakdown

**Input Layer (11 neurons):**
- Same 11 boolean sensors as Q-Learning
- Values: [0 or 1] for each sensor
- Example: `[0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]`

**Hidden Layer 1 (128 neurons with ReLU):**
```python
h1 = ReLU(W1 @ input + b1)
# W1 shape: (128, 11)
# b1 shape: (128,)
```
- Learns abstract features from raw sensors
- ReLU activation: `max(0, x)` introduces non-linearity
- 128 neurons chosen empirically (balances capacity and speed)

**Hidden Layer 2 (128 neurons with ReLU):**
```python
h2 = ReLU(W2 @ h1 + b2)
# W2 shape: (128, 128)
# b2 shape: (128,)
```
- Combines features from first layer
- Learns higher-level patterns
- Same size as layer 1 (common practice)

**Output Layer (3 neurons, no activation):**
```python
output = W3 @ h2 + b3
# W3 shape: (3, 128)
# b3 shape: (3,)
# output: [0.45, 0.82, 0.31]  (Q-values for each action)
```
- One neuron per action
- No activation (we want raw Q-values, not probabilities)
- Values can be positive or negative

### Total Parameters

```
W1: 11 × 128 = 1,408
b1: 128
W2: 128 × 128 = 16,384
b2: 128
W3: 128 × 3 = 384
b3: 3
───────────────────
Total: 18,435 parameters
```

## 4. The Three Key Innovations

DQN introduced three critical techniques to make neural Q-Learning stable:

### Innovation #1: Experience Replay

**The Problem:** In Q-Learning, we train immediately on each new experience:
```
See state → Take action → Get reward → Update immediately → Repeat
```

This causes:
- **High correlation:** Consecutive states are similar, leading to overfitting
- **Catastrophic forgetting:** New experiences overwrite old knowledge
- **Sample inefficiency:** Each experience used only once

**The Solution:** Store experiences in a buffer and sample randomly:

```python
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=64):
        return random.sample(self.buffer, batch_size)
```

**Benefits:**
- ✅ Breaks temporal correlation
- ✅ Each experience can be used multiple times
- ✅ More stable learning (variance reduction)

### Innovation #2: Target Network

**The Problem:** When training, we update Q-values using a target:
$$\text{Target} = r + \gamma \cdot \max_{a'} Q_\theta(s', a')$$

But $Q_\theta$ is the network we're training! This creates a "moving target" problem:
- We update the network
- The target changes
- We update again to match the new target
- The target changes again
- **Result:** Unstable, oscillating learning

**The Solution:** Use a **separate target network** ($Q_{\theta^-}$) that updates slowly:

```python
# Q-Network: Updated every step
self.q_network = QNetwork()

# Target Network: Updated every 1000 steps
self.target_network = QNetwork()
self.target_network.load_state_dict(self.q_network.state_dict())

# Training loop
if steps % 1000 == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```

**Training with Target Network:**
$$\text{Target} = r + \gamma \cdot \max_{a'} Q_{\theta^-}(s', a')$$
$$\text{Loss} = \left( Q_\theta(s, a) - \text{Target} \right)^2$$

**Benefits:**
- ✅ Stable targets during training
- ✅ Prevents oscillations
- ✅ Better convergence

### Innovation #3: Frame Skipping and Preprocessing (Not in our version)

**Note:** This is used in Atari games but not needed for our discrete Snake environment.

DeepMind's original DQN:
- Stacks 4 consecutive frames (to capture motion)
- Converts to grayscale
- Resizes to 84×84 pixels
- This reduces the input from raw pixels to manageable size

## 5. The DQN Update Algorithm

### Loss Function

DQN minimizes the **Temporal Difference (TD) Error** using Mean Squared Error:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( \underbrace{r + \gamma \cdot \max_{a'} Q_{\theta^-}(s', a')}_{\text{Target}} - \underbrace{Q_\theta(s, a)}_{\text{Prediction}} \right)^2 \right]$$

Where:
- $\mathcal{D}$: Replay buffer (experience distribution)
- $\theta$: Parameters of Q-Network
- $\theta^-$: Parameters of Target Network
- $\gamma$: Discount factor (0.9)

### Gradient Descent Update

```python
# Forward pass
current_q = q_network(states).gather(1, actions)

# Compute target (no gradient)
with torch.no_grad():
    next_q = target_network(next_states).max(1)[0]
    target_q = rewards + (1 - dones) * gamma * next_q

# Compute loss
loss = MSELoss(current_q, target_q)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Update parameters: θ ← θ - α∇Loss
```

### Full Training Loop

```python
Initialize Q-Network with random weights θ
Initialize Target Network θ⁻ = θ
Initialize Replay Buffer D (capacity 100k)
Set epsilon = 1.0

for episode in range(3000):
    state = reset_game()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(Q_θ(state))
        
        # Execute action
        next_state, reward, done = game.step(action)
        
        # Store transition in replay buffer
        D.push(state, action, reward, next_state, done)
        
        # Train if buffer has enough samples
        if len(D) >= batch_size:
            # Sample random mini-batch
            batch = D.sample(batch_size)
            
            # Compute loss and update
            loss = compute_loss(batch, Q_θ, Q_θ⁻)
            θ = θ - α∇loss
        
        # Periodically update target network
        if steps % 1000 == 0:
            θ⁻ = θ
        
        state = next_state
        steps += 1
    
    # Decay exploration
    epsilon = max(epsilon * 0.995, 0.01)
```

## 6. Hyperparameters

These values control the DQN learning behavior:

| Parameter | Symbol | Value | Explanation |
|-----------|--------|-------|-------------|
| **Learning Rate** | $\alpha$ | `0.001` | Much lower than Q-Learning (0.1) because neural networks need gentle updates |
| **Discount Factor** | $\gamma$ | `0.9` | Same as Q-Learning - importance of future rewards |
| **Epsilon Start** | $\epsilon$ | `1.0` | Start with 100% exploration |
| **Epsilon Decay** | - | `0.995` | Gradual transition to exploitation |
| **Min Epsilon** | $\epsilon_{min}$ | `0.01` | Always maintain 1% exploration |
| **Batch Size** | - | `64` | Number of experiences to sample per update |
| **Buffer Size** | - | `100,000` | Maximum stored experiences |
| **Target Update Freq** | - | `1000` | Steps between target network updates |
| **Hidden Layer Size** | - | `128` | Neurons per hidden layer |

### Why These Values?

**Learning Rate (0.001 vs 0.1 in Q-Learning):**
- Neural networks require smaller steps
- Too high → Unstable training, divergence
- Too low → Extremely slow learning
- Adam optimizer helps adapt learning rate

**Batch Size (64):**
- Smaller: More updates, faster learning, higher variance
- Larger: Fewer updates, slower learning, lower variance
- 64 is a good balance (powers of 2 are GPU-efficient)

**Buffer Size (100,000):**
- Larger: More diverse experiences, better learning
- Smaller: Less memory, faster sampling
- 100k provides ~1000 episodes of history

**Target Update Frequency (1000):**
- Too frequent: Defeats the purpose of target network
- Too infrequent: Target becomes stale
- 1000 steps ≈ 10-20 episodes