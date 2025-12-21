# Deep Q-Network (DQN) with CNN for Snake Game

> **Prerequisites:** 
> 1. Read the [Complete MDP Formulation](mdp.md) to understand the problem structure
> 2. Understand basic reinforcement learning concepts (Q-Learning, value functions)

## Overview

**Deep Q-Network (DQN)** replaces traditional Q-tables with a **neural network** that can generalize across similar states. Instead of storing Q-values for every possible state (impossible for our 10×10 grid with ~10^10 states), DQN learns a function $Q_\theta(s, a)$ that estimates Q-values for any state.

**Key Innovation:** By using a **Convolutional Neural Network (CNN)**, DQN exploits the spatial structure of the grid to learn meaningful patterns and strategies.

**Our Implementation:** We use DQN with experience replay and target networks to train an agent that can navigate the Snake game efficiently on a 10×10 grid.

---

## Why Neural Networks for Snake?

### The Problem with Tables

**Tabular Q-Learning** requires storing a Q-value for every state-action pair:

```
State 1: [Q(s₁,UP), Q(s₁,RIGHT), Q(s₁,DOWN), Q(s₁,LEFT)]
State 2: [Q(s₂,UP), Q(s₂,RIGHT), Q(s₂,DOWN), Q(s₂,LEFT)]
...
State 10,000,000,000: [Q(s_n,UP), Q(s_n,RIGHT), Q(s_n,DOWN), Q(s_n,LEFT)]
```

**Problems:**
- **Memory explosion:** 10×10 grid ≈ 10^10 states × 4 actions × 8 bytes = 320 GB
- **Sample inefficiency:** Must visit each state multiple times
- **No generalization:** Learning about state A doesn't help with similar state B

### The Neural Network Solution

**DQN** uses a single network with shared parameters:

```
Any State → CNN → [Q(s,UP), Q(s,RIGHT), Q(s,DOWN), Q(s,LEFT)]
```

**Advantages:**
- ✅ **Compact:** ~3.3M parameters ≈ 13 MB
- ✅ **Generalizes:** Similar states produce similar Q-values
- ✅ **Sample efficient:** One update affects all similar states
- ✅ **Scalable:** Same network works for 10×10, 20×20, or larger grids

---

## CNN Architecture

### Input: 4-Channel Grid State

Our state representation is a 4-channel grid of size (H, W, 4):

**Channel 0 - Snake Head:**
```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  ← Head at (5,2)
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
...
```

**Channel 1 - Snake Body (Gradient):**
```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]  ← Neck (1.0)
[0, 0, 0, 0, 0.67, 0, 0, 0, 0, 0] ← Middle
[0, 0, 0, 0, 0.33, 0, 0, 0, 0, 0] ← Tail (approaching 0)
...
```

**Channel 2 - Food:**
```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  ← Food at (8,1)
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
...
```

**Channel 3 - Time (Global Urgency):**
```
[0.4, 0.4, 0.4, 0.4, 0.4, ...]  ← All cells = 0.4
[0.4, 0.4, 0.4, 0.4, 0.4, ...]  ← (40% toward timeout)
[0.4, 0.4, 0.4, 0.4, 0.4, ...]
...
```

### Network Architecture

```
Input: (4, 10, 10) tensor
         ↓
    Conv2D(4→32, kernel=3×3, padding=1)
         ↓
      ReLU Activation
         ↓
    Conv2D(32→64, kernel=3×3, padding=1)
         ↓
      ReLU Activation
         ↓
    Conv2D(64→64, kernel=3×3, padding=1)
         ↓
      ReLU Activation
         ↓
    Flatten (64 × 10 × 10 = 6,400)
         ↓
    Linear (6,400 → 256)
         ↓
      ReLU Activation
         ↓
    Linear (256 → 4)
         ↓
    Output: [Q(UP), Q(RIGHT), Q(DOWN), Q(LEFT)]
```

### Layer-by-Layer Explanation

**Convolutional Layers (3 layers):**
- **Purpose:** Extract spatial features from the grid
- **Kernel size 3×3:** Looks at 3×3 neighborhoods
- **Padding=1:** Preserves spatial dimensions (output same size as input)
- **Layer 1 (4→32):** Learns basic patterns ("food nearby", "wall ahead")
- **Layer 2 (32→64):** Combines basic features into mid-level concepts
- **Layer 3 (64→64):** High-level strategic features

**Fully Connected Layers (2 layers):**
- **FC1 (6,400→256):** Aggregates spatial information into abstract features
- **FC2 (256→4):** Maps to Q-values for each action

**Activation Functions:**
- **ReLU:** $f(x) = \max(0, x)$ - Introduces non-linearity
- **No activation on output:** Q-values can be positive or negative

### Parameter Count

For a 10×10 grid:

```
Conv1: (3×3×4×32) + 32 bias = 1,184
Conv2: (3×3×32×64) + 64 bias = 18,496
Conv3: (3×3×64×64) + 64 bias = 36,928
FC1: (6,400×256) + 256 bias = 1,638,656
FC2: (256×4) + 4 bias = 1,028
────────────────────────────────────
Total: ~1.7 million parameters ≈ 6.8 MB
```

**Note:** Most parameters are in FC1, which connects the spatial features to the decision-making layer.

---

## The Three Key DQN Innovations

### 1. Experience Replay

**Problem:** Training on consecutive game frames leads to:
- High correlation (state at step t is very similar to step t+1)
- Catastrophic forgetting (new experiences overwrite old knowledge)
- Sample inefficiency (each experience used only once)

**Solution:** Store experiences in a replay buffer and sample randomly.

**Implementation:**
```python
# Replay Buffer: Fixed-size FIFO queue
buffer = deque(maxlen=200_000)

# During gameplay:
buffer.append((state, action, reward, next_state, done))

# During training:
batch = random.sample(buffer, batch_size=64)  # Random sampling
```

**Benefits:**
- ✅ Breaks temporal correlation (random sampling decorrelates data)
- ✅ Reuses experiences efficiently (each experience used multiple times)
- ✅ Stabilizes training (diverse batches reduce variance)

**Buffer Capacity:** 200,000 transitions ≈ 2,000-3,000 episodes worth of experience

### 2. Target Network

**Problem:** When computing the TD target, we use the same network we're training:

$$\text{Target} = r + \gamma \max_{a'} Q_\theta(s', a')$$

But $Q_\theta$ is constantly changing! This creates a **moving target problem:**
- Update Q-values → Targets change → Update again → Targets change more → Instability

**Solution:** Use a separate **target network** $Q_{\theta^-}$ that updates slowly.

**Implementation:**
```python
# Two identical networks
q_network = ConvQNetwork()        # Updated every training step
target_network = ConvQNetwork()   # Updated every 1,000 steps

# Initially identical
target_network.load_state_dict(q_network.state_dict())

# Training: use target network for stable targets
with torch.no_grad():
    next_q = target_network(next_state).max()
    target = reward + gamma * next_q

# Update target network periodically
if steps % 1000 == 0:
    target_network.load_state_dict(q_network.state_dict())
```

**Benefits:**
- ✅ Stable targets for 1,000 steps (no moving target)
- ✅ Prevents oscillations and divergence
- ✅ More reliable convergence

**Update Frequency:** Every 1,000 steps ≈ 10-20 episodes

### 3. Double DQN (Action Selection)

**Standard DQN Problem:** Can overestimate Q-values due to max operator.

**Double DQN Solution:** Use online network to **select** action, target network to **evaluate** it:

```python
# Standard DQN (overestimates):
next_q = target_network(next_state).max()

# Double DQN (more accurate):
next_action = q_network(next_state).argmax()  # Online selects
next_q = target_network(next_state)[next_action]  # Target evaluates
```

**Benefit:** Reduces overestimation bias, leading to more accurate Q-values.

---

## Training Algorithm

### Loss Function

DQN minimizes the **Mean Squared Bellman Error**:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{B}} \left[ \left( Q_\theta(s, a) - \left(r + \gamma (1-d) \max_{a'} Q_{\theta^-}(s', a')\right) \right)^2 \right]$$

Where:
- $\mathcal{B}$: Replay buffer (uniformly sampled)
- $\theta$: Q-network parameters
- $\theta^-$: Target network parameters (frozen during optimization)
- $\gamma = 0.99$: Discount factor
- $d \in \{0,1\}$: Done flag (1 if terminal state)

**Intuition:** Make predicted Q-value match the observed reward + estimated future value.

**Loss Implementation:**
```python
# Current Q-values
current_q = q_network(states).gather(1, actions.unsqueeze(1))

# Target Q-values (no gradient)
with torch.no_grad():
    next_actions = q_network(next_states).argmax(1)  # Double DQN
    next_q = target_network(next_states).gather(1, next_actions.unsqueeze(1))
    target_q = rewards + (1 - dones) * gamma * next_q

# Smooth L1 Loss (Huber loss)
loss = nn.SmoothL1Loss()(current_q, target_q)
```

**Smooth L1 Loss:** More robust to outliers than MSE:
$$\text{SmoothL1}(x) = \begin{cases}
0.5x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}$$

### Training Loop

```
Initialize:
    Q-network θ with random weights
    Target network θ⁻ = θ
    Replay buffer B (capacity 200k)
    ε = 1.0 (exploration rate)

For episode = 1 to 20,000:
    state = reset_game()
    episode_reward = 0
    
    While not done:
        # 1. Action Selection (ε-greedy)
        if random() < ε:
            action = random_action()
        else:
            action = argmax(Q_θ(state))
        
        # 2. Environment Step
        next_state, reward, done = env.step(action)
        
        # 3. Store Transition
        B.push(state, action, reward, next_state, done)
        
        # 4. Training (if enough samples)
        if len(B) >= batch_size:
            batch = B.sample(batch_size=64)
            
            # Compute loss and update
            loss = compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(θ, max_norm=1.0)
            optimizer.step()
        
        # 5. Update Target Network
        if steps % 1000 == 0:
            θ⁻ ← θ
        
        state = next_state
        steps += 1
    
    # 6. Decay Exploration
    ε = max(ε_min, ε * decay_rate)
```

---

## Hyperparameters

### Core Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Grid Size** | 10×10 | Optimal balance: complex enough to be interesting, small enough to train quickly |
| **Learning Rate** | 0.0001 | Low rate for stable neural network training (Adam adapts this) |
| **Discount Factor** | 0.99 | Long-term planning (effective horizon ≈ 460 steps) |
| **Batch Size** | 64 (CPU) / 128 (GPU) | Balance between gradient quality and update frequency |
| **Replay Buffer** | 200,000 | Stores 2,000-3,000 episodes worth of diverse experiences |
| **Target Update** | 1,000 steps | Sync networks every 10-20 episodes for stable targets |
| **Optimizer** | AdamW | Adaptive learning rate with weight decay |

### Exploration Strategy

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Initial ε** | 1.0 | Start with 100% random exploration |
| **Min ε** | 0.01 | Always maintain 1% exploration (avoid getting stuck) |
| **Decay** | Linear over 18,000 episodes | Gradual shift from exploration to exploitation |
| **Learning Starts** | 20,000 steps | Pure random exploration for buffer warmup |

**ε Schedule:**
```python
# First 20,000 steps: pure random (warm up buffer)
if steps < 20_000:
    ε = 1.0

# Linear decay over 90% of training
progress = min(1.0, episodes / (0.9 * total_episodes))
ε = max(0.01, 1.0 - progress * 0.99)
```

### Training Duration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Total Episodes** | 20,000 | Sufficient for convergence on 10×10 grid |
| **Expected Convergence** | 5,000-10,000 | When average score plateaus |
| **Training Time (CPU)** | 30-60 minutes | Depends on CPU speed |
| **Training Time (GPU)** | 10-20 minutes | GPU accelerates forward/backward passes |

---

## Training Dynamics

### Phase 1: Warmup (Episodes 1-1,000)

**Behavior:**
- Pure random exploration
- Filling replay buffer with diverse experiences
- Network sees all parts of state space

**Metrics:**
- Average score: 1-3 apples
- ε = 1.0 (100% random)
- Loss: High and noisy

**What's Happening:**
- Network learning basic patterns ("don't hit walls immediately")
- Gradients large as network adjusts to data distribution

### Phase 2: Initial Learning (Episodes 1,000-5,000)

**Behavior:**
- ε decreasing (more exploitation)
- Agent starts following food
- Avoiding obvious collisions

**Metrics:**
- Average score: 5-10 apples
- ε: 1.0 → 0.6
- Loss: Decreasing, stabilizing

**What's Happening:**
- Network learning: "move toward food", "avoid walls", "don't trap yourself"
- Q-values becoming more accurate
- Policy improving rapidly

### Phase 3: Refinement (Episodes 5,000-15,000)

**Behavior:**
- Strategic navigation
- Planning around body segments
- Taking longer paths when necessary

**Metrics:**
- Average score: 12-20 apples
- ε: 0.6 → 0.1
- Loss: Low, stable

**What's Happening:**
- Network learning complex patterns
- Handling edge cases
- Fine-tuning Q-values

### Phase 4: Mastery (Episodes 15,000-20,000)

**Behavior:**
- Consistent high performance
- Efficient food collection
- Rare mistakes

**Metrics:**
- Average score: 18-25+ apples
- ε: 0.1 → 0.01
- Loss: Very stable

**What's Happening:**
- Near-optimal policy
- Minimal exploration
- Exploitation of learned strategy

---

## Understanding the Learned Features

### What CNNs Learn

**Convolutional Layer 1 (Basic Features):**
- Filter detecting "food in neighborhood"
- Filter detecting "wall within 2 cells"
- Filter detecting "body segment blocking path"
- Filter detecting "open space"

**Convolutional Layer 2 (Combinations):**
- "Clear path to food" (food nearby + no obstacles)
- "Dangerous corridor" (walls on both sides)
- "Body creating trap" (body forming closed loop)

**Convolutional Layer 3 (Strategic Patterns):**
- "Safe food collection" (can reach food and escape)
- "Risky maneuver" (tight space navigation)
- "Time pressure" (combining spatial + temporal features)

**Fully Connected Layers (Decision Making):**
- Aggregating all spatial information
- Weighing short-term vs. long-term rewards
- Outputting Q-values based on strategic assessment

### Example Q-Value Interpretation

Given this state:
```
Head: (5, 5)
Food: (8, 5)  [3 cells to the right]
Body: blocking below
Time: 0.3 (moderate urgency)
```

Q-Network might output:
```python
Q(UP) = 2.3      # Neutral, no immediate benefit
Q(RIGHT) = 8.5   # Highest! Moves toward food
Q(DOWN) = -5.2   # Negative! Would hit body
Q(LEFT) = 1.1    # Positive but not optimal
```

**Action Selected:** RIGHT (highest Q-value)

---

## Implementation Details

### Gradient Clipping

Prevents exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
```

If gradient norm exceeds 1.0, scale it down to 1.0.

### Weight Initialization

Kaiming initialization for ReLU networks:
```python
for layer in q_network.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
```

### Optimizer Configuration

AdamW with default parameters:
```python
optimizer = optim.AdamW(
    q_network.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.01
)
```

---

## Expected Performance

### Training Metrics (10×10 Grid)

| Metric | Value | When |
|--------|-------|------|
| **Episodes to Convergence** | 5,000-10,000 | When avg score plateaus |
| **Final Average Score** | 18-25 apples | Last 100 episodes |
| **Best Score** | 30-40 apples | Peak performance |
| **Training Time (CPU)** | 30-60 minutes | Typical desktop |
| **Training Time (GPU)** | 10-20 minutes | With CUDA |

### Learning Curve Characteristics

**Typical training curve:**
```
Score
  |
40|                                    *  *
  |                                 *        *
30|                             *                *
  |                          *
20|                      *
  |                  *
10|             *
  |         *
 0|  * * *
  |_________________________________________
    0     5k    10k   15k   20k  Episodes
```

**Key Observations:**
1. Slow start (first 1,000 episodes): Random exploration
2. Rapid improvement (1,000-5,000): Learning basic strategies
3. Steady growth (5,000-15,000): Refining policy
4. Plateau (15,000+): Near-optimal performance

---

## Common Issues and Solutions

### Issue 1: No Learning After 5,000 Episodes

**Symptoms:** Score stays around 1-3, no improvement

**Possible Causes:**
- Learning rate too high/low
- Reward signal too weak
- Network too small/large
- Exploration decaying too fast

**Solutions:**
- Check learning rate (try 1e-4 to 1e-5)
- Verify reward function is working
- Print Q-values to check if network is learning
- Slow down ε decay

### Issue 2: Performance Oscillates Wildly

**Symptoms:** Score jumps between 5 and 20 unpredictably

**Possible Causes:**
- Target network updating too frequently
- Batch size too small
- High learning rate

**Solutions:**
- Increase target update frequency (1,000 → 2,000 steps)
- Increase batch size (64 → 128)
- Reduce learning rate

### Issue 3: Training Very Slow

**Symptoms:** 10 minutes per 100 episodes

**Solutions:**
- Use GPU if available
- Reduce replay buffer size
- Simplify network architecture
- Increase batch size

### Issue 4: Q-Values Explode to NaN

**Symptoms:** Loss becomes NaN, training crashes

**Possible Causes:**
- Learning rate too high
- No gradient clipping
- Reward scale too large

**Solutions:**
- Reduce learning rate
- Enable gradient clipping (max_norm=1.0)
- Normalize rewards

---

## Advantages and Limitations

### Advantages ✅

1. **Handles Large State Spaces:** Works where tabular methods fail (10×10, 20×20 grids)
2. **Generalizes:** Similar states produce similar Q-values without explicit storage
3. **Scalable Memory:** Same network size for any grid size
4. **Sample Efficient:** Learns from similar states through shared weights
5. **Spatial Understanding:** CNN naturally processes grid structure

### Limitations ❌

1. **Sample Inefficiency vs. Policy Gradient Methods:** Requires more episodes than PPO
2. **Hyperparameter Sensitive:** Requires careful tuning of learning rate, buffer size, etc.
3. **Training Time:** Minutes to hours vs. seconds for tabular (on small grids)
4. **No Convergence Guarantee:** Unlike tabular Q-Learning, can diverge
5. **Less Interpretable:** Cannot inspect exact Q-values, network is a black box
6. **Instability Risk:** Can suffer from catastrophic forgetting or divergence