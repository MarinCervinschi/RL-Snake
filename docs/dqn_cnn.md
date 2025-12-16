# Deep Q-Network (DQN) with CNN for Snake Game

> **Prerequisites:** 
> 1. Read the [Complete MDP Formulation](mdp_full_formulation.md) to understand the problem
> 2. Read [Tabular Q-Learning](tabular_q_learning.md) to understand Q-Learning basics and why tabular methods fail

## Overview

**Deep Q-Network (DQN)** is a breakthrough algorithm that replaces the Q-table with a **neural network**. Instead of storing Q-values for every state-action pair, DQN learns a function $Q_\theta(s, a)$ that approximates Q-values for any state, including those never seen before.

**Key Innovation:** By using **function approximation**, DQN can generalize across similar states and handle state spaces that are too large for tabular methods.

**For Snake:** We use a **Convolutional Neural Network (CNN)** architecture to exploit the spatial structure of the grid state.

## From Tables to Functions

### The Fundamental Shift

**Tabular Q-Learning:**
```
State 42 → Look up row 42 in table → Get [Q(s,UP), Q(s,RIGHT), Q(s,DOWN), Q(s,LEFT)]
```
- Each state has its own independent entry
- No relationship between similar states
- Memory: O(|S| × |A|)

**Deep Q-Learning:**
```
State 42 → Pass through neural network → Compute [Q(s,UP), Q(s,RIGHT), Q(s,DOWN), Q(s,LEFT)]
```
- All states share the same network weights
- Similar states produce similar Q-values (generalization)
- Memory: O(parameters) ~ constant

### Mathematical Formulation

**Tabular:** $Q: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ (lookup table)

**DQN:** $Q_\theta: \mathcal{S} \rightarrow \mathbb{R}^{|\mathcal{A}|}$ (neural network with parameters $\theta$)

The network takes a state as input and outputs Q-values for all actions simultaneously.

## CNN Architecture for Grid States

### Why CNNs for Snake?

Grid-based games have **spatial structure**:
- Nearby cells are related (e.g., wall nearby means danger)
- Patterns repeat across the grid (e.g., "food to the right" looks similar regardless of absolute position)
- Local features matter (e.g., 3×3 neighborhood around head)

**Convolutional Neural Networks** are designed to exploit these properties through:
1. **Local receptive fields:** Each neuron looks at a small patch
2. **Weight sharing:** Same filters applied across the grid
3. **Translation invariance:** Pattern learned at position (5,5) transfers to (10,10)

### Network Architecture

Our DQN uses a 3-layer CNN followed by fully-connected layers:

```
Input: Grid State (3 × H × W)
         ↓
    Conv2D (3→32, kernel=3×3, padding=1)
         ↓
      Tanh Activation
         ↓
    Conv2D (32→64, kernel=3×3, padding=1)
         ↓
      Tanh Activation
         ↓
    Conv2D (64→64, kernel=3×3, padding=1)
         ↓
      Tanh Activation
         ↓
    Flatten (64 × H × W)
         ↓
    Linear (64×H×W → 512)
         ↓
      Tanh Activation
         ↓
    Linear (512 → 4)  [Output: Q-values]
```

### Layer-by-Layer Breakdown

**Input Layer: (3, H, W)**
- Channel 0: Snake head positions (1.0 at head, 0.0 elsewhere)
- Channel 1: Snake body positions (1.0 at body, 0.0 elsewhere)
- Channel 2: Food position (1.0 at food, 0.0 elsewhere)

**Conv Layer 1: 3 → 32 channels**
- Kernel size: 3×3 (looks at 3×3 neighborhoods)
- Padding: 1 (preserves spatial dimensions)
- Output: (32, H, W)
- **Learns:** Basic features like "head near food", "body nearby", "wall ahead"

**Conv Layer 2: 32 → 64 channels**
- Output: (64, H, W)
- **Learns:** Mid-level features combining basic patterns

**Conv Layer 3: 64 → 64 channels**
- Output: (64, H, W)
- **Learns:** High-level spatial strategies

**Fully Connected Layer 1: (64×H×W) → 512**
- Flattens spatial dimensions
- **Learns:** Global decision-making from local features

**Output Layer: 512 → 4**
- Produces Q(s, UP), Q(s, RIGHT), Q(s, DOWN), Q(s, LEFT)
- No activation (raw Q-values, can be positive or negative)

### Why Tanh Activation?

We use **Tanh** instead of ReLU for the following reasons:

**Mathematical Form:**
$$\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$

Output range: $[-1, 1]$

**Advantages for Snake:**

1. **Bounded Outputs:** Tanh outputs are naturally bounded, preventing explosive activations
   - ReLU: $[0, \infty)$ → Can grow unbounded
   - Tanh: $[-1, 1]$ → Always bounded

2. **Zero-Centered:** Tanh is centered around 0, which helps with learning
   - Negative Q-values (bad states) map naturally to negative activations
   - Positive Q-values (good states) map to positive activations

3. **Smooth Gradients:** Tanh provides gradients in both positive and negative directions
   - ReLU: Gradient is 0 for negative inputs (dead neurons)
   - Tanh: Gradient exists everywhere except at extremes

4. **Better for Grid Representations:** Our input values are binary (0 or 1), and Tanh handles this range well

**Comparison:**

| Activation | Range | Dead Neurons? | Zero-Centered? | Best For |
|------------|-------|---------------|----------------|----------|
| ReLU | $[0, \infty)$ | Yes (negative inputs) | No | Very deep networks, images |
| Tanh | $[-1, 1]$ | Rare (saturated extremes) | Yes | Grid games, bounded inputs |

### Parameter Count

For a 10×10 grid:

```
Conv1: (3 × 3 × 3 × 32) + 32 = 896 parameters
Conv2: (3 × 3 × 32 × 64) + 64 = 18,496 parameters
Conv3: (3 × 3 × 64 × 64) + 64 = 36,928 parameters
FC1:   (64 × 10 × 10 × 512) + 512 = 3,277,312 parameters
FC2:   (512 × 4) + 4 = 2,052 parameters
────────────────────────────────────────────────
Total: ~3.3 million parameters
```

**Note:** Most parameters are in the first fully-connected layer. This is typical for CNN architectures transitioning from spatial to decision layers.

## The Three Key Innovations of DQN

DQN introduced three critical techniques to stabilize neural network training for Q-Learning:

### 1. Experience Replay

**Problem:** Training on consecutive game steps leads to:
- **High correlation:** States at time $t$ and $t+1$ are very similar
- **Catastrophic forgetting:** New experiences overwrite old knowledge
- **Sample inefficiency:** Each experience used only once

**Solution:** Store experiences in a **replay buffer** and sample randomly.

**Replay Buffer Structure:**
```python
buffer = deque(maxlen=100_000)  # Fixed-size FIFO queue

# Store transitions
buffer.append((state, action, reward, next_state, done))

# Sample random batch for training
batch = random.sample(buffer, batch_size=64)
```

**Training Process:**
1. Play game, store transition $(s, a, r, s', \text{done})$ in buffer
2. Sample random batch of 64 transitions from buffer
3. Compute loss using batch
4. Update network weights

**Benefits:**
- ✅ Breaks temporal correlation (random sampling)
- ✅ Reuses experiences multiple times (sample efficiency)
- ✅ Stabilizes training (diverse batch composition)

**Memory Cost:** 100,000 transitions × ~50 bytes/transition ≈ 5 MB

### 2. Target Network

**Problem:** When computing the TD target, we use the network we're training:

$$\text{Target} = r + \gamma \max_{a'} Q_\theta(s', a')$$

But $Q_\theta$ is constantly changing! This creates a **moving target problem:**
- Update network → Target changes → Update again → Target changes more → Instability

**Solution:** Use a **separate target network** $Q_{\theta^-}$ that updates slowly.

**Two Networks:**
```python
q_network = QNetwork()        # Updated every step (fast)
target_network = QNetwork()   # Updated every 1000 steps (slow)

# Initially identical
target_network.load_state_dict(q_network.state_dict())
```

**Training with Target Network:**
```python
# Compute target using frozen target network
with torch.no_grad():  # No gradients through target
    target_q = reward + gamma * target_network(next_state).max()

# Compute current Q from main network
current_q = q_network(state)[action]

# Loss: squared difference
loss = (current_q - target_q) ** 2
```

**Update Schedule:**
```python
if steps % 1000 == 0:
    target_network.load_state_dict(q_network.state_dict())  # Sync networks
```

**Benefits:**
- ✅ Stable targets for 1000 steps
- ✅ Prevents oscillations
- ✅ More reliable convergence

**Why 1000 steps?**
- Too frequent (e.g., every 10 steps): Targets still move too much
- Too infrequent (e.g., every 10,000 steps): Target becomes stale
- 1000 steps ≈ 10-20 episodes: Good balance

### 3. CNN for Spatial Understanding

**Problem:** Fully-connected networks treat grid cells independently:
- Cell (5,5) and cell (5,6) have no relationship
- Network must learn the entire grid structure from scratch
- Exponential parameter count with grid size

**Solution:** Use CNNs to exploit spatial structure:
- Convolutional filters detect local patterns
- Weight sharing across spatial locations
- Translation invariance (pattern learned once, applies everywhere)

**Example Learned Features:**

*Conv Layer 1 might learn:*
- Filter 1: Detects "food in 3×3 neighborhood"
- Filter 2: Detects "body segment above or below"
- Filter 3: Detects "wall nearby"

*Conv Layer 2 might learn:*
- Filter 10: Combines "food nearby" + "no body blocking" → "clear path to food"
- Filter 20: Combines "walls on two sides" → "corridor situation"

*Conv Layer 3 might learn:*
- Filter 40: "Trap configuration" (walls + body creating closed space)
- Filter 50: "Food reachable via curved path"

## The DQN Algorithm

### Loss Function

DQN minimizes the **Temporal Difference (TD) error**:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ \left( \underbrace{Q_\theta(s, a)}_{\text{Prediction}} - \underbrace{\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a')\right)}_{\text{Target}} \right)^2 \right]$$

Where:
- $\mathcal{B}$: Replay buffer (sample distribution)
- $\theta$: Q-network parameters
- $\theta^-$: Target network parameters (frozen)
- $\gamma = 0.99$: Discount factor

**Intuition:** Make the predicted Q-value match what we actually observed (reward + discounted future value).

### Training Loop (High-Level)

```
Initialize Q-network with random weights θ
Initialize target network θ⁻ = θ
Initialize replay buffer B (capacity 100k)
Set ε = 1.0 (exploration rate)

For episode = 1 to N:
    state = reset_game()
    
    While not done:
        # 1. Select action (ε-greedy)
        if random() < ε:
            action = random_action()
        else:
            action = argmax(Q_θ(state))
        
        # 2. Execute action
        next_state, reward, done = env.step(action)
        
        # 3. Store transition
        B.push(state, action, reward, next_state, done)
        
        # 4. Train (if enough samples)
        if len(B) >= batch_size:
            batch = B.sample(batch_size)
            
            # Compute target
            targets = rewards + γ × max(Q_θ⁻(next_states))
            
            # Compute loss
            predictions = Q_θ(states)[actions]
            loss = MSE(predictions, targets)
            
            # Update Q-network
            θ ← θ - α∇loss
        
        # 5. Update target network (periodic)
        if steps % 1000 == 0:
            θ⁻ ← θ
        
        state = next_state
        steps += 1
    
    # 6. Decay exploration
    ε = max(ε × 0.995, 0.01)
```

## Hyperparameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Learning Rate** | 0.0001 | Low rate for neural networks (Adam optimizer adapts this) |
| **Discount Factor** | 0.99 | Same as tabular; long-term planning |
| **Batch Size** | 64 | Standard; balances gradient quality and update frequency |
| **Replay Buffer Size** | 100,000 | Stores ~1000 episodes of experience |
| **Target Update Freq** | 1000 steps | Sync target network every 1000 steps |
| **Initial Epsilon** | 1.0 | Start with full exploration |
| **Epsilon Decay** | 0.995 | Gradual decay per episode |
| **Min Epsilon** | 0.01 | Always maintain 1% exploration |
| **Optimizer** | Adam | Adaptive learning rate, handles sparse gradients well |

### Why Learning Rate = 0.0001?

**Neural networks require much smaller learning rates than tabular methods:**
- Tabular: α = 0.1 (update individual Q-values)
- DQN: α = 0.0001 (update millions of parameters that interact)

Too high → Unstable training, divergence  
Too low → Very slow learning  
0.0001 with Adam → Good balance

### Why Batch Size = 64?

- Smaller (16): Noisy gradients, less stable
- Larger (256): Smoother gradients, slower updates, more memory
- 64: Sweet spot (also GPU-efficient as power of 2)

## Expected Behavior by Grid Size

### 10×10 Grid: ✅ Success

**State Space:** ~10^9 states (intractable for tabular)

**Expected Performance:**
- Episodes to convergence: 2000-3000
- Final average score: 15-25 apples
- Training time: 15-20 minutes (CPU)

**Network Behavior:**
- First 500 episodes: Random exploration, learning basic features
- Episodes 500-1500: Learns to pursue food, avoid walls
- Episodes 1500-3000: Refines strategy, handles complex situations
- After 3000: Stable policy, can navigate efficiently

**Key Observation:** Despite visiting <0.1% of state space, the network generalizes to unseen states.

---

### 20×20 Grid: ✅ Success (More Challenging)

**State Space:** ~10^40 states (astronomical)

**Expected Performance:**
- Episodes to convergence: 5000-8000
- Final average score: 25-40 apples
- Training time: 30-45 minutes (CPU)

**Network Behavior:**
- Slower initial learning (larger space to explore)
- Benefits from CNN's spatial generalization
- May show more variance in performance
- Can still learn effective strategy despite massive state space

---

### Scalability: 30×30+ Grid: ⚠️ Possible

**With proper tuning:**
- Larger networks (more filters/layers)
- More training episodes
- Potentially GPU acceleration

DQN can theoretically handle very large grids because memory is O(parameters), not O(states).

## Advantages of DQN

### ✅ Generalization
Similar states produce similar Q-values without explicit storage.

**Example:**
```
Seen during training: Snake at (5,5), food at (7,5) → Q(RIGHT) = 8.2
Never seen: Snake at (12,8), food at (14,8) → Q(RIGHT) ≈ 8.0 (generalized!)
```

### ✅ Scalable Memory
~3M parameters (~12 MB) work for 10×10, 20×20, even 50×50 grids.

### ✅ Sample Efficient (vs. Tabular)
Learns from similar states through shared weights. One experience updates the entire network, affecting all states.

### ✅ Handles Large State Spaces
Can operate in state spaces with billions or trillions of states.

### ✅ Continuous Features Possible
Could handle continuous state features (e.g., exact pixel coordinates) with appropriate input normalization.

## Limitations of DQN

### ❌ Convergence Not Guaranteed
Unlike tabular Q-Learning, no theoretical guarantee of convergence. Can diverge or oscillate.

### ❌ Sample Inefficient (vs. Modern Methods)
DQN requires many more samples than policy gradient methods like PPO. Typical: 10^5 - 10^6 steps.

### ❌ Instability Issues
Without proper tuning, can suffer from:
- Divergence (Q-values explode)
- Forgetting (overwrites good policies)
- Oscillation (performance cycles)

### ❌ Hyperparameter Sensitive
Requires careful tuning of learning rate, batch size, network architecture, buffer size, etc.

### ❌ Computationally Expensive
- Neural network forward/backward passes are slow vs. table lookup
- Training time: minutes to hours vs. seconds for tabular

### ❌ Less Interpretable
Cannot inspect exact Q-values for a state. Network is a black box.

## Comparison: Tabular vs. DQN

| Aspect | Tabular Q-Learning | DQN |
|--------|-------------------|-----|
| **State Representation** | Hash/index | Raw grid tensor |
| **Q-Value Storage** | Dictionary/array | Neural network |
| **Memory** | O(\|visited states\|) | O(parameters) ≈ 12 MB |
| **Grid Size Limit** | ~7×7 | 20×20+ |
| **Training Time (10×10)** | Never converges | ~15 minutes |
| **Generalization** | None | Excellent |
| **Sample Efficiency** | Must revisit each state | Learns from similar states |
| **Convergence** | Guaranteed (theory) | Not guaranteed |
| **Interpretability** | High | Low |
| **Implementation** | Simple | Complex |

## Implementation Considerations

### Gradient Clipping

To prevent explosive gradients:
```python
torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
```

### Weight Initialization

Use **Xavier/Glorot initialization** for Tanh:
```python
for layer in q_network.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
```

### Exploration Strategy

ε-greedy with exponential decay:
- Start: ε = 1.0 (100% random)
- Decay: ε ← ε × 0.995 per episode
- End: ε = 0.01 (1% random, 99% exploitation)

## Common Pitfalls and Solutions

### Pitfall 1: Q-Values Explode
**Symptom:** Q-values grow to >1000, then NaN

**Solution:**
- Use target network (prevents moving targets)
- Clip gradients (max_norm=1.0)
- Lower learning rate

### Pitfall 2: No Learning After Initial Exploration
**Symptom:** Performance improves early, then plateaus

**Solution:**
- Check ε decay (not decaying too fast?)
- Ensure replay buffer is diverse
- Try larger buffer or smaller batch size

### Pitfall 3: Performance Oscillates Wildly
**Symptom:** Good policy suddenly becomes bad, then recovers

**Solution:**
- Increase target network update frequency (every 500 steps instead of 1000)
- Reduce learning rate
- Increase batch size for smoother gradients

### Pitfall 4: Very Slow Training
**Symptom:** 10,000 episodes, still random performance

**Solution:**
- Check reward function (is it providing signal?)
- Increase learning rate (carefully)
- Verify network architecture (too small? too large?)
- Check if gradients are flowing (print gradient norms)

## Summary

**DQN with CNN** solves the curse of dimensionality through:
1. **Function approximation** → Generalizes across states
2. **Experience replay** → Stabilizes training, improves sample efficiency
3. **Target network** → Prevents moving target problem
4. **CNN architecture** → Exploits spatial structure of grid

**For our Snake project:**
- ✅ Works well on 10×10 grid (where tabular fails)
- ✅ Scales to 20×20 grid
- ✅ Demonstrates power of deep RL

**Trade-offs:**
- More complex to implement and tune
- Longer training time per step
- Less interpretable
- But: handles problems that tabular methods cannot touch

**Next:** See how [Proximal Policy Optimization (PPO)](ppo.md) improves upon DQN with policy gradient methods and often achieves better sample efficiency and stability.

---

## Key Takeaways

1. **Function approximation enables generalization** - Learn patterns, not individual states
2. **CNNs are natural for grid games** - Spatial structure maps to convolutional filters
3. **Three innovations make DQN stable** - Replay, target network, CNN
4. **Tanh activation works well for bounded inputs** - Better than ReLU for grid games
5. **DQN scales where tabular fails** - 10×10, 20×20 grids are feasible
6. **Not perfect** - Still has instability issues, high sample complexity

DQN represents the bridge from classical RL to modern deep RL, showing how neural networks can tackle problems that were previously intractable.