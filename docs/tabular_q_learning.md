# Tabular Q-Learning for Snake Game

> **Prerequisites:** Read the [Complete MDP Formulation](mdp_full_formulation.md) first to understand the state space, action space, and problem structure.

## Overview

**Tabular Q-Learning** is a classical reinforcement learning algorithm that stores Q-values (action-value estimates) in a lookup table. For each state-action pair $(s, a)$, the algorithm maintains an explicit value $Q(s, a)$ representing the expected cumulative reward of taking action $a$ in state $s$ and following the optimal policy thereafter.

**Key Characteristic:** Every unique state gets its own row in the table. This works for small state spaces but becomes impractical as the number of states grows exponentially.

## Algorithm Fundamentals

### The Q-Table

The Q-table is a 2D data structure:

$$Q: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

**Structure:**
```
           Action 0   Action 1   Action 2   Action 3
           (UP)       (RIGHT)    (DOWN)     (LEFT)
State 0  [  0.0    ,   0.0    ,   0.0    ,   0.0   ]
State 1  [  2.3    ,   5.7    ,  -1.2    ,   3.4   ]
State 2  [ -0.8    ,   8.9    ,   4.2    ,   1.1   ]
...
State N  [  1.5    ,   3.2    ,   0.9    ,  -2.1   ]
```

**Interpretation:** For State 1, the best action is Action 1 (RIGHT) with Q-value 5.7.

### The Bellman Update Rule

Q-Learning updates the table using the **Bellman equation**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Components:**
- $\alpha \in (0, 1]$: Learning rate (how fast to update beliefs)
- $r$: Immediate reward received
- $\gamma \in [0, 1]$: Discount factor (importance of future rewards)
- $\max_{a'} Q(s', a')$: Best Q-value achievable from next state $s'$

**Intuition:** 
> "Update my estimate of Q(s,a) by blending it with what I actually observed: the reward r plus the discounted value of the best action from the next state."

### The Learning Process

**Step-by-step:**
1. Observe current state $s$
2. Choose action $a$ using ε-greedy policy
3. Execute action, observe reward $r$ and next state $s'$
4. Update: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
5. Repeat until episode terminates

**Convergence Property:** Under mild conditions (all state-action pairs visited infinitely often, appropriate learning rate decay), Q-Learning provably converges to the optimal Q-function $Q^*$.

## State Representation Challenge

### The Core Problem

In our full MDP formulation, a state consists of:
- Full grid configuration (H×W with 3 channels)
- Current direction

For a 10×10 grid, the state space is approximately $10^9$ states. We cannot create a table with a billion rows!

### State Hashing Strategy

**Solution:** Convert the grid state into a hashable representation.

**Option 1: Position Tuple**
```python
def state_to_hash(state):
    """Convert GridState to hashable tuple"""
    snake_positions = tuple(tuple(pos) for pos in state.snake)
    food_position = tuple(state.food)
    direction = state.direction.value
    return (snake_positions, food_position, direction)
```

**Example:**
```
Snake: [(2,2), (2,1), (2,0)]
Food: (3,4)
Direction: RIGHT

Hash: (((2,2), (2,1), (2,0)), (3,4), 1)
```

**Option 2: Grid Encoding**
```python
def state_to_hash(state):
    """Convert grid to binary string"""
    # Flatten 3-channel grid and convert to binary string
    flat = state.grid.flatten()
    return tuple(flat)  # Hashable tuple of 0s and 1s
```

### Dictionary-Based Q-Table

Instead of pre-allocating a massive array, use a **dictionary**:

```python
q_table = {}  # Key: state_hash, Value: [Q(s,a₀), Q(s,a₁), Q(s,a₂), Q(s,a₃)]

def get_q_values(state_hash):
    if state_hash not in q_table:
        q_table[state_hash] = [0.0, 0.0, 0.0, 0.0]  # Initialize
    return q_table[state_hash]
```

**Advantage:** Only stores states that have been visited (sparse representation).

**Disadvantage:** Still limited by the number of reachable states, which grows exponentially with grid size.

## Action Selection: ε-Greedy Policy

The agent balances **exploration** (trying new actions) and **exploitation** (using known good actions):

$$\pi(s) = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon
\end{cases}$$

**Exploration Decay:**
- Start: $\epsilon = 1.0$ (100% random exploration)
- Decay: $\epsilon \leftarrow \epsilon \times 0.995$ after each episode
- Minimum: $\epsilon_{\min} = 0.01$ (always maintain 1% exploration)

**Rationale:** Early in training, Q-values are unreliable → explore widely. Later, Q-values are more accurate → exploit learned policy.

## Hyperparameters

| Parameter | Symbol | Recommended Value | Explanation |
|-----------|--------|-------------------|-------------|
| **Learning Rate** | $\alpha$ | 0.1 | Standard for tabular methods; balances stability and speed |
| **Discount Factor** | $\gamma$ | 0.99 | High value for long-term planning (full grid needs foresight) |
| **Initial Epsilon** | $\epsilon_0$ | 1.0 | Start with full exploration |
| **Epsilon Decay** | - | 0.995 | Gradual transition to exploitation |
| **Minimum Epsilon** | $\epsilon_{\min}$ | 0.01 | Maintain small exploration to avoid getting stuck |

### Why α = 0.1?

**Too high (α > 0.5):** New experiences drastically overwrite old knowledge → unstable, oscillating Q-values

**Too low (α < 0.01):** Learning is extremely slow → may never converge in reasonable time

**α = 0.1:** Good balance; each update adjusts Q-value by 10% toward the observed target.

### Why γ = 0.99?

With larger grids, the snake may need 20-30 steps to reach food. With γ = 0.99:

$$\text{Reward 30 steps away: } 10 \times 0.99^{30} \approx 7.4$$

Still significant! With γ = 0.9, it would only be worth ~0.4.

## Expected Behavior by Grid Size

### 5×5 Grid: ✅ Success

**State Space:** ~10,000 - 100,000 reachable states

**Expected Performance:**
- Episodes to convergence: 500-1000
- Final average score: 8-12 apples
- Training time: 1-2 minutes

**Q-Table Size:** ~1-10 MB (feasible)

**Behavior:** Smooth learning curve, converges to near-optimal policy. The agent learns to navigate efficiently, avoid walls, and pursue food.

---

### 7×7 Grid: ⚠️ Marginal

**State Space:** ~1-10 million reachable states

**Expected Performance:**
- Episodes to convergence: 5000+ (may not fully converge)
- Final average score: 3-7 apples
- Training time: 10-30 minutes

**Q-Table Size:** ~100 MB - 1 GB

**Behavior:** Learning is noisy and slow. The agent may learn basic strategies but struggles to refine the policy. Many states are visited only once or twice, leading to unreliable Q-values.

---

### 10×10 Grid: ❌ Failure

**State Space:** ~1 billion+ reachable states

**Expected Performance:**
- Episodes to convergence: Never
- Final average score: 1-3 apples (random-like behavior)
- Training time: Hours with no improvement

**Q-Table Size:** 10-100 GB (impractical)

**Behavior:** 
- The agent visits less than 0.01% of the state space even after 10,000 episodes
- Q-values remain mostly at initialization (0.0)
- No meaningful learning occurs
- Performance is essentially random

**This is the curse of dimensionality in action.**

---

## The Curse of Dimensionality

### Mathematical Explanation

State space size grows **super-exponentially** with grid dimensions:

| Grid | Cells | Approx. States | Q-Table Memory |
|------|-------|----------------|----------------|
| 5×5 | 25 | $10^4$ | ~1 MB ✅ |
| 7×7 | 49 | $10^6$ | ~100 MB ⚠️ |
| 10×10 | 100 | $10^9$ | ~10 GB ❌ |
| 20×20 | 400 | $10^{40}$ | Astronomical ❌ |

### Why This Happens

For a grid of size $H \times W$ with a snake of length $L$:
- Snake can occupy $\binom{H \times W}{L}$ different position sets
- For each position set, there are $\approx L!$ orderings (path configurations)
- Food can be in $H \times W - L$ locations
- Direction has 4 possibilities

Rough estimate: $\mathcal{O}((H \times W)^L \times 4)$ states

As the grid grows, $L$ can grow toward $H \times W$, causing exponential explosion.

### Practical Implication

**Even with sparse storage (dictionary), we cannot visit enough states to learn:**

Suppose we train for 10,000 episodes, each lasting 100 steps on average:
- Total steps: $10^6$
- Unique states visited: $\sim 10^5 - 10^6$ (with some revisits)
- For 10×10 grid with $10^9$ states: We visit only **0.1%** of the space!

**Conclusion:** Tabular Q-Learning fundamentally cannot scale to medium/large grids.

## Advantages of Tabular Q-Learning

Despite its limitations, tabular Q-Learning has important strengths:

### ✅ Simplicity
- Easy to implement (~50 lines of code)
- No neural network complexity
- No hyperparameter tuning for network architecture

### ✅ Interpretability
- Can inspect exact Q-values for any state
- Understand why the agent chose a particular action
- Debug by examining Q-table entries

### ✅ Theoretical Guarantees
- Proven convergence to optimal policy $Q^*$ (under conditions)
- No approximation error (unlike function approximation)
- Exact value function representation

### ✅ Fast Updates
- O(1) table lookup and update
- No gradient computation or backpropagation
- Suitable for real-time learning

### ✅ Pedagogical Value
- Teaches core RL concepts without deep learning complexity
- Shows the curse of dimensionality empirically
- Motivates need for function approximation

## Limitations of Tabular Q-Learning

### ❌ Exponential State Space Growth
Cannot handle problems with more than ~10^6 states.

### ❌ No Generalization
Learning about state $s_1$ provides zero information about similar state $s_2$. Each state is treated independently.

**Example:** 
- State A: Snake at (5,5), food at (7,5)
- State B: Snake at (5,6), food at (7,6)

These states are almost identical (shifted by one cell), but the agent treats them as completely unrelated.

### ❌ Sample Inefficiency
Must visit each state many times to get accurate Q-values. With sparse state visitation, most Q-values remain unreliable.

### ❌ Memory Constraints
Even with sparse storage, dictionaries with millions of entries consume significant RAM.

### ❌ Cannot Handle Continuous States
Requires discrete state space. Cannot directly handle continuous observations like exact pixel coordinates.

## Comparison Preview

### Tabular Q-Learning vs. DQN

| Aspect | Tabular Q-Learning | DQN (next doc) |
|--------|-------------------|-----------------|
| **Representation** | Hash table | Neural network |
| **Generalization** | None | Excellent |
| **Memory** | O(\|visited states\|) | O(parameters) ~ constant |
| **Grid Size Limit** | ~7×7 | 20×20+ |
| **Sample Efficiency** | Low (must revisit states) | High (learns from similar states) |
| **Interpretability** | High | Low |
| **Implementation** | Simple | Complex |

### When to Use Tabular Q-Learning

✅ **Use tabular methods when:**
- State space is small (<100k states)
- States are discrete and enumerable
- You need interpretability
- Problem is for learning/teaching purposes

❌ **Avoid tabular methods when:**
- State space is large (>1M states)
- States are continuous
- You need generalization
- Computational resources are limited

## Summary

**Tabular Q-Learning** is the foundation of reinforcement learning, providing:
- A clear introduction to value-based methods
- Theoretical grounding with convergence guarantees
- Empirical demonstration of the curse of dimensionality

**For our Snake project:**
- ✅ Works well on 5×5 grid (demonstrates the algorithm)
- ⚠️ Struggles on 7×7 grid (shows scaling challenges)
- ❌ Fails on 10×10 grid (motivates deep RL)

This failure is not a weakness of the project—it's a **feature**! It demonstrates empirically why the field moved toward function approximation and deep reinforcement learning.

**Next:** See how [Deep Q-Networks (DQN)](dqn_cnn.md) solve these limitations using neural networks and convolutional architectures.

---

## Key Takeaways

1. **Q-Learning is elegant and simple** - One update rule, proven convergence
2. **The curse of dimensionality is real** - State space grows exponentially
3. **Sparse storage helps but isn't enough** - Can't visit enough states to learn
4. **Tabular methods motivate deep RL** - Empirically shows why we need approximation
5. **Perfect for small problems** - Still the go-to for discrete, small state spaces

This is why we implement tabular Q-Learning first: to understand both its power and its fundamental limitations.