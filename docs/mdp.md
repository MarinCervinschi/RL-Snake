# Snake Game: Complete MDP Formulation

> **Document Purpose:** This document provides a rigorous mathematical formulation of the Snake game as a **true Markov Decision Process (MDP)**, using full grid state representation.

## Table of Contents
1. [MDP Definition](#1-mdp-definition)
2. [State Space (S)](#2-state-space-s)
3. [Action Space (A)](#3-action-space-a)
4. [Transition Function (P)](#4-transition-function-p)
5. [Reward Function (R)](#5-reward-function-r)
6. [Discount Factor (γ)](#6-discount-factor-γ)
7. [Initial State Distribution](#7-initial-state-distribution)
8. [Terminal States](#8-terminal-states)
9. [Markov Property Verification](#9-markov-property-verification)
10. [Problem Complexity Analysis](#10-problem-complexity-analysis)

---

## 1. MDP Definition

A **Markov Decision Process** is formally defined as a 5-tuple:

$$\text{MDP} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

Where:
- $\mathcal{S}$: State space (set of all possible game configurations)
- $\mathcal{A}$: Action space (set of available actions)
- $\mathcal{P}$: Transition probability function $\mathcal{P}(s' | s, a)$
- $\mathcal{R}$: Reward function $\mathcal{R}(s, a, s')$
- $\gamma \in [0, 1]$: Discount factor

**Key Property:** The Markov property must hold:
$$\mathbb{P}(S_{t+1} = s' | S_t = s_t, A_t = a_t, S_{t-1}, A_{t-1}, \ldots) = \mathbb{P}(S_{t+1} = s' | S_t = s_t, A_t = a_t)$$

The future depends only on the current state and action, not on history.

---

## 2. State Space ($\mathcal{S}$)

### 2.1 Grid Representation

We represent the game state as a **3-channel grid** of size $H \times W$, where:
- $H$: Grid height (e.g., 5, 10, or 20)
- $W$: Grid width (e.g., 5, 10, or 20)

Each state $s \in \mathcal{S}$ is represented as:

$$s = (G, d)$$

Where:
- $G \in \{0, 1\}^{H \times W \times 3}$: Binary 3-channel grid
- $d \in \{\text{UP}, \text{RIGHT}, \text{DOWN}, \text{LEFT}\}$: Current direction

### 2.2 Channel Decomposition

The grid $G$ has three channels:

**Channel 0 - Snake Head:** $G[:, :, 0]$.  
$$G_{ij}^{(0)} = \begin{cases} 1 & \text{if cell }(i,j)\text{ contains snake head} \\ 0 & \text{otherwise} \end{cases}$$

**Channel 1 - Snake Body:** $G[:, :, 1]$.  
$$G_{ij}^{(1)} = \begin{cases} 1 & \text{if cell }(i,j)\text{ contains snake body segment} \\ 0 & \text{otherwise} \end{cases}$$

**Channel 2 - Food:** $G[:, :, 2]$.  
$$G_{ij}^{(2)} = \begin{cases} 1 & \text{if cell }(i,j)\text{ contains food} \\ 0 & \text{otherwise} \end{cases}$$

### 2.3 State Space Constraints

A valid state must satisfy:

1. **Single Head:** $\sum_{i,j} G_{ij}^{(0)} = 1$ (exactly one head)

2. **Connected Snake:** Body segments form a contiguous path from head

3. **Single Food:** $\sum_{i,j} G_{ij}^{(2)} = 1$ (exactly one food)

4. **No Overlap:** $G_{ij}^{(0)} \cdot G_{ij}^{(1)} = 0$ and $G_{ij}^{(0)} \cdot G_{ij}^{(2)} = 0$ and $G_{ij}^{(1)} \cdot G_{ij}^{(2)} = 0$ (head, body, and food don't overlap)

5. **Within Bounds:** Snake occupies between 3 and $H \times W - 1$ cells

### 2.4 State Space Size

The state space size depends on grid dimensions and snake length:

$$|\mathcal{S}| \approx \sum_{L=3}^{H \times W - 1} N_{\text{snake}}(L) \times N_{\text{food}}(L) \times 4$$

Where:
- $L$: Snake length
- $N_{\text{snake}}(L)$: Number of valid snake configurations of length $L$
- $N_{\text{food}}(L)$: Number of valid food positions for a snake of length $L$ (approximately $H \times W - L$)
- $4$: Number of possible directions

**Estimates for different grid sizes:**

| Grid Size | Approximate $\|\mathcal{S}\|$ | Feasibility for Tabular Methods |
|-----------|-------------------------------|----------------------------------|
| **5×5** | $\sim 10^4 - 10^5$ | ✅ Feasible |
| **7×7** | $\sim 10^6 - 10^7$ | ⚠️ Marginal |
| **10×10** | $\sim 10^9 - 10^{10}$ | ❌ Infeasible |
| **20×20** | $\sim 10^{40} - 10^{50}$ | ❌ Completely impractical |

**Note:** These are rough estimates. The actual number of reachable states through valid gameplay is smaller, but still grows exponentially.

## 3. Action Space ($\mathcal{A}$)

### 3.1 Action Definition

The action space consists of **four absolute directions**:

$$\mathcal{A} = \{\text{UP}, \text{RIGHT}, \text{DOWN}, \text{LEFT}\}$$

Or numerically:
$$\mathcal{A} = \{0, 1, 2, 3\}$$

Where:
- Action 0 = UP: Move in $-y$ direction (decrease row index)
- Action 1 = RIGHT: Move in $+x$ direction (increase column index)
- Action 2 = DOWN: Move in $+y$ direction (increase row index)
- Action 3 = LEFT: Move in $-x$ direction (decrease column index)


### 3.2 Invalid Actions (180° Turns)

The snake **cannot reverse direction** (move directly opposite to current direction):

$$\text{Invalid}(a, d) = \begin{cases} 
\text{True} & \text{if } (a = \text{UP} \land d = \text{DOWN}) \\
& \text{or } (a = \text{DOWN} \land d = \text{UP}) \\
& \text{or } (a = \text{LEFT} \land d = \text{RIGHT}) \\
& \text{or } (a = \text{RIGHT} \land d = \text{LEFT}) \\
\text{False} & \text{otherwise}
\end{cases}$$

**Handling:** If an invalid action is attempted, the agent continues in the current direction $d$.

### 3.3 Why Absolute Actions?

**Decision:** We use absolute actions (UP, RIGHT, DOWN, LEFT) instead of relative actions (STRAIGHT, TURN_LEFT, TURN_RIGHT) because:

1. **Full Observability:** With complete grid state, absolute position is known
2. **Simpler Policy:** No need to mentally track "relative to what direction?"
3. **Standard for Grid Environments:** Consistent with environments like GridWorld, Atari
4. **Better for CNNs:** Spatial convolutions naturally understand absolute directions

---

## 4. Transition Function ($\mathcal{P}$)

### 4.1 Deterministic Dynamics

The Snake environment is **deterministic**:

$$\mathcal{P}(s' | s, a) = \begin{cases} 1 & \text{if } s' = \mathcal{T}(s, a) \\ 0 & \text{otherwise} \end{cases}$$

Where $\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$ is the deterministic transition function.

### 4.2 Transition Function Definition

Given current state $s = (G, d)$ and action $a$:

**Step 1: Determine Effective Action**
$$a_{\text{eff}} = \begin{cases} d & \text{if Invalid}(a, d) \\ a & \text{otherwise} \end{cases}$$

**Step 2: Compute New Head Position**

Let $(h_x, h_y)$ be the current head position. The new head position is:

$$(h_x', h_y') = \begin{cases}
(h_x, h_y - 1) & \text{if } a_{\text{eff}} = \text{UP} \\
(h_x + 1, h_y) & \text{if } a_{\text{eff}} = \text{RIGHT} \\
(h_x, h_y + 1) & \text{if } a_{\text{eff}} = \text{DOWN} \\
(h_x - 1, h_y) & \text{if } a_{\text{eff}} = \text{LEFT}
\end{cases}$$

**Step 3: Check Collision**

Collision occurs if:
$$\text{Collision}(h_x', h_y', G) = \begin{cases}
\text{True} & \text{if } h_x' < 0 \lor h_x' \geq W \lor h_y' < 0 \lor h_y' \geq H \\
& \text{or } G_{h_y', h_x'}^{(1)} = 1 \\
\text{False} & \text{otherwise}
\end{cases}$$

If collision: **Terminate episode** (transition to terminal state $s_{\text{term}}$)

**Step 4: Check Food**

$$\text{AteFood}(h_x', h_y', G) = \begin{cases}
\text{True} & \text{if } G_{h_y', h_x'}^{(2)} = 1 \\
\text{False} & \text{otherwise}
\end{cases}$$

**Step 5: Update Snake**

If food was eaten:
- Prepend new head position to snake
- Snake length increases by 1
- **Do not** remove tail

If food was not eaten:
- Prepend new head position to snake
- Remove tail (last segment)
- Snake length remains constant

**Step 6: Update Food**

If food was eaten:
- Randomly place new food at an empty cell: $f' \sim \text{Uniform}(\{(i,j) : G_{ij}^{(0)} = 0 \land G_{ij}^{(1)} = 0\})$

If food was not eaten:
- Food position remains the same: $f' = f$

**Step 7: Construct Next State**

$$s' = (G', a_{\text{eff}})$$

Where $G'$ is the updated grid with new snake and food positions.


## 5. Reward Function ($\mathcal{R}$)

### 5.1 Reward Definition

The reward function is:

$$\mathcal{R}(s, a, s') = \begin{cases}
+10 & \text{if food was eaten in transition} \\
-10 & \text{if collision occurred (terminal state)} \\
0 & \text{otherwise (normal movement)}
\end{cases}$$

### 5.2 Formal Specification

Let:
- $(h_x, h_y)$: Head position in state $s$
- $(h_x', h_y')$: Head position after taking action $a$
- $G$: Grid in state $s$

$$\mathcal{R}(s, a, s') = \begin{cases}
-10 & \text{if Collision}(h_x', h_y', G) \\
+10 & \text{if } G_{h_y', h_x'}^{(2)} = 1 \text{ and not Collision} \\
0 & \text{otherwise}
\end{cases}$$

### 5.3 Reward Design Rationale

**+10 for Food:**
- Primary objective of the game
- Strong positive signal to guide learning
- Magnitude chosen to dominate discounted future rewards

**-10 for Collision:**
- Terminal failure state
- Strong negative signal to avoid dangerous actions
- Equal magnitude to food ensures balanced learning

**0 for Survival:**
- Neutral reward for normal movement
- Prevents "reward farming" by circling indefinitely
- Encourages efficiency (shorter paths to food are implicitly better)

### 5.4 Expected Return Calculation

The **expected return** from state $s_t$ following policy $\pi$ is:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$


## 6. Discount Factor ($\gamma$)

### 6.1 Value

We use:
$$\gamma = 0.99$$

### 6.2 Rationale

**High Discount Factor (0.99 vs 0.9 in compressed version):**

1. **Planning Horizon:** Snake may need to navigate around obstacles for many steps
   - 20×20 grid: May take 30+ steps to reach food
   - With $\gamma = 0.99$: Reward 30 steps away is worth $0.99^{30} \times 10 \approx 7.4$
   - With $\gamma = 0.90$: Reward 30 steps away is worth $0.90^{30} \times 10 \approx 0.4$

2. **Long-term Strategy:** Larger grids require more strategic planning

3. **Delayed Rewards:** Food collection can be far in the future

**Comparison:**

| $\gamma$ | Horizon | 10 steps away | 20 steps away | 30 steps away |
|----------|---------|---------------|---------------|---------------|
| 0.90 | Short | $10 \times 0.9^{10} = 3.49$ | $10 \times 0.9^{20} = 1.22$ | $10 \times 0.9^{30} = 0.42$ |
| 0.95 | Medium | $10 \times 0.95^{10} = 5.99$ | $10 \times 0.95^{20} = 3.58$ | $10 \times 0.95^{30} = 2.15$ |
| **0.99** | **Long** | $10 \times 0.99^{10} = 9.04$ | $10 \times 0.99^{20} = 8.18$ | $10 \times 0.99^{30} = 7.40$ |

### 6.3 Effective Horizon

The **effective horizon** is the number of steps $T$ such that $\gamma^T \approx 0.01$:

$$T \approx \frac{\log(0.01)}{\log(\gamma)}$$

For $\gamma = 0.99$:
$$T \approx \frac{\log(0.01)}{\log(0.99)} \approx 459 \text{ steps}$$

This means the agent effectively plans ~460 steps into the future.

---

## 7. Initial State Distribution

### 7.1 Definition

The initial state distribution $s_0 \sim \mu_0$ specifies how the game starts.

**Components:**
1. Snake starts with length 3
2. Snake is placed in the center of the grid
3. Initial direction is chosen randomly (or fixed to RIGHT)
4. Food is placed randomly at an empty cell

### 7.2 Formal Specification

**Snake Initialization:**

Let $h_0 = (\lfloor W/2 \rfloor, \lfloor H/2 \rfloor)$ be the center cell.

Initial snake positions:
$$\text{Snake}_0 = [h_0, (h_0^x - 1, h_0^y), (h_0^x - 2, h_0^y)]$$

This creates a horizontal snake of length 3 facing RIGHT.

**Direction Initialization:**
$$d_0 = \text{RIGHT}$$

**Food Initialization:**
$$f_0 \sim \text{Uniform}(\{(i,j) : (i,j) \notin \text{Snake}_0\})$$

Food is placed uniformly at random among all empty cells.


## 8. Terminal States

### 8.1 Definition

A state $s$ is **terminal** if:

1. **Collision with Wall:**
   $$h_x < 0 \lor h_x \geq W \lor h_y < 0 \lor h_y \geq H$$

2. **Collision with Self:**
   $$G_{h_y, h_x}^{(1)} = 1$$
   (Head position collides with body)

3. **Timeout (Optional):**
   $$t > 100 \times L_{\text{snake}}$$
   (Prevents infinite loops; steps exceed 100× snake length)

### 8.2 Terminal State Transition

Once in a terminal state $s_{\text{term}}$:
- All actions lead back to $s_{\text{term}}$ (absorbing state)
- Reward is 0 for all subsequent transitions
- Episode ends

$$\mathcal{P}(s_{\text{term}} | s_{\text{term}}, a) = 1, \quad \forall a \in \mathcal{A}$$
$$\mathcal{R}(s_{\text{term}}, a, s_{\text{term}}) = 0, \quad \forall a \in \mathcal{A}$$

### 8.3 Episode Termination

In practice, when a terminal state is reached:
- Episode is terminated
- New episode begins with initial state from $\mu_0$

**Episode Length:**
- Minimum: 1 step (immediate collision, rare)
- Average: 10-50 steps (depends on policy quality)
- Maximum (with timeout): $100 \times L_{\text{max}}$ steps

---

## 9. Markov Property Verification

### 9.1 Formal Verification

We verify that the current state $(G, d)$ contains **all information** needed to predict the future.

**Required Information:**
1. ✅ **Snake positions:** Encoded in $G^{(0)}$ (head) and $G^{(1)}$ (body)
2. ✅ **Food position:** Encoded in $G^{(2)}$
3. ✅ **Current direction:** Encoded in $d$
4. ✅ **Snake length:** Implicit in $\sum_{i,j} (G_{ij}^{(0)} + G_{ij}^{(1)})$

**What about history?**
- ❌ **Past positions:** Not needed (current positions fully determine valid actions)
- ❌ **Previous actions:** Not needed (state is fully observable)
- ❌ **Score history:** Not needed (current snake length implies score)

### 9.2 Proof of Markov Property

**Theorem:** The state representation $s = (G, d)$ satisfies the Markov property.

**Proof:**

Let $s_t = (G_t, d_t)$ be the state at time $t$.

**Claim:** $\mathbb{P}(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \ldots) = \mathbb{P}(S_{t+1} | S_t, A_t)$

**Argument:**

The next state $s_{t+1}$ is determined by:
1. New head position: $h_{t+1} = f(h_t, a_t)$ (deterministic function)
2. New body positions: Derived from current body $+ h_t$ (head becomes body) $-$ tail (removed)
3. New food position: Either unchanged or randomly placed (if food eaten)
4. New direction: $d_{t+1} = a_t$ (effective action)

All of these depend **only** on $(G_t, d_t, a_t)$:
- $h_t$ is extracted from $G_t^{(0)}$
- Body positions are in $G_t^{(1)}$
- Food position is in $G_t^{(2)}$
- Current direction is $d_t$

No information from $s_{t-1}, s_{t-2}, \ldots$ is needed.

Therefore, $s_t$ is **sufficient** to predict $s_{t+1}$, satisfying the Markov property. ∎

---

## 10. Problem Complexity Analysis

### 10.1 State Space Growth

| Grid Size | States (Order of Magnitude) | Tabular Q-Table Memory | Feasibility |
|-----------|----------------------------|------------------------|-------------|
| 5×5 | $\sim 10^{4}$ | ~100 KB | ✅ Excellent |
| 7×7 | $\sim 10^{6}$ | ~10 MB | ⚠️ Marginal |
| 10×10 | $\sim 10^{9}$ | ~10 GB | ❌ Impractical |
| 15×15 | $\sim 10^{20}$ | Exabytes | ❌ Impossible |
| 20×20 | $\sim 10^{40}$ | Beyond universe | ❌ Impossible |

**Calculation for Q-Table:**
$$\text{Memory} = |\mathcal{S}| \times |\mathcal{A}| \times 8 \text{ bytes (float64)}$$

### 10.2 Why Tabular Methods Fail

**The Curse of Dimensionality:**

As grid size increases, the state space grows **super-exponentially**:

$$|\mathcal{S}| \propto (H \times W)^{H \times W}$$

**Example:**
- 5×5 grid: 25 cells, state space $\sim 10^4$
- 10×10 grid: 100 cells, state space $\sim 10^{40}$ (not just $4 \times$ larger!)

**Why?**
- Snake can occupy any subset of cells
- Combinatorial explosion in valid configurations
- Number of possible paths grows factorially

**Consequence:**
- Tabular Q-Learning must visit each state many times to learn accurate Q-values
- With $10^{40}$ states, even visiting each once would take longer than age of universe

### 10.3 Why Function Approximation Works

**Key Insight:** Nearby states have similar Q-values.

**Example:**
```
State A: Snake at (10, 10), food at (15, 10)
State B: Snake at (10, 11), food at (15, 11)

These states are very similar!
Optimal action is likely the same (move toward food).
```

**Function Approximation (Neural Networks):**
- Learn a **function** $Q_\theta(s, a)$ instead of a table
- **Generalize** from seen states to unseen states
- **Compress** state space using learned features

**Convolutional Neural Networks (CNNs):**
- Exploit **spatial structure** of grid
- Learn features like "food nearby", "wall ahead", "body blocking"
- Share weights across grid positions (translation invariance)

**Memory Requirement:**
- Tabular: $O(|\mathcal{S}| \times |\mathcal{A}|)$ (exponential in grid size)
- CNN: $O(\text{parameters})$ (constant, independent of grid size!)

For our CNN:
```
~100K parameters × 4 bytes = 400 KB
```

This works for 5×5, 10×10, 20×20, even 50×50 grids!

### 10.4 Sample Complexity

**Number of samples needed to learn:**

**Tabular Q-Learning:**
- Must visit each state-action pair multiple times
- Sample complexity: $O(|\mathcal{S}| \times |\mathcal{A}|)$
- For 10×10: $\sim 10^9$ samples (impossible)

**DQN with CNN:**
- Generalizes across similar states
- Sample complexity: $O(\text{task complexity})$, independent of $|\mathcal{S}|$
- For 10×10: $\sim 10^5 - 10^6$ samples (feasible!)

### 10.5 Summary Table

| Method | State Rep | Memory | Sample Complexity | 5×5 | 10×10 | 20×20 |
|--------|-----------|--------|-------------------|-----|-------|-------|
| **Tabular Q-Learning** | Hash table | $O(\|\mathcal{S}\| \times \|\mathcal{A}\|)$ | $O(\|\mathcal{S}\| \times \|\mathcal{A}\|)$ | ✅ | ❌ | ❌ |
| **DQN (FC network)** | Flatten grid | $O(H \times W \times \text{neurons})$ | $O(\text{moderate})$ | ✅ | ⚠️ | ❌ |
| **DQN (CNN)** | 3-channel grid | $O(\text{filters} \times \text{kernels})$ | $O(\text{moderate})$ | ✅ | ✅ | ✅ |
| **PPO (CNN)** | 3-channel grid | $O(\text{filters} \times \text{kernels})$ | $O(\text{lower})$ | ✅ | ✅ | ✅ |
