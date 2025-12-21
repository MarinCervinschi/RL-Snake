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

We represent the game state as a **4-channel grid** of size $H \times W$, where:

- $H$: Grid height (default: 10)
- $W$: Grid width (default: 10)

Each state $s \in \mathcal{S}$ is represented as:

$$s = (G, d, t)$$

Where:

- $G \in \mathbb{R}^{H \times W \times 4}$: 4-channel grid with continuous values
- $d \in \{\text{UP}, \text{RIGHT}, \text{DOWN}, \text{LEFT}\}$: Current direction
- $t \in [0, 1]$: Normalized time-to-starvation

### 2.2 Channel Decomposition

The grid $G$ has four channels:

**Channel 0 - Snake Head:** $G[:, :, 0]$
$$G_{ij}^{(0)} = \begin{cases} 1.0 & \text{if cell }(i,j)\text{ contains snake head} \\ 0.0 & \text{otherwise} \end{cases}$$

**Channel 1 - Snake Body (Gradient):** $G[:, :, 1]$

$$
G_{ij}^{(1)} = \begin{cases}
\frac{n - k}{n} & \text{if cell }(i,j)\text{ contains body segment }k\text{ from head} \\
0.0 & \text{otherwise}
\end{cases}
$$

Where $n$ is the number of body segments and $k$ is the distance from the head (neck = 1, tail = n).

**Channel 2 - Food:** $G[:, :, 2]$
$$G_{ij}^{(2)} = \begin{cases} 1.0 & \text{if cell }(i,j)\text{ contains food} \\ 0.0 & \text{otherwise} \end{cases}$$

**Channel 3 - Time (Global Feature):** $G[:, :, 3]$
$$G_{ij}^{(3)} = t_{\text{norm}} = \min\left(\frac{\text{frame\_iteration}}{\text{max\_steps}}, 1.0\right), \quad \forall i,j$$

Where $\text{max\_steps} = 100 \times L_{\text{snake}}$ and the entire channel is filled with the same time value.

### 2.3 State Space Constraints

A valid state must satisfy:

1. **Single Head:** $\sum_{i,j} G_{ij}^{(0)} = 1$ (exactly one head)

2. **Connected Snake:** Body gradient decreases from neck to tail

3. **Single Food:** $\sum_{i,j} G_{ij}^{(2)} = 1$ (exactly one food)

4. **No Overlap:** Head, body, and food don't overlap

5. **Within Bounds:** Snake occupies between 3 and $H \times W$ cells

6. **Valid Time:** $0 \leq t_{\text{norm}} \leq 1$

### 2.4 State Space Size

The state space size depends on grid dimensions, snake length, and time discretization:

$$|\mathcal{S}| \approx \sum_{L=3}^{H \times W} N_{\text{snake}}(L) \times N_{\text{food}}(L) \times 4 \times N_{\text{time}}$$

Where:

- $L$: Snake length
- $N_{\text{snake}}(L)$: Number of valid snake configurations of length $L$
- $N_{\text{food}}(L)$: Number of valid food positions ≈ $H \times W - L$
- $4$: Number of possible directions
- $N_{\text{time}}$: Number of time buckets (10 for discretization)


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

$$
\text{Invalid}(a, d) = \begin{cases}
\text{True} & \text{if } (a = \text{UP} \land d = \text{DOWN}) \\
& \text{or } (a = \text{DOWN} \land d = \text{UP}) \\
& \text{or } (a = \text{LEFT} \land d = \text{RIGHT}) \\
& \text{or } (a = \text{RIGHT} \land d = \text{LEFT}) \\
\text{False} & \text{otherwise}
\end{cases}
$$

**Handling:** If an invalid action is attempted, the agent continues in the current direction $d$.

---

## 4. Transition Function ($\mathcal{P}$)

### 4.1 Deterministic Dynamics

The Snake environment is **deterministic**:

$$\mathcal{P}(s' | s, a) = \begin{cases} 1 & \text{if } s' = \mathcal{T}(s, a) \\ 0 & \text{otherwise} \end{cases}$$

Where $\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$ is the deterministic transition function.

### 4.2 Transition Function Definition

Given current state $s = (G, d, t)$ and action $a$:

**Step 1: Determine Effective Action**
$$a_{\text{eff}} = \begin{cases} d & \text{if Invalid}(a, d) \\ a & \text{otherwise} \end{cases}$$

**Step 2: Compute New Head Position**

Let $(h_x, h_y)$ be the current head position. The new head position is:

$$
(h_x', h_y') = (h_x, h_y) + \begin{cases}
(0, -1) & \text{if } a_{\text{eff}} = \text{UP} \\
(1, 0) & \text{if } a_{\text{eff}} = \text{RIGHT} \\
(0, 1) & \text{if } a_{\text{eff}} = \text{DOWN} \\
(-1, 0) & \text{if } a_{\text{eff}} = \text{LEFT}
\end{cases}
$$

**Step 3: Check Collision**

Collision occurs if:

$$
\text{Collision}(h_x', h_y', G) = \begin{cases}
\text{True} & \text{if } h_x' < 0 \lor h_x' \geq W \lor h_y' < 0 \lor h_y' \geq H \\
& \text{or } G_{h_y', h_x'}^{(1)} > 0 \\
\text{False} & \text{otherwise}
\end{cases}
$$

If collision: **Terminate episode** (transition to terminal state)

**Step 4: Check Food**

$$\text{AteFood}(h_x', h_y', G) = (G_{h_y', h_x'}^{(2)} = 1)$$

**Step 5: Update Snake**

If food was eaten:

- Prepend new head position to snake
- Snake length increases by 1
- **Do not** remove tail
- Reset frame counter to 0

If food was not eaten:

- Prepend new head position to snake
- Remove tail (last segment)
- Snake length remains constant
- Increment frame counter

**Step 6: Update Food**

If food was eaten:

- Randomly place new food at an empty cell

If food was not eaten:

- Food position remains the same

**Step 7: Update Time**

$$t' = \min\left(\frac{\text{frame\_iteration'}}{\text{max\_steps'}}, 1.0\right)$$

Where $\text{max\_steps'} = 100 \times L_{\text{snake'}}$

**Step 8: Construct Next State**

$$s' = (G', a_{\text{eff}}, t')$$

---

## 5. Reward Function ($\mathcal{R}$)

### 5.1 Reward Definition

The reward function includes:

$$
\mathcal{R}(s, a, s') = \begin{cases}
r_{\text{win}} = 1000 & \text{if snake fills entire grid} \\
r_{\text{collision}} = -10 & \text{if collision occurred} \\
r_{\text{timeout}} = -5 & \text{if frame\_iteration > max\_steps} \\
r_{\text{food}} + r_{\text{milestone}} & \text{if food was eaten} \\
r_{\text{step}} = 0 & \text{otherwise (normal movement)}
\end{cases}
$$

### 5.2 Milestone Rewards (Reward Shaping)

When food is eaten, additional milestone rewards are given at specific lengths:

$$
r_{\text{milestone}} = \begin{cases}
50 & \text{if } L = \lfloor 0.3 \times \text{max\_capacity} \rfloor \text{ (Expert)} \\
100 & \text{if } L = \lfloor 0.5 \times \text{max\_capacity} \rfloor \text{ (Master)} \\
500 & \text{if } L = \lfloor 0.9 \times \text{max\_capacity} \rfloor \text{ (Grandmaster)} \\
0 & \text{otherwise}
\end{cases}
$$

Where $\text{max\_capacity} = \lfloor 0.7 \times H \times W \rfloor$ (70% of grid).

### 5.3 Default Reward Values

| Event                 | Reward | Description        |
| --------------------- | ------ | ------------------ |
| Food eaten            | +10    | Primary objective  |
| Collision (wall/self) | -10    | Terminal failure   |
| Timeout (starvation)  | -5     | Efficiency penalty |
| Expert (30%)          | +50    | First milestone    |
| Master (50%)          | +100   | Second milestone   |
| Grandmaster (90%)     | +500   | Third milestone    |
| Win (100%)            | +1000  | Perfect game       |
| Normal step           | 0      | Neutral survival   |

---

## 6. Discount Factor ($\gamma$)

### 6.1 Value

We use:
$$\gamma = 0.99$$

### 6.2 Rationale

**High Discount Factor (0.99):**

1. **Planning Horizon:** Snake may need to navigate for many steps

   - 10×10 grid: May take 20-30 steps to reach food
   - With $\gamma = 0.99$: Reward 30 steps away is worth $0.99^{30} \times 10 \approx 7.4$

2. **Long-term Strategy:** Larger grids require strategic planning

3. **Effective Horizon:** $T \approx \frac{\log(0.01)}{\log(0.99)} \approx 459$ steps

---

## 7. Initial State Distribution

### 7.1 Definition

**Snake Initialization:**

Let $h_0 = (\lfloor W/2 \rfloor, \lfloor H/2 \rfloor)$ be the center cell.

Initial snake positions (length 3):
$$\text{Snake}_0 = [h_0, (h_0^x - 1, h_0^y), (h_0^x - 2, h_0^y)]$$

**Direction Initialization:**
$$d_0 = \text{RIGHT}$$

**Food Initialization:**
$$f_0 \sim \text{Uniform}(\{(i,j) : (i,j) \notin \text{Snake}_0\})$$

**Time Initialization:**
$$t_0 = 0.0$$

---

## 8. Terminal States

### 8.1 Definition

A state $s$ is **terminal** if:

1. **Collision with Wall:**
   $$h_x < 0 \lor h_x \geq W \lor h_y < 0 \lor h_y \geq H$$

2. **Collision with Self:**
   $$G_{h_y, h_x}^{(1)} > 0$$

3. **Timeout (Starvation):**
   $$\text{frame\_iteration} > 100 \times L_{\text{snake}}$$

4. **Win Condition:**
   $$L_{\text{snake}} = H \times W$$

---

## 9. Markov Property Verification

### 9.1 Formal Verification

We verify that the current state $(G, d, t)$ contains **all information** needed to predict the future.

**Required Information:**

1. ✅ **Snake positions:** Encoded in $G^{(0)}$ (head) and $G^{(1)}$ (body gradient)
2. ✅ **Food position:** Encoded in $G^{(2)}$
3. ✅ **Current direction:** Encoded in $d$
4. ✅ **Time pressure:** Encoded in $t$ and $G^{(3)}$
5. ✅ **Snake length:** Implicit in body gradient
6. ✅ **Snake order:** Body gradient reveals tail location

**What about history?**

- ❌ **Past positions:** Not needed (current positions + gradient fully determine state)
- ❌ **Previous actions:** Not needed (current direction is sufficient)
- ❌ **Score history:** Not needed (current length implies score)

### 9.2 Why Time is Necessary

The time channel $t$ is essential for the Markov property because:

1. **Starvation Detection:** Without time, the agent can't distinguish:

   - State A: Just ate food (safe)
   - State B: Haven't eaten for 95 steps (near timeout)

2. **Urgency Signal:** Time provides a global feature indicating risk level

3. **Timeout Rule:** $\text{max\_steps} = 100 \times L_{\text{snake}}$ means timeout depends on current length, which changes when food is eaten

**Therefore:** $(G, d, t)$ is sufficient to predict $(G', d', t')$ given action $a$. ∎