# Proximal Policy Optimization (PPO) for Snake Game

> **Prerequisites:**
> 1. Read the [Complete MDP Formulation](mdp_full_formulation.md) to understand the problem
> 2. Read [Tabular Q-Learning](tabular_q_learning.md) to understand value-based methods
> 3. Read [DQN-CNN](dqn_cnn.md) to understand deep Q-learning and CNN architectures

## Overview

**Proximal Policy Optimization (PPO)** is a state-of-the-art policy gradient algorithm that learns a policy directly, rather than learning Q-values and deriving a policy from them. PPO is known for its **stability**, **sample efficiency**, and **ease of tuning** compared to other reinforcement learning methods.

**Key Innovation:** PPO optimizes the policy while constraining updates to stay within a "trust region," preventing destructive policy changes that could ruin learning progress.

**For Snake:** We use an **Actor-Critic architecture** with a shared CNN backbone, where the actor learns the policy π(a|s) and the critic learns the value function V(s).

## Value-Based vs. Policy-Based Methods

### The Fundamental Difference

**Value-Based (Q-Learning, DQN):**
```
Learn Q(s,a) → Derive policy: π(s) = argmax_a Q(s,a)
```
- Learn "how good is each action?"
- Policy is implicit (always pick best Q-value)
- Works well but can be unstable

**Policy-Based (PPO):**
```
Learn π(a|s) directly → Sample actions from policy
```
- Learn "what action should I take?"
- Policy is explicit (probability distribution over actions)
- More stable, can learn stochastic policies

### Mathematical Formulation

**DQN:** Learn $Q_\theta(s, a) \rightarrow \mathbb{R}$ (Q-value function)
- Policy: $\pi(s) = \arg\max_a Q_\theta(s, a)$ (deterministic)

**PPO:** Learn $\pi_\theta(a | s) \rightarrow [0, 1]$ (policy distribution)
- Policy: Sample $a \sim \pi_\theta(\cdot | s)$ (stochastic)
- Also learn $V_\phi(s) \rightarrow \mathbb{R}$ (value function for variance reduction)

### Example

**State:** Snake at (5,5), food at (7,5)

**DQN Output:**
```
Q(s, UP) = 2.3
Q(s, RIGHT) = 8.7  ← Always choose this
Q(s, DOWN) = 1.2
Q(s, LEFT) = -3.4
```
Deterministic: Always go RIGHT (argmax)

**PPO Output:**
```
π(UP | s) = 0.10    (10% probability)
π(RIGHT | s) = 0.75  (75% probability) ← Usually choose this
π(DOWN | s) = 0.12   (12% probability)
π(LEFT | s) = 0.03   (3% probability)
```
Stochastic: Sample from distribution (usually RIGHT, sometimes others)

## Actor-Critic Architecture

PPO uses an **Actor-Critic** framework with two components:

### The Actor (Policy Network)

**Role:** Decides what action to take

**Input:** State $s$  
**Output:** Probability distribution over actions $\pi_\theta(a | s)$

**Mathematical Form:**
$$\pi_\theta(a | s) = \text{Softmax}(f_\theta(s))$$

Where $f_\theta(s)$ are raw logits (unnormalized scores).

### The Critic (Value Network)

**Role:** Evaluates how good the current state is

**Input:** State $s$  
**Output:** State value $V_\phi(s)$

**Mathematical Form:**
$$V_\phi(s) \approx \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s \right]$$

The critic estimates the expected cumulative reward from state $s$.

### Why Both?

**Actor alone:** High variance (policy gradients are noisy)  
**Critic alone:** Cannot represent stochastic policies  
**Actor-Critic together:** Critic reduces variance, actor learns policy

**Advantage Function:**
$$A(s, a) = Q(s, a) - V(s)$$

"How much better is action $a$ compared to the average action in state $s$?"

We can estimate: $A(s, a) \approx r + \gamma V(s') - V(s)$ (TD error)

## Network Architecture

Our PPO implementation uses a **shared CNN backbone** with two heads:

```
Input: Grid State (3 × H × W)
         ↓
    ┌────────────────────┐
    │  Shared CNN Backbone │
    └────────────────────┘
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
    ┌──────────┴──────────┐
    ↓                     ↓
┌─────────┐         ┌─────────┐
│  Actor  │         │ Critic  │
│  Head   │         │  Head   │
└─────────┘         └─────────┘
    ↓                     ↓
Linear (512 → 256)   Linear (512 → 256)
    ↓                     ↓
  Tanh                  Tanh
    ↓                     ↓
Linear (256 → 4)     Linear (256 → 1)
    ↓                     ↓
 Softmax             (No activation)
    ↓                     ↓
π(a|s) ∈ [0,1]⁴      V(s) ∈ ℝ
(probabilities)       (state value)
```

### Shared Backbone Rationale

**Why share CNN layers?**

1. **Efficiency:** Compute features once, use for both actor and critic
2. **Representation Learning:** Both tasks benefit from same spatial features
3. **Parameter Sharing:** Reduces total parameters (vs. two separate networks)
4. **Faster Training:** Shared representations learn faster

**What does each part do?**

**Shared CNN (Conv layers + first FC):**
- Extracts spatial features: "food nearby", "wall ahead", "body configuration"
- These features are useful for both deciding actions AND evaluating state quality

**Actor Head:**
- Maps features → action probabilities
- Learns: "Given this situation, what's the best distribution over actions?"

**Critic Head:**
- Maps features → state value
- Learns: "Given this situation, what's the expected cumulative reward?"

### Why Tanh (Again)?

Same rationale as DQN:
- Bounded activations: [-1, 1]
- Zero-centered: Good for both positive/negative values and probabilities
- Smooth gradients: No dead neurons
- Works well with grid inputs

## The PPO Algorithm

### Core Idea: Trust Region Optimization

**Problem with vanilla policy gradients:**
- Large policy updates can be catastrophic
- If new policy is bad, we've lost the good policy we had
- Training becomes unstable (policy can diverge)

**PPO Solution:**
- Constrain how much the policy can change
- "Stay close to the old policy" (proximal)
- If new policy is too different, clip the update

### The PPO Objective

PPO maximizes the **clipped surrogate objective**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ (probability ratio)
- $A_t$ = advantage (how good was this action?)
- $\epsilon = 0.2$ (clipping parameter, typically)

**Intuition:**

```
If action was good (A > 0):
  - Want to increase π(a|s)
  - But clip increase at ratio = 1.2 (can't grow too much)

If action was bad (A < 0):
  - Want to decrease π(a|s)
  - But clip decrease at ratio = 0.8 (can't shrink too much)
```

**The clip function:**
$$\text{clip}(r, 1-\epsilon, 1+\epsilon) = \begin{cases}
1 - \epsilon & \text{if } r < 1 - \epsilon \\
r & \text{if } 1 - \epsilon \leq r \leq 1 + \epsilon \\
1 + \epsilon & \text{if } r > 1 + \epsilon
\end{cases}$$

### Complete PPO Loss

The total loss combines three terms:

$$L(\theta, \phi) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\phi) + c_2 S[\pi_\theta](s)$$

**1. Policy Loss** (Actor): $L^{\text{CLIP}}(\theta)$
- Encourages actions that gave good rewards
- Clipped to prevent large updates

**2. Value Loss** (Critic): $L^{\text{VF}}(\phi) = (V_\phi(s) - V^{\text{target}})^2$
- Mean squared error between predicted value and actual return
- Teaches critic to estimate returns accurately

**3. Entropy Bonus**: $S[\pi_\theta](s) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$
- Encourages exploration (prevents premature convergence to deterministic policy)
- Higher entropy = more uniform distribution = more exploration

**Coefficients:**
- $c_1 = 0.5$: Weight for value loss
- $c_2 = 0.01$: Weight for entropy bonus

### Training Procedure

**Collect Experience (Rollout):**
```
For K steps:
    Sample action: a_t ~ π_θ(·|s_t)
    Execute: s_{t+1}, r_t, done = env.step(a_t)
    Store: (s_t, a_t, r_t, log_prob_t, V(s_t))
```

**Compute Advantages:**
```
For each timestep:
    Compute returns: G_t = Σ γ^k r_{t+k}
    Compute advantage: A_t = G_t - V(s_t)
    Normalize advantages: A_t = (A_t - mean(A)) / (std(A) + ε)
```

**Update Networks (Multiple Epochs):**
```
For epoch = 1 to 4:  # Multiple passes over data
    For each mini-batch in experience:
        # Compute new policy and value
        π_new(a|s), V_new(s) = network(s)
        
        # Compute probability ratio
        r(θ) = π_new(a|s) / π_old(a|s)
        
        # Compute clipped objective
        L_clip = min(r(θ) A, clip(r(θ), 0.8, 1.2) A)
        
        # Compute value loss
        L_value = (V_new(s) - G)²
        
        # Compute entropy
        H = -Σ π_new(a|s) log π_new(a|s)
        
        # Total loss
        L = -L_clip + 0.5 L_value - 0.01 H
        
        # Gradient descent
        θ ← θ - α∇L
```

**Key Difference from DQN:**
- DQN: Sample single transitions, train immediately
- PPO: Collect batch of experience, train multiple epochs on it

## Hyperparameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Learning Rate** | 0.0003 | Slightly higher than DQN (PPO is more stable) |
| **Discount Factor** | 0.99 | Same as DQN; long-term planning |
| **GAE Lambda** | 0.95 | For advantage estimation (balances bias-variance) |
| **Clip Epsilon** | 0.2 | Trust region size (typical value) |
| **Value Coef** | 0.5 | Weight for value loss ($c_1$) |
| **Entropy Coef** | 0.01 | Weight for exploration bonus ($c_2$) |
| **Rollout Length** | 2048 | Steps to collect before update |
| **Batch Size** | 64 | Mini-batch size for SGD |
| **Epochs** | 4 | Number of passes through experience |
| **Optimizer** | Adam | Adaptive learning rate |

### Key Hyperparameter Explanations

**Clip Epsilon (0.2):**
- Defines trust region: policy can change by ±20%
- Too small (0.05): Very conservative, slow learning
- Too large (0.5): Allows destructive updates, unstable
- 0.2: Standard, works well across many environments

**Rollout Length (2048):**
- Collect 2048 steps before updating
- Longer: More stable estimates, but slower iteration
- Shorter: Faster iteration, but noisier estimates
- 2048: Good balance for on-policy methods

**Epochs (4):**
- Train on collected experience 4 times
- PPO can reuse data (unlike Q-Learning which is strictly off-policy)
- More epochs: Better sample efficiency, but risk overfitting
- 4: Standard compromise

**Entropy Coefficient (0.01):**
- Small bonus for exploration
- Decays naturally as policy becomes more confident
- Prevents premature convergence to suboptimal deterministic policy

## Generalized Advantage Estimation (GAE)

PPO uses GAE to compute advantages, which balances bias and variance:

$$A^{\text{GAE}}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

**Parameters:**
- $\gamma = 0.99$: Discount factor
- $\lambda = 0.95$: GAE parameter

**What GAE does:**
- $\lambda = 0$: Uses 1-step TD error (low variance, high bias)
- $\lambda = 1$: Uses full Monte Carlo returns (high variance, low bias)
- $\lambda = 0.95$: Balanced trade-off

**Intuition:** GAE smoothly interpolates between TD and MC, getting benefits of both.

## Expected Behavior by Grid Size

### 10×10 Grid: ✅ Excellent

**State Space:** ~10^9 states

**Expected Performance:**
- Episodes to convergence: 1500-2500
- Final average score: 20-30 apples
- Training time: 15-20 minutes (CPU)

**Advantages over DQN:**
- More stable learning curve (less oscillation)
- Better sample efficiency (fewer episodes needed)
- Higher final performance
- More consistent (lower variance across runs)

---

### 20×20 Grid: ✅ Excellent

**State Space:** ~10^40 states

**Expected Performance:**
- Episodes to convergence: 4000-6000
- Final average score: 30-50 apples
- Training time: 25-35 minutes (CPU)

**Advantages over DQN:**
- Significantly more stable
- Better at long-term planning (policy gradient bias)
- Handles sparse rewards better
- Less prone to catastrophic forgetting

---

### 30×30+ Grid: ✅ Best Among All Methods

**PPO shines on larger problems:**
- Most sample-efficient
- Most stable
- Best final performance

## Advantages of PPO

### ✅ Stability
The clipped objective prevents destructive policy updates. Training is smooth and reliable.

### ✅ Sample Efficiency
Reuses experience through multiple epochs. Typically needs 50-70% fewer samples than DQN.

**Example:**
- DQN: Needs 5000 episodes to converge
- PPO: Needs 2500 episodes to converge (same performance)

### ✅ Easy to Tune
Fewer hyperparameters, more robust to different settings. Works well with default values.

### ✅ Handles Sparse Rewards
Policy gradients work better than value-based methods when rewards are rare.

### ✅ Natural Exploration
Stochastic policy naturally explores. Entropy bonus encourages diversity.

### ✅ Direct Policy Learning
Learns what to do (policy) rather than evaluating what's good (Q-values). Often more intuitive.

### ✅ Works for Continuous Actions
Can easily extend to continuous action spaces (DQN cannot).

## Limitations of PPO

### ❌ On-Policy Method
Must collect new experience for each update. Cannot use old replay buffer like DQN.

**Impact:** Slightly less sample-efficient than off-policy methods in some cases.

### ❌ Computationally Complex
Training on rollouts with multiple epochs is more complex than DQN's single-step updates.

### ❌ Hyperparameter Sensitivity (Mild)
More hyperparameters than Q-Learning (clip ε, GAE λ, epochs, rollout length).

**But:** Less sensitive than vanilla policy gradients. Default values work well.

### ❌ Requires More Memory (During Training)
Must store entire rollout (2048 steps) in memory during training.

**vs. DQN:** DQN samples small batches (64) from replay buffer.

### ❌ Less Interpretable
Like DQN, neural network is a black box. Cannot inspect exact values.

## Comparison: DQN vs. PPO

| Aspect | DQN | PPO |
|--------|-----|-----|
| **Paradigm** | Value-based | Policy-based |
| **Learns** | Q(s,a) | π(a\|s), V(s) |
| **Policy Type** | Deterministic | Stochastic |
| **Data Usage** | Off-policy (replay buffer) | On-policy (fresh rollouts) |
| **Sample Efficiency** | Moderate | High |
| **Stability** | Moderate (can oscillate) | High (clipped updates) |
| **Convergence** | Slower, noisier | Faster, smoother |
| **Exploration** | ε-greedy (explicit) | Policy entropy (implicit) |
| **Action Spaces** | Discrete only | Discrete & continuous |
| **Training Time/Step** | Fast (single batch) | Slower (multiple epochs) |
| **Final Performance** | Good | Better |
| **Best For** | Q-learning purists, off-policy learning | Modern applications, best performance |

### Performance Comparison (Predicted)

**10×10 Grid:**

| Metric | DQN | PPO |
|--------|-----|-----|
| Episodes to Score 15 | 2000 | 1500 |
| Final Average Score | 18-25 | 22-30 |
| Training Time | 15 min | 18 min |
| Stability | Moderate variance | Low variance |

**20×20 Grid:**

| Metric | DQN | PPO |
|--------|-----|-----|
| Episodes to Score 25 | 5000 | 3500 |
| Final Average Score | 25-40 | 35-50 |
| Training Time | 35 min | 30 min |
| Stability | High variance | Low variance |

**Key Observation:** PPO typically achieves higher scores with fewer episodes and more stable training.

## Implementation Considerations

### Advantage Normalization

Always normalize advantages before computing loss:

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Why?** Prevents exploding gradients and stabilizes training.

### Clipping Safeguards

Implement both policy clipping and value clipping:

```python
# Policy clip
ratio = new_prob / old_prob
clip_ratio = torch.clamp(ratio, 1-ε, 1+ε)
loss_policy = -torch.min(ratio * advantages, clip_ratio * advantages)

# Value clip (optional but recommended)
value_pred_clipped = old_value + torch.clamp(value_pred - old_value, -ε, ε)
loss_value = max((value_pred - returns)², (value_pred_clipped - returns)²)
```

### Learning Rate Scheduling

Optionally decay learning rate during training:

```python
# Linear decay from 3e-4 to 3e-5
lr = 3e-4 - (3e-4 - 3e-5) * (epoch / total_epochs)
```

Can improve final performance by 5-10%.

## Common Pitfalls and Solutions

### Pitfall 1: Policy Collapse
**Symptom:** Policy becomes deterministic too early (always same action)

**Solution:**
- Increase entropy coefficient (0.01 → 0.05)
- Ensure entropy loss is negative in total loss formula
- Check that entropy is being computed correctly

### Pitfall 2: Value Function Not Learning
**Symptom:** Critic outputs constant value for all states

**Solution:**
- Increase value coefficient (0.5 → 1.0)
- Check that returns are computed correctly
- Verify value loss is being minimized (not maximized)

### Pitfall 3: Training is Too Slow
**Symptom:** 10,000 episodes with minimal improvement

**Solution:**
- Increase learning rate (3e-4 → 5e-4)
- Reduce rollout length (2048 → 1024)
- Increase batch size (64 → 128)

### Pitfall 4: Unstable Performance
**Symptom:** Performance improves then suddenly crashes

**Solution:**
- Decrease learning rate (3e-4 → 1e-4)
- Decrease clip epsilon (0.2 → 0.1)
- Reduce number of epochs (4 → 2)

## Why PPO is State-of-the-Art

PPO has become the **de facto standard** for many RL applications because:

1. **Robustness:** Works well across diverse problems with minimal tuning
2. **Performance:** Achieves highest final scores in most benchmarks
3. **Stability:** Smooth, reliable learning curves
4. **Simplicity:** Easier to implement than other policy gradient methods (TRPO, A3C)
5. **Versatility:** Handles discrete and continuous actions
6. **Industry Adoption:** Used in robotics, game AI, autonomous systems

**Notable Successes:**
- OpenAI Five (Dota 2)
- OpenAI's robotic hand (Rubik's cube solving)
- Many DeepMind projects
- Unity ML-Agents framework

## Summary

**PPO** represents the current state-of-the-art in reinforcement learning:

**Core Innovation:** Clipped objective ensures safe, stable policy updates

**Actor-Critic Framework:** Policy (actor) and value (critic) learned jointly with shared CNN

**Key Advantages:**
- Most stable training (vs. DQN, vanilla PG)
- Best sample efficiency (vs. DQN)
- Highest final performance (vs. tabular, DQN)
- Easy to tune (vs. TRPO, A3C)

**For our Snake project:**
- ✅ Performs best on all grid sizes (5×5, 10×10, 20×20)
- ✅ Most stable learning curves
- ✅ Fastest convergence (fewest episodes needed)
- ✅ Demonstrates modern deep RL capabilities

**The Evolution:**
1. **Tabular Q-Learning:** Simple, works on 5×5, fails on 10×10
2. **DQN:** Handles 10×10, 20×20 through function approximation
3. **PPO:** Best performance, most stable, current state-of-the-art

**Trade-off:**
- More complex than tabular methods
- More hyperparameters than DQN
- But: Best performance with reasonable computational cost

---

## Key Takeaways

1. **Policy gradients learn policies directly** - More natural for many tasks
2. **Actor-Critic reduces variance** - Critic helps stabilize policy learning  
3. **Clipped objective ensures safety** - Prevents destructive updates
4. **PPO is the modern standard** - Best balance of performance and simplicity
5. **Sample efficiency matters** - PPO needs fewer episodes than DQN
6. **Stochastic policies enable exploration** - No need for ε-greedy
7. **Shared CNN backbone is efficient** - Learn features once, use twice

PPO demonstrates why modern deep RL has moved toward policy gradient methods: better performance, more stability, and wider applicability than value-based approaches.

**This completes the progression:** Tabular → DQN → PPO shows the evolution of RL from classical to modern techniques.