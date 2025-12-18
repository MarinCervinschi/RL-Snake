from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from core.interfaces import IAgent
from game.entities import Action, State


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared CNN backbone.

    Architecture:
        Input: (3, H, W) - 3-channel grid

        Shared CNN Backbone:
            Conv2D (3→32) + Tanh
            Conv2D (32→64) + Tanh
            Conv2D (64→64) + Tanh
            Flatten
            Linear (64*H*W → 512) + Tanh

        Actor Head:
            Linear (512 → 256) + Tanh
            Linear (256 → 4) + Softmax
            Output: π(a|s) - action probabilities

        Critic Head:
            Linear (512 → 256) + Tanh
            Linear (256 → 1)
            Output: V(s) - state value
    """

    def __init__(self, grid_size: int, num_actions: int = 4):
        """
        Initialize Actor-Critic network.

        Args:
            grid_size: Size of the game grid
            num_actions: Number of actions (4 for absolute directions)
        """
        super().__init__()

        self.grid_size = grid_size
        self.num_actions = num_actions

        # Shared CNN backbone for spatial feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Calculate flattened size
        self.flat_size = 64 * grid_size * grid_size

        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.Tanh(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, num_actions),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic.

        Args:
            x: Input tensor of shape (batch, 3, H, W)

        Returns:
            Tuple of (action_probs, state_values):
                - action_probs: shape (batch, num_actions) - policy distribution
                - state_values: shape (batch, 1) - state value estimates
        """
        # Shared feature extraction
        conv_features = self.shared_conv(x)
        flat = conv_features.view(conv_features.size(0), -1)
        shared_features = self.shared_fc(flat)

        # Actor: policy distribution
        action_logits = self.actor(shared_features)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Critic: state value
        state_values = self.critic(shared_features)

        return action_probs, state_values

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Used during training to compute all required quantities.

        Args:
            x: State tensor (batch, 3, H, W)
            action: Optional action tensor for computing log_prob

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        action_probs, value = self.forward(x)

        # Create categorical distribution
        dist = Categorical(action_probs)

        # Sample action if not provided
        if action is None:
            action = dist.sample()

        # Compute log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization for Tanh."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class RolloutBuffer:
    """
    Buffer for storing rollout experience for PPO training.

    Stores trajectories and computes advantages using GAE.
    """

    def __init__(self):
        """Initialize empty buffer."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """
        Store a transition.

        Args:
            state: State array (3, H, W)
            action: Action index
            log_prob: Log probability of action
            reward: Reward received
            value: Value estimate V(s)
            done: Whether episode terminated
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self) -> Tuple:
        """
        Get all stored data as numpy arrays.

        Returns:
            Tuple of (states, actions, log_probs, rewards, values, dones)
        """
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
        )

    def clear(self) -> None:
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.states)


class PPOAgent(IAgent):
    """
    Proximal Policy Optimization agent.

    Key features:
    - Actor-Critic architecture with shared CNN
    - Clipped surrogate objective for stable updates
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs on collected rollouts
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        grid_size: int = 20,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        rollout_length: int = 2048,
        batch_size: int = 64,
        epochs: int = 4,
        max_grad_norm: float = 0.5,
    ):
        """
        Initialize PPO agent.

        Args:
            grid_size: Size of the game grid
            learning_rate: Learning rate for optimizer
            discount_factor: Gamma (discount factor)
            gae_lambda: Lambda for GAE
            clip_epsilon: Clipping parameter for PPO objective
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            rollout_length: Steps to collect before update
            batch_size: Mini-batch size for SGD
            epochs: Number of epochs to train on each rollout
            max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__(grid_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  PPO using device: {self.device}")

        # Actor-Critic network
        self.policy = ActorCriticNetwork(grid_size).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training statistics
        self.steps = 0
        self.updates = 0
        self.episodes_trained = 0
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_losses: List[float] = []

    def get_action(self, state: State) -> Action:
        """
        Select action from policy.

        Args:
            state: Current game state

        Returns:
            Selected action
        """
        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)

        # Sample from policy distribution
        dist = Categorical(action_probs)
        action_idx = dist.sample().item()

        return Action(action_idx)

    def train(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ) -> None:
        """
        Store transition in rollout buffer and train when buffer is full.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Convert state to array
        state_array = state.to_tensor()

        # Get log_prob and value for this transition
        state_tensor = self._state_to_tensor(state)
        action_tensor = torch.tensor([action.value], device=self.device)

        with torch.no_grad():
            _, log_prob, _, value = self.policy.get_action_and_value(
                state_tensor, action_tensor
            )

        # Store in buffer
        self.buffer.push(
            state_array, action.value, log_prob.item(), reward, value.item(), done
        )

        self.steps += 1

        # Train when buffer reaches rollout_length
        if len(self.buffer) >= self.rollout_length:
            self._update()
            self.buffer.clear()

        # Track episode completion
        if done:
            self.episodes_trained += 1

    def _update(self) -> None:
        """
        Perform PPO update using collected rollout.
        """
        # Get data from buffer
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()

        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)

        # Compute returns (targets for value function)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Train for multiple epochs
        for epoch in range(self.epochs):
            # Create mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]

                # Forward pass
                _, new_log_probs, entropy, values_pred = (
                    self.policy.get_action_and_value(batch_states, batch_actions)
                )

                # Compute ratio (pi_new / pi_old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = values_pred.squeeze()
                value_loss = ((values_pred - batch_returns) ** 2).mean()

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )

                self.optimizer.step()

                # Record losses
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy_loss.item())

        self.updates += 1

    def _compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags

        Returns:
            Array of advantages
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]

            # TD error
            delta = (
                rewards[t]
                + self.discount_factor * next_value * (1 - dones[t])
                - values[t]
            )

            # GAE
            last_gae = (
                delta
                + self.discount_factor * self.gae_lambda * (1 - dones[t]) * last_gae
            )
            advantages[t] = last_gae

        return advantages

    def _state_to_tensor(self, state: State) -> torch.Tensor:
        """
        Convert State to tensor.

        Args:
            state: Grid state

        Returns:
            Tensor of shape (1, 3, H, W)
        """
        state_array = state.to_tensor()
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
        return state_tensor.to(self.device)

    def save(self, filepath: str = "models/ppo.pkl") -> None:
        """
        Save model and training state.

        Args:
            filepath: Path to save file
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grid_size": self.grid_size,
            "steps": self.steps,
            "updates": self.updates,
            "episodes_trained": self.episodes_trained,
            "policy_losses": self.policy_losses[-1000:],
            "value_losses": self.value_losses[-1000:],
            "entropy_losses": self.entropy_losses[-1000:],
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "rollout_length": self.rollout_length,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
            },
        }

        torch.save(save_dict, save_path)

        print(f"💾 Model saved to {filepath}")
        print(f"   Episodes trained: {self.episodes_trained}")
        print(f"   Steps: {self.steps:,}")
        print(f"   Updates: {self.updates}")

    def load(self, filepath: str = "models/ppo.pkl", play: bool = False) -> None:
        """
        Load model and training state.

        Args:
            filepath: Path to load file
        """
        load_path = Path(filepath)

        if not load_path.exists():
            print(f"⚠️  No saved model found at {filepath}")
            print(f"   Starting with randomly initialized network")
            if play:
                raise FileNotFoundError(f"No saved model found at {filepath}")
            return

        checkpoint = torch.load(load_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.grid_size = checkpoint["grid_size"]
        self.steps = checkpoint["steps"]
        self.updates = checkpoint["updates"]
        self.episodes_trained = checkpoint.get("episodes_trained", 0)
        self.policy_losses = checkpoint.get("policy_losses", [])
        self.value_losses = checkpoint.get("value_losses", [])
        self.entropy_losses = checkpoint.get("entropy_losses", [])

        print(f"✅ Model loaded from {filepath}")
        print(f"   Episodes trained: {self.episodes_trained}")
        print(f"   Steps: {self.steps:,}")
        print(f"   Updates: {self.updates}")

    def get_value(self, state: State) -> float:
        """
        Get value estimate for a state.

        Args:
            state: State to evaluate

        Returns:
            Value estimate
        """
        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            _, value = self.policy(state_tensor)

        return value.item()

    def get_action_probs(self, state: State) -> np.ndarray:
        """
        Get action probabilities for a state.

        Args:
            state: State to evaluate

        Returns:
            Array of action probabilities
        """
        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)

        return action_probs.cpu().numpy().flatten()

    def _check_configuration(self) -> None:
        if self.grid_size < 8:
            print("ℹ️  INFO: Using neural network on small grid.")
            print(
                f"   Grid: {self.grid_size}×{self.grid_size} is small, tabular might be sufficient."
            )
            print(f"   This is fine for testing, but overkill for actual training.\n")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PPOAgent("
            f"grid_size={self.grid_size}, "
            f"steps={self.steps:,}, "
            f"updates={self.updates}, "
            f"buffer_size={len(self.buffer)})"
        )
