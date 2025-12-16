"""
Deep Q-Network (DQN) Agent with CNN architecture.

Uses convolutional neural networks to process the 3-channel grid state
and learn Q-values through experience replay and target networks.
"""

import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.entities import Action, State
from core.interfaces import IAgent


class ConvQNetwork(nn.Module):
    """
    Convolutional Q-Network for processing grid states.

    Architecture:
        Input: (3, H, W) - 3-channel grid (head, body, food)
        Conv2D (3→32) + Tanh
        Conv2D (32→64) + Tanh
        Conv2D (64→64) + Tanh
        Flatten
        Linear (64*H*W → 512) + Tanh
        Linear (512 → 4) - Q-values for 4 actions
    """

    def __init__(self, grid_size: int, num_actions: int = 4):
        """
        Initialize Q-Network.

        Args:
            grid_size: Size of the game grid (H = W)
            num_actions: Number of actions (4 for absolute directions)
        """
        super().__init__()

        self.grid_size = grid_size
        self.num_actions = num_actions

        # Convolutional layers for spatial feature extraction
        self.conv = nn.Sequential(
            # Conv1: 3 channels → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            # Conv2: 32 channels → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            # Conv3: 64 channels → 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Calculate flattened size after convolutions
        # With padding=1, spatial dimensions are preserved
        self.flat_size = 64 * grid_size * grid_size

        # Fully connected layers for decision making
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 512), nn.Tanh(), nn.Linear(512, num_actions)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W)

        Returns:
            Q-values of shape (batch, num_actions)
        """
        # Convolutional feature extraction
        features = self.conv(x)

        # Flatten spatial dimensions
        flat = features.view(features.size(0), -1)

        # Fully connected layers
        q_values = self.fc(flat)

        return q_values

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization for Tanh."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores transitions as (state, action, reward, next_state, done)
    and allows random sampling to break temporal correlation.
    """

    def __init__(self, capacity: int = 100_000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer: Deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition.

        Args:
            state: State array (3, H, W)
            action: Action index
            reward: Reward received
            next_state: Next state array (3, H, W)
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent(IAgent):
    """
    Deep Q-Network agent with experience replay and target network.

    Key features:
    - CNN architecture for spatial reasoning
    - Experience replay buffer
    - Separate target network for stable training
    - ε-greedy exploration
    """

    def __init__(
        self,
        grid_size: int = 20,
        learning_rate: float = 0.0001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 1000,
    ):
        """
        Initialize DQN agent.

        Args:
            grid_size: Size of the game grid
            learning_rate: Learning rate for optimizer
            discount_factor: Gamma (discount factor)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate per episode
            min_epsilon: Minimum exploration rate
            batch_size: Size of training batches
            buffer_size: Capacity of replay buffer
            target_update_freq: Steps between target network updates
        """
        super().__init__(grid_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device configuration (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  DQN using device: {self.device}")

        # Q-network and target network
        self.q_network = ConvQNetwork(grid_size).to(self.device)
        self.target_network = ConvQNetwork(grid_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Optimizer (Adam with default parameters)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Training statistics
        self.steps = 0
        self.losses: List[float] = []
        self.episodes_trained = 0

    def get_action(self, state: State) -> Action:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current game state

        Returns:
            Selected action
        """
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(list(Action))

        # Exploitation: best action from Q-network
        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        action_idx = q_values.argmax().item()
        return Action(action_idx)

    def train(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ) -> None:
        """
        Store transition and train on batch if ready.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Convert states to arrays
        state_array = state.to_tensor()
        next_state_array = next_state.to_tensor()

        # Store transition in replay buffer
        self.memory.push(state_array, action.value, reward, next_state_array, done)

        # Only train if we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze()

        # Compute target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t)
            next_q = next_q_values.max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.discount_factor * next_q

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Record loss
        self.losses.append(loss.item())
        self.steps += 1

        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon after episode ends
        if done and self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.episodes_trained += 1

    def _state_to_tensor(self, state: State) -> torch.Tensor:
        """
        Convert State to tensor.

        Args:
            state: Grid state

        Returns:
            Tensor of shape (1, 3, H, W) ready for network
        """
        state_array = state.to_tensor()  # (3, H, W)
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)  # (1, 3, H, W)
        return state_tensor.to(self.device)

    def save(self, filepath: str = "models/dqn_cnn.pkl") -> None:
        """
        Save model and training state.

        Args:
            filepath: Path to save file
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grid_size": self.grid_size,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes_trained": self.episodes_trained,
            "losses": self.losses[-1000:],  # Save last 1000 losses
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "batch_size": self.batch_size,
                "target_update_freq": self.target_update_freq,
            },
        }

        torch.save(save_dict, save_path)

        print(f"💾 Model saved to {filepath}")
        print(f"   Episodes trained: {self.episodes_trained}")
        print(f"   Steps: {self.steps:,}")
        print(f"   Current epsilon: {self.epsilon:.4f}")
        print(f"   Replay buffer: {len(self.memory):,} transitions")

    def load(self, filepath: str = "models/dqn_cnn.pkl") -> None:
        """
        Load model and training state.

        Args:
            filepath: Path to load file
        """
        load_path = Path(filepath)

        if not load_path.exists():
            print(f"⚠️  No saved model found at {filepath}")
            print(f"   Starting with randomly initialized network")
            return

        checkpoint = torch.load(load_path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.grid_size = checkpoint["grid_size"]
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes_trained = checkpoint.get("episodes_trained", 0)
        self.losses = checkpoint.get("losses", [])

        print(f"✅ Model loaded from {filepath}")
        print(f"   Episodes trained: {self.episodes_trained}")
        print(f"   Steps: {self.steps:,}")
        print(f"   Current epsilon: {self.epsilon:.4f}")

    def get_q_values(self, state: State) -> np.ndarray:
        """
        Get Q-values for a state (for visualization/debugging).

        Args:
            state: State to query

        Returns:
            Array of Q-values for each action
        """
        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.cpu().numpy().flatten()

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
            f"DQNAgent("
            f"grid_size={self.grid_size}, "
            f"epsilon={self.epsilon:.4f}, "
            f"steps={self.steps:,}, "
            f"buffer_size={len(self.memory):,})"
        )
