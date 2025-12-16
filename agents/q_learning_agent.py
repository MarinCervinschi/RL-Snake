import pickle
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from core.interfaces import IAgent
from game.entities import Action, State


class QLearningAgent(IAgent):
    """
    Tabular Q-Learning with dictionary-based Q-table.

    Uses state hashing to create a sparse Q-table that only stores
    visited states.
    """

    def __init__(
        self,
        grid_size: int = 5,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):
        """
        Initialize Tabular Q-Learning agent.

        Args:
            grid_size: Size of the game grid
            learning_rate: Alpha (how much to update Q-values)
            discount_factor: Gamma (importance of future rewards)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate per episode
            min_epsilon: Minimum exploration rate
        """
        super().__init__(grid_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table: dictionary mapping state_key -> [Q(s,a) for each action]
        # Key: tuple of (snake_positions, food_position, direction)
        # Value: np.array of shape (4,) for 4 absolute actions
        self.q_table: Dict[Tuple, np.ndarray] = {}

        self.updates_performed = 0

    def get_action(self, state: State) -> Action:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current game state

        Returns:
            Selected action
        """
        state_key = self._state_to_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(4, dtype=np.float32)

        if random.random() < self.epsilon:
            # Exploration: Random action
            return random.choice(list(Action))
        else:
            # Exploitation: Best known action
            q_values = self.q_table[state_key]
            action_idx = np.argmax(q_values)
            return Action(action_idx)

    def train(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """
        Update Q-table using Bellman equation.

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        action_idx = action.value

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(4, dtype=np.float32)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(4, dtype=np.float32)

        current_q = self.q_table[state_key][action_idx]

        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: reward + discounted max future Q
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * max_next_q

        self.q_table[state_key][action_idx] = current_q + self.learning_rate * (
            target_q - current_q
        )

        self.updates_performed += 1

        if done and self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def _state_to_key(self, state: State) -> Tuple:
        """
        Convert State to hashable key for dictionary.

        Uses position-based representation:
        - Snake positions as tuple of tuples
        - Food position as tuple
        - Direction as integer

        Args:
            state: Grid state to convert

        Returns:
            Hashable tuple representing state
        """
        return state.to_position_tuple()

    def save(self, filepath: str = "models/tabular_q_learning.pkl") -> None:
        """
        Save Q-table and agent state to file.

        Args:
            filepath: Path to save file
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "q_table": self.q_table,
            "grid_size": self.grid_size,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "updates_performed": self.updates_performed,
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"💾 Model saved to {filepath}")
        print(f"   Q-table size: {len(self.q_table):,} entries")
        print(f"   Updates performed: {self.updates_performed:,}")
        print(f"   Memory: ~{self._estimate_memory_mb():.2f} MB")

    def load(
        self, filepath: str = "models/tabular_q_learning.pkl", play: bool = False
    ) -> None:
        """
        Load Q-table and agent state from file.

        Args:
            filepath: Path to load file
        """
        load_path = Path(filepath)

        if not load_path.exists():
            print(f"⚠️  No saved model found at {filepath}")
            print(f"   Starting with empty Q-table")
            if play:
                raise FileNotFoundError(f"No saved model found at {filepath}")
            return

        with open(load_path, "rb") as f:
            save_dict = pickle.load(f)

        self.q_table = save_dict["q_table"]
        self.grid_size = save_dict["grid_size"]
        self.epsilon = save_dict["epsilon"]
        self.learning_rate = save_dict["learning_rate"]
        self.discount_factor = save_dict["discount_factor"]
        self.updates_performed = save_dict.get("updates_performed", 0)

        print(f"✅ Model loaded from {filepath}")
        print(f"   States in Q-table: {len(self.q_table):,}")
        print(f"   Current epsilon: {self.epsilon:.4f}")
        print(f"   Memory: ~{self._estimate_memory_mb():.2f} MB")

    def _estimate_memory_mb(self) -> float:
        """
        Estimate memory usage of Q-table in MB.

        Returns:
            Approximate memory in MB
        """
        # Each entry: tuple key + numpy array (4 floats)
        # Rough estimate: ~100 bytes per entry (tuple overhead + array)
        bytes_per_entry = 100 + 4 * 4  # tuple + 4 float32s
        total_bytes = len(self.q_table) * bytes_per_entry
        return total_bytes / (1024 * 1024)

    def _check_configuration(self) -> None:
        """Check for suboptimal configurations and warn user."""
        if self.grid_size > 7:
            print(
                f"⚠️  WARNING: Grid size {self.grid_size}×{self.grid_size} is very large for tabular methods!"
            )
            print(
                f"   State space will grow exponentially. Consider using DQN instead."
            )
            print(f"   Expected behavior: Poor performance, never converges.")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"QLearningAgent("
            f"grid_size={self.grid_size}, "
            f"q_table_size={len(self.q_table)}, "
            f"epsilon={self.epsilon:.4f}, "
        )
