"""
Agent loader implementations.

These are lightweight wrappers that load trained models for inference.
For training, use the notebooks directly.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from game.entities import Action, State


# ============================================================================
# Tabular Q-Learning Loader
# ============================================================================


class TabularQLearningAgent:
    """Minimal wrapper for loading trained Q-Learning agents."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.q_table: Dict[Tuple, np.ndarray] = {}
        self.epsilon = 0.0  # No exploration during inference

    def get_action(self, state: State) -> Action:
        """Select best action from Q-table."""
        state_key = state.to_position_tuple()

        if state_key not in self.q_table:
            # Unseen state - random action
            return random.choice(list(Action))

        q_values = self.q_table[state_key]
        return Action(int(np.argmax(q_values)))

    def load(self, filepath: str):
        """Load Q-table from file."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        self.q_table = save_dict["q_table"]
        self.grid_size = save_dict["grid_size"]
        print(f"✅ Loaded Tabular Q-Learning model")
        print(f"   Q-table size: {len(self.q_table):,} states")


def load_tabular_agent(grid_size: int, model_path: str | None = None):
    """Load tabular Q-learning agent."""
    if model_path is None:
        model_path = "models/tabular_q_learning.pkl"

    agent = TabularQLearningAgent(grid_size)
    agent.load(model_path)
    return agent


# ============================================================================
# DQN Loader
# ============================================================================


class ConvQNetwork(nn.Module):
    """CNN Q-Network for DQN."""

    def __init__(self, grid_size: int, num_actions: int = 4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.flat_size = 64 * grid_size * grid_size
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.Tanh(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        return self.fc(flat)


class DQNAgent:
    """Minimal wrapper for loading trained DQN agents."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = ConvQNetwork(grid_size).to(self.device)
        self.epsilon = 0.0  # No exploration

    def get_action(self, state: State) -> Action:
        """Select best action from Q-network."""
        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return Action(int(q_values.argmax().item()))

    def _state_to_tensor(self, state: State) -> torch.Tensor:
        state_array = state.to_tensor()
        return torch.FloatTensor(state_array).unsqueeze(0).to(self.device)

    def load(self, filepath: str):
        """Load Q-network weights."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.q_network.eval()

        print(f"✅ Loaded DQN model")


def load_dqn_agent(grid_size: int, model_path: str | None = None):
    """Load DQN agent."""
    if model_path is None:
        model_path = "models/dqn_cnn.pkl"

    agent = DQNAgent(grid_size)
    agent.load(model_path)
    return agent


# ============================================================================
# PPO Loader
# ============================================================================


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, grid_size: int, num_actions: int = 4):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.flat_size = 64 * grid_size * grid_size
        self.shared_fc = nn.Sequential(nn.Linear(self.flat_size, 512), nn.Tanh())

        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        conv_features = self.shared_conv(x)
        flat = conv_features.view(conv_features.size(0), -1)
        shared = self.shared_fc(flat)

        action_logits = self.actor(shared)
        action_probs = torch.softmax(action_logits, dim=-1)
        state_values = self.critic(shared)

        return action_probs, state_values


class PPOAgent:
    """Minimal wrapper for loading trained PPO agents."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCriticNetwork(grid_size).to(self.device)

    def get_action(self, state: State) -> Action:
        """Sample action from policy."""
        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)

        dist = Categorical(action_probs)
        return Action(int(dist.sample().item()))

    def _state_to_tensor(self, state: State):
        state_array = state.to_tensor()
        return torch.FloatTensor(state_array).unsqueeze(0).to(self.device)

    def load(self, filepath: str):
        """Load policy network weights."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state"])
        self.policy.eval()

        print(f"✅ Loaded PPO model")


def load_ppo_agent(grid_size: int, model_path: str | None = None):
    """Load PPO agent."""
    if model_path is None:
        model_path = "models/ppo.pkl"

    agent = PPOAgent(grid_size)
    agent.load(model_path)
    return agent