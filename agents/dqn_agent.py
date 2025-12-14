import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import DISCOUNT_FACTOR, EPSILON, EPSILON_DECAY, MIN_EPSILON
from core.interfaces import IAgent
from game.entities import Action, State


class QNetwork(nn.Module):
    """
    Neural Network that approximates Q(s, a).

    Architecture:
    Input (11) → Hidden (128) → Hidden (128) → Output (3)
    """

    def __init__(self, input_size=11, hidden_size=128, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Stores past experiences for training.
    Breaks temporal correlation by sampling randomly.
    """

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent(IAgent):
    """
    Deep Q-Network Agent using Experience Replay and Target Network.
    """

    EPISODES: int = 3000

    def __init__(
        self,
        learning_rate=0.001,
        gamma=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
        batch_size=64,
        buffer_size=100_000,
        target_update_freq=1000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN using device: {self.device}")

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Experience Replay
        self.memory = ReplayBuffer(buffer_size)

        # Training state
        self.steps = 0
        self.losses = []

        self.load()

    def get_action(self, state: State) -> Action:
        """
        Epsilon-greedy action selection.
        """
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(list(Action))

        # Exploitation
        state_tensor = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        action_idx = q_values.argmax().item()
        return Action(action_idx)

    def train(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ) -> None:
        """
        Store experience and train on a batch if buffer is ready.
        """
        # Store transition
        self.memory.push(
            state.to_vector(), action.value, reward, next_state.to_vector(), done
        )

        # Train only if we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values (using target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.steps += 1

        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon after each step (more gradual than per-episode)
        if done and self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath="models/dqn_model.pth"):
        """Save the trained model."""
        import os

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load(self, filepath="models/dqn_model.pth"):
        """Load a trained model."""

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
        except FileNotFoundError:
            print(f"No model found at {filepath}. Will start fresh.")
            return

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        print(f"Model loaded from {filepath}")
