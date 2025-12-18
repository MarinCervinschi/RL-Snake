import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.entities import Action, State


class ConvQNetwork(nn.Module):
    """
    Convolutional Q-Network for processing grid states.

    Input: (3, H, W)
    Output: Q-values for each action
    """

    def __init__(self, grid_size: int, num_actions: int = 4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.flat_size = 64 * grid_size * grid_size

        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent tuned for Snake on a 10x10 grid.

    Key improvements:
    - Step-based epsilon decay
    - Exploration warmup
    - Huber loss
    - Lower learning rate
    """

    def __init__(
        self,
        grid_size: int = 10,
        learning_rate: float = 0.0001,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 200_000,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 1_000,
        warmup_steps: int = 20_000,
    ):
        self.grid_size = grid_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")

        # Hyperparameters
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.warmup_steps = warmup_steps
        self.epsilon = epsilon_start

        # Networks
        self.q_network = ConvQNetwork(grid_size).to(self.device)
        self.target_network = ConvQNetwork(grid_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        self.memory = ReplayBuffer(buffer_size)

        # Training stats
        self.steps = 0
        self.episodes_trained = 0
        self.losses: List[float] = []

        # self.load()

    def get_action(self, state: State) -> Action:
        # Warmup: pure exploration
        if self.steps < self.warmup_steps:
            return random.choice(list(Action))

        if random.random() < self.epsilon:
            return random.choice(list(Action))

        state_t = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return Action(q_values.argmax().item())

    def train(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ):
        state_arr = state.to_tensor()
        next_state_arr = next_state.to_tensor()

        self.memory.push(state_arr, action.value, reward, next_state_arr, done)

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        q_values = self.q_network(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.steps += 1

        # Step-based epsilon decay
        progress = min(1.0, self.steps / self.epsilon_decay_steps)
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end),
        )

        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if done:
            self.episodes_trained += 1

    def _state_to_tensor(self, state: State) -> torch.Tensor:
        arr = state.to_tensor()
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    def save(self, filepath: str = "models/dqn_snake_10x10.pkl"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q": self.q_network.state_dict(),
                "target": self.target_network.state_dict(),
                "opt": self.optimizer.state_dict(),
                "steps": self.steps,
                "episodes": self.episodes_trained,
                "epsilon": self.epsilon,
            },
            filepath,
        )
        print(f"💾 Saved model to {filepath}")
        print(f"   Episodes trained: {self.episodes_trained}")
        print(f"   Steps: {self.steps:,}")
        print(f"   Current epsilon: {self.epsilon:.4f}")
        print(f"   Replay buffer: {len(self.memory):,} transitions")

    def load(self, filepath: str = "models/dqn_snake_10x10.pkl"):
        path = Path(filepath)
        if not path.exists():
            print("⚠️  No saved model found. Starting fresh.")
            return

        data = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(data["q"])
        self.target_network.load_state_dict(data["target"])
        self.optimizer.load_state_dict(data["opt"])
        self.steps = data.get("steps", 0)
        self.episodes_trained = data.get("episodes", 0)
        self.epsilon = data.get("epsilon", self.epsilon)

        print(f"✅ Loaded model from {filepath}")
        print(f"   Episodes trained: {self.episodes_trained}")
        print(f"   Steps: {self.steps:,}")
        print(f"   Current epsilon: {self.epsilon:.4f}")

    def __repr__(self):
        return (
            f"DQNAgent(grid={self.grid_size}, steps={self.steps}, "
            f"episodes={self.episodes_trained}, epsilon={self.epsilon:.3f})"
        )

    def _check_configuration(self) -> None:
        if self.grid_size < 8:
            print("ℹ️  INFO: Using neural network on small grid.")
            print(
                f"   Grid: {self.grid_size}×{self.grid_size} is small, tabular might be sufficient."
            )
            print(f"   This is fine for testing, but overkill for actual training.\n")
