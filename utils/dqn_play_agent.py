from pathlib import Path

import torch
import torch.nn as nn

from game.entities import Action, State


class ConvQNetwork(nn.Module):
    """CNN Q-Network for DQN."""

    def __init__(self, grid_size: int, num_actions: int = 3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        q_values = self.fc(flat)
        return q_values


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


def load_dqn_agent(grid_size: int = 10, model_path: str | None = None):
    """Load DQN agent."""
    if model_path is None:
        model_path = "models/dqn_cnn.pkl"

    if grid_size != 10:
        raise ValueError("DQN agent only supports grid size of 10")

    agent = DQNAgent(grid_size)
    agent.load(model_path)
    return agent
