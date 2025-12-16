from abc import ABC, abstractmethod
from typing import Optional

from game.entities import Action, Point, State


class IRenderer(ABC):
    @abstractmethod
    def render(
        self, snake: list, food: Optional[Point], score: int, record: dict
    ) -> None:
        """Renders the current game state."""
        pass

    def close(self) -> None:
        """Cleans up any resources used by the renderer."""
        pass


class IAgent(ABC):

    def __init__(self, grid_size: int):
        """
        Initialize agent.

        Args:
            grid_size: Size of the game grid
        """
        self.grid_size = grid_size
        self._check_configuration()

    @abstractmethod
    def get_action(self, state: State) -> Action:
        """Decides the next action based on the current state."""
        pass

    @abstractmethod
    def train(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ) -> None:
        """Updates the agent's knowledge based on the transition."""
        pass

    @abstractmethod
    def save(self, path: Optional[str] = None) -> None:
        """Saves the agent's model or parameters to disk."""
        pass

    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        """Loads the agent's model or parameters from disk."""
        pass

    @abstractmethod
    def _check_configuration(self) -> None:
        """Optional method to warn about suboptimal configurations."""
        pass


class IMetricsLogger(ABC):
    @abstractmethod
    def record_episode(self, episode: int, score: int, steps: int):
        """
        Saves data for a single finished game.
        Args:
            episode: Current game number
            score: Total apples eaten
            steps: Total frames survived
        """
        pass

    @abstractmethod
    def plot(self):
        """Generates and saves/shows the graphs."""
        pass
