from abc import ABC, abstractmethod
from game.entities import State, Action, Point
from typing import Optional


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
