from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np


class Action(Enum):
    """Absolute directional actions."""

    UP = 0  # Move in -y direction (decrease row index)
    RIGHT = 1  # Move in +x direction (increase column index)
    DOWN = 2  # Move in +y direction (increase row index)
    LEFT = 3  # Move in -x direction (decrease column index)


class Direction(Enum):
    """Current direction the snake is facing."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @staticmethod
    def from_action(action: Action) -> Direction:
        """Convert Action to Direction (they map 1:1 in absolute system)."""
        return Direction(action.value)

    def opposite(self) -> Direction:
        """Get the opposite direction (for invalid action detection)."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]

    def to_vector(self) -> Tuple[int, int]:
        """Convert direction to (dx, dy) movement vector."""
        vectors = {
            Direction.UP: (0, -1),  # Move up (decrease y)
            Direction.RIGHT: (1, 0),  # Move right (increase x)
            Direction.DOWN: (0, 1),  # Move down (increase y)
            Direction.LEFT: (-1, 0),  # Move left (decrease x)
        }
        return vectors[self]


Point = namedtuple("Point", "x, y")


@dataclass
class State:
    """
    Full MDP state representation using 3-channel grid.

    Grid Channels:
        Channel 0: Snake head (1.0 at head position, 0.0 elsewhere)
        Channel 1: Snake body (1.0 at body segments, 0.0 elsewhere)
        Channel 2: Food (1.0 at food position, 0.0 elsewhere)

    Direction: Current heading (needed to prevent 180° turns)

    Shape: (grid_size, grid_size, 3)
    """

    grid_size: int
    grid: np.ndarray  # Shape: (H, W, 3)
    direction: Direction

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
        self.direction = Direction.RIGHT  # Default initial direction

    @classmethod
    def from_game_state(
        cls,
        grid_size: int,
        snake_positions: List[Point],
        food_position: Point,
        direction: Direction,
    ) -> State:
        """
        Create State from game components.

        Args:
            grid_size: Size of the game grid
            snake_positions: List of Points [head, body[0], body[1], ...]
            food_position: Point where food is located
            direction: Current direction snake is facing

        Returns:
            State with populated grid channels
        """
        state = cls(grid_size)
        state.direction = direction

        # Clear grid
        state.grid.fill(0.0)

        # Channel 0: Snake head (first position)
        if snake_positions:
            head = snake_positions[0]
            if 0 <= head.x < grid_size and 0 <= head.y < grid_size:
                state.grid[head.y, head.x, 0] = 1.0

        # Channel 1: Snake body (remaining positions)
        for segment in snake_positions[1:]:
            if 0 <= segment.x < grid_size and 0 <= segment.y < grid_size:
                state.grid[segment.y, segment.x, 1] = 1.0

        # Channel 2: Food
        if 0 <= food_position.x < grid_size and 0 <= food_position.y < grid_size:
            state.grid[food_position.y, food_position.x, 2] = 1.0

        return state

    def to_tensor(self) -> np.ndarray:
        """
        Convert to tensor format for neural networks.

        Returns:
            np.ndarray with shape (3, H, W) - channels first
        """
        # Transpose from (H, W, 3) to (3, H, W)
        return np.transpose(self.grid, (2, 0, 1))

    def to_position_tuple(self) -> Tuple:
        """
        Alternative hash representation using position tuples.
        More efficient than to_hash() for dictionary keys.

        Returns:
            Tuple of (snake_positions, food_position, direction)
        """
        # Extract head position
        head_positions = np.argwhere(self.grid[:, :, 0] == 1.0)
        head = tuple(head_positions[0]) if len(head_positions) > 0 else None

        # Extract body positions
        body_positions = np.argwhere(self.grid[:, :, 1] == 1.0)
        body = tuple(map(tuple, body_positions))

        # Extract food position
        food_positions = np.argwhere(self.grid[:, :, 2] == 1.0)
        food = tuple(food_positions[0]) if len(food_positions) > 0 else None

        # Create snake tuple (head + body)
        snake_tuple = (head,) + body if head else body

        return (snake_tuple, food, self.direction.value)

    def get_snake_positions(self) -> List[Point]:
        """
        Extract snake positions from grid.

        Returns:
            List of Points [head, body segments...] in order
        """
        positions = []

        # Get head
        head_positions = np.argwhere(self.grid[:, :, 0] == 1.0)
        if len(head_positions) > 0:
            y, x = head_positions[0]
            positions.append(Point(x, y))

        # Get body
        body_positions = np.argwhere(self.grid[:, :, 1] == 1.0)
        for y, x in body_positions:
            positions.append(Point(x, y))

        return positions

    def get_food_position(self) -> Point:
        """
        Extract food position from grid.

        Returns:
            Point where food is located
        """
        food_positions = np.argwhere(self.grid[:, :, 2] == 1.0)
        if len(food_positions) > 0:
            y, x = food_positions[0]
            return Point(x, y)
        return Point(0, 0)  # Fallback (shouldn't happen)

    def __repr__(self) -> str:
        """String representation for debugging."""
        snake_pos = self.get_snake_positions()
        food_pos = self.get_food_position()
        return (
            f"State(size={self.grid_size}, "
            f"snake_len={len(snake_pos)}, "
            f"head={snake_pos[0] if snake_pos else None}, "
            f"food={food_pos}, "
            f"dir={self.direction.name})"
        )
