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
            Direction.UP: (0, -1),
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
        }
        return vectors[self]


Point = namedtuple("Point", "x, y")


@dataclass
class State:
    """
    Full MDP State: Grid (Spatial) + Direction (Heading) + Time (Temporal).

    Channels:
        0: Head (1.0)
        1: Body (Gradient 1.0 -> 0.0)
        2: Food (1.0)
        3: Time (Filled with normalized time-to-starvation)

    This satisfies the Markov Property because:
    1. Spatial: We know where everything is.
    2. Sequential: Body gradient reveals history/tail.
    3. Temporal: Time channel reveals urgency/starvation risk.
    """

    grid_size: int
    grid: np.ndarray  # Shape: (H, W, 4) <--- NOW 4 CHANNELS
    direction: Direction
    time_norm: float  # Normalized time (0.0 = fresh, 1.0 = starving)

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        # 4 Channels: Head, Body, Food, Time
        self.grid = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        self.direction = Direction.RIGHT
        self.time_norm = 0.0

    @classmethod
    def from_game_state(
        cls,
        grid_size: int,
        snake_positions: List[Point],
        food_position: Point,
        direction: Direction,
        frame_iteration: int,
        max_steps: int,
    ) -> State:
        """
        Create State including temporal info.
        """
        state = cls(grid_size)
        state.direction = direction

        # Calculate normalized time (0.0 to 1.0)
        # We clip it at 1.0 to prevent standardizers from breaking if we go slightly over
        if max_steps > 0:
            state.time_norm = min(float(frame_iteration) / float(max_steps), 1.0)
        else:
            state.time_norm = 0.0

        # Clear grid
        state.grid.fill(0.0)

        # --- Channel 0: Head ---
        if snake_positions:
            head = snake_positions[0]
            if 0 <= head.x < grid_size and 0 <= head.y < grid_size:
                state.grid[head.y, head.x, 0] = 1.0

        # --- Channel 1: Body (Gradient) ---
        body_segments = snake_positions[1:]
        num_body = len(body_segments)
        if num_body > 0:
            for i, segment in enumerate(body_segments):
                if 0 <= segment.x < grid_size and 0 <= segment.y < grid_size:
                    # Neck=1.0, Tail -> >0.0
                    value = (num_body - i) / num_body
                    state.grid[segment.y, segment.x, 1] = value

        # --- Channel 2: Food ---
        if 0 <= food_position.x < grid_size and 0 <= food_position.y < grid_size:
            state.grid[food_position.y, food_position.x, 2] = 1.0

        # --- Channel 3: Time (Global Feature) ---
        # We fill the entire channel with the time value.
        # This allows a CNN to learn "Global Urgency" from the pixel intensity.
        state.grid[:, :, 3] = state.time_norm

        return state

    def to_tensor(self) -> np.ndarray:
        """
        Convert to tensor format for neural networks.
        Returns: np.ndarray with shape (4, H, W)
        """
        # Transpose from (H, W, 4) to (4, H, W)
        return np.transpose(self.grid, (2, 0, 1))

    def to_position_tuple(self) -> Tuple:
        """
        Hashable representation. Includes discretized time bucket.
        """
        head_indices = np.argwhere(self.grid[:, :, 0] == 1.0)
        head = tuple(head_indices[0]) if len(head_indices) > 0 else None

        # Body handling (same as before, sorted by gradient)
        body_indices = np.argwhere(self.grid[:, :, 1] > 0.0)
        if len(body_indices) > 0:
            segments_with_val = []
            for y, x in body_indices:
                val = self.grid[y, x, 1]
                segments_with_val.append((val, y, x))
            segments_with_val.sort(key=lambda s: s[0], reverse=True)
            body = tuple((y, x) for _, y, x in segments_with_val)
        else:
            body = ()

        food_indices = np.argwhere(self.grid[:, :, 2] == 1.0)
        food = tuple(food_indices[0]) if len(food_indices) > 0 else None

        # Discretize time for hashing (e.g., 10% buckets)
        # This prevents the hash from being unique every single millisecond
        time_bucket = int(self.time_norm * 10)

        snake_tuple = (head,) + body if head else body

        return (snake_tuple, food, self.direction.value, time_bucket)
