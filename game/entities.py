from collections import namedtuple
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Action(Enum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2


@dataclass
class State:
    # Sensor 1: Danger detection
    danger_straight: bool
    danger_right: bool
    danger_left: bool

    # Sensor 2: Current Movement Direction
    moving_left: bool
    moving_right: bool
    moving_up: bool
    moving_down: bool

    # Sensor 3: Food Location
    food_left: bool
    food_right: bool
    food_up: bool
    food_down: bool

    def to_vector(self) -> np.ndarray:
        """Converts the state object into the 11-value vector the AI needs."""
        return np.array(
            [
                self.danger_straight,
                self.danger_right,
                self.danger_left,
                self.moving_left,
                self.moving_right,
                self.moving_up,
                self.moving_down,
                self.food_left,
                self.food_right,
                self.food_up,
                self.food_down,
            ],
            dtype=int,
        )

    def to_index(self) -> int:
        """Converts the boolean state directly to a unique integer (0-2047) for the Q-Table."""
        binary_string = "".join(str(int(x)) for x in self.to_vector())
        return int(binary_string, 2)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")
