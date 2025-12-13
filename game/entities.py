from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Simple immutable data structure for coordinates
Point = namedtuple('Point', 'x, y')