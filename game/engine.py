import random
import numpy as np
from .entities import Direction, Point
from config import GRID_SIZE


class SnakeGameEngine:
    def __init__(self):
        self.w = GRID_SIZE
        self.h = GRID_SIZE
        self.reset()

    def reset(self):
        # Init snake, food, score
        self.snake = [Point(self.w // 2, self.h // 2)]
        self.score = 0
        self._place_food()
        return self._get_state_vector()

    def step(self, action):
        # 1. Move snake head
        # 2. Check collision (Game Over logic)
        # 3. Check food (Reward logic)
        # Returns: (reward, game_over, score)
        pass

    def _place_food(self):
        # Place food at random position not occupied by the snake
        pass

    def _get_state_vector(self):
        # Returns a vector representation of the current state
        pass
