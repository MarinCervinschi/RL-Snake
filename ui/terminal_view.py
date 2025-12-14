import os
import platform
import time
from typing import Optional

from core.interfaces import IRenderer
from game.entities import Point


class TerminalRenderer(IRenderer):
    def __init__(self, width: int, height: int, speed: float = 0.1):
        self.width = width
        self.height = height
        self.speed = speed  # Delay in seconds (to make it watchable)
        self.chars = {
            "EMPTY": " . ",
            "SNAKE_HEAD": " @ ",
            "SNAKE_BODY": " o ",
            "FOOD": " X ",
        }

    def render(
        self, snake: list, food: Optional[Point], score: int, record: dict
    ) -> None:
        """
        Draws the game state to the terminal.
        Args:
            snake (list): List of Point objects representing the snake.
            food (Point): The location of the food.
            score (int): Current game score.
            record (int): All-time high score.
        """
        grid = [
            [self.chars["EMPTY"] for _ in range(self.width)] for _ in range(self.height)
        ]

        if food:
            grid[food.y][food.x] = self.chars["FOOD"]

        for i, point in enumerate(snake):
            if i == 0:
                char = self.chars["SNAKE_HEAD"]
            else:
                char = self.chars["SNAKE_BODY"]

            # Safety check to ensure we don't crash drawing if snake is out of bounds (during debug)
            if 0 <= point.y < self.height and 0 <= point.x < self.width:
                grid[point.y][point.x] = char

        command = "cls" if platform.system() == "Windows" else "clear"
        os.system(command)

        print(f"--- GENERATION: {record['episode']} ---")
        print(f"SCORE: {score} | HIGH SCORE: {record['record']}")
        print("-" * (self.width * 3))

        for row in grid:
            print("".join(row))

        print("-" * (self.width * 3))

        time.sleep(self.speed)
