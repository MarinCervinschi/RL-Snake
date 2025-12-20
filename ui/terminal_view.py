import time
from typing import Optional
from IPython.display import clear_output
from game.entities import Point


class TerminalRenderer:
    def __init__(self, grid_size: int, speed: float = 0.1):
        self.width = grid_size
        self.height = grid_size
        self.speed = speed  # Delay in seconds (to make it watchable)
        self.chars = {
            "EMPTY": " . ",
            "SNAKE_HEAD": " @ ",
            "SNAKE_BODY": " o ",
            "FOOD": " X ",
        }

    def render(
        self, snake: list, food: Optional[Point], score: int, record: int
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

            if 0 <= point.y < self.height and 0 <= point.x < self.width:
                grid[point.y][point.x] = char

        clear_output(wait=True)

        print(f"SCORE: {score} | HIGH SCORE: {record}")
        print("-" * (self.width * 3))

        for row in grid:
            print("".join(row))

        print("-" * (self.width * 3))

        time.sleep(self.speed)
