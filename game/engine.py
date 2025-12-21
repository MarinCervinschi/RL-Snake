import random
from typing import Optional, Tuple

from .config import GameConfig
from .entities import Action, Direction, Point, State


class SnakeGameEngine:
    """
    Snake game engine with valid MDP dynamics and Reward Shaping.
    """

    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.grid_size = self.config.grid_size

        # Game state variables
        self.snake: list[Point] = []
        self.food: Optional[Point] = None
        self.direction: Direction = Direction.RIGHT
        self.score: int = 0
        self.frame_iteration: int = 0
        self.game_over: bool = False
        self.won: bool = False

        self.reset()

    def reset(self) -> State:
        self.score = 0
        self.frame_iteration = 0
        self.game_over = False
        self.won = False

        center_x = self.grid_size // 2
        center_y = self.grid_size // 2

        self.snake = [
            Point(center_x, center_y),
            Point(center_x - 1, center_y),
            Point(center_x - 2, center_y),
        ]

        self.direction = Direction.RIGHT
        self._place_food()

        return self.get_state()

    def step(self, action: Action) -> Tuple[float, bool, int]:
        self.frame_iteration += 1

        self._apply_relative_action(action)
        self._move_snake()

        if self._is_collision():
            self.game_over = True
            return self.config.reward_collision, self.game_over, self.score

        if self.frame_iteration > self.config.max_steps_multiplier * len(self.snake):
            self.game_over = True
            return self.config.reward_timeout, self.game_over, self.score

        # Check Food / Win / Step
        reward = 0.0

        if self.snake[0] == self.food:
            self.score += 1
            reward = self.config.reward_food

            self.frame_iteration = 0

            if len(self.snake) == self.grid_size * self.grid_size:
                self.won = True
                self.game_over = True
                return self.config.reward_win, self.game_over, self.score

            # CHECK MILESTONE REWARDS (Reward Shaping)
            length = len(self.snake)
            if length == self.config.get_milestone_length(
                self.config.milestone_expert_ratio
            ):
                reward += self.config.reward_milestone_expert
            elif length == self.config.get_milestone_length(
                self.config.milestone_master_ratio
            ):
                reward += self.config.reward_milestone_master
            elif length == self.config.get_milestone_length(
                self.config.milestone_grandmaster_ratio
            ):
                reward += self.config.reward_milestone_grandmaster
            self._place_food()

        else:
            # Normal move, pop tail
            self.snake.pop()
            reward = self.config.reward_step

        return reward, self.game_over, self.score

    def _place_food(self) -> None:
        """
        Place food efficiently.
        Uses random sampling for early game, and list filtering for late game.
        """

        available_spots = (self.grid_size * self.grid_size) - len(self.snake)
        if available_spots <= 0:
            return  # Should be caught by Win Condition, but safety first.

        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            food = Point(x, y)

            if food not in self.snake:
                self.food = food
                break

    def _apply_relative_action(self, action: Action) -> None:
        """
        Update self.direction based on a relative action.
        """
        if action == Action.FORWARD:
            return

        elif action == Action.TURN_LEFT:
            # Rotate -90°
            self.direction = Direction((self.direction.value - 1) % 4)

        elif action == Action.TURN_RIGHT:
            # Rotate +90°
            self.direction = Direction((self.direction.value + 1) % 4)

    def _move_snake(self) -> None:
        head = self.snake[0]
        dx, dy = self.direction.to_vector()
        new_head = Point(head.x + dx, head.y + dy)
        self.snake.insert(0, new_head)

    def _is_collision(self, point: Optional[Point] = None) -> bool:
        if point is None:
            point = self.snake[0]

        # Wall
        if (
            point.x < 0
            or point.x >= self.grid_size
            or point.y < 0
            or point.y >= self.grid_size
        ):
            return True

        # Body (Head is at index 0, check against 1:)
        if point in self.snake[1:]:
            return True

        return False

    def get_state(self) -> State:
        assert self.food is not None
        current_max_steps = self.config.max_steps_multiplier * len(self.snake)

        return State.from_game_state(
            grid_size=self.grid_size,
            snake_positions=self.snake,
            food_position=self.food,
            direction=self.direction,
            frame_iteration=self.frame_iteration,
            max_steps=current_max_steps,
        )
