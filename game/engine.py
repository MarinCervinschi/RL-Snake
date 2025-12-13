import random
from .entities import Direction, Point, Action, State
from config import GRID_SIZE
from typing import Optional


class SnakeGameEngine:
    def __init__(self):
        self.w = GRID_SIZE
        self.h = GRID_SIZE
        self.reset()

    def reset(self) -> State:
        """Restarts the game to initial state."""
        self.direction = Direction.RIGHT

        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        return self._get_state_object()

    def _place_food(self) -> None:
        """Places food randomly, ensuring it's not inside the snake body."""
        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def step(self, action: Action) -> tuple[int, bool, int]:
        """Executes one game step based on the action taken by the agent.
        Returns:
            reward (int): Reward obtained from this action.
            game_over (bool): Whether the game has ended.
            score (int): Current score.
        """
        self.frame_iteration += 1

        # 1. Move based on Action Enum
        self._move(action)
        self.snake.insert(0, self.head)

        # 2. Check collisions
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 3. Check Food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def _is_collision(self, pt: Optional[Point] = None) -> bool:
        """Checks if a point (default: head) hits wall or self."""
        if pt is None:
            pt = self.head

        # Hits Wall
        if pt.x > self.w - 1 or pt.x < 0 or pt.y > self.h - 1 or pt.y < 0:
            return True

        # Hits Self (ignore head itself)
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action: Action) -> None:
        """
        Updates the Head position and Direction based on relative action.
        [Straight, Right, Left] -> e.g., [1, 0, 0]
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == Action.STRAIGHT:
            new_dir = clock_wise[idx]
        elif action == Action.RIGHT:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        # Update coordinates
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)

    def get_state(self) -> State:
        """Public alias for _get_state_object"""
        return self._get_state_object()

    def _get_state_object(self) -> State:
        """Constructs and returns the State object."""
        head = self.snake[0]

        # Check immediate surroundings
        pt_l = Point(head.x - 1, head.y)
        pt_r = Point(head.x + 1, head.y)
        pt_u = Point(head.x, head.y - 1)
        pt_d = Point(head.x, head.y + 1)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        assert (
            self.food is not None
        ), "Food should always be placed before getting state."

        return State(
            # Danger Sensors
            danger_straight=(dir_r and self._is_collision(pt_r))
            or (dir_l and self._is_collision(pt_l))
            or (dir_u and self._is_collision(pt_u))
            or (dir_d and self._is_collision(pt_d)),
            danger_right=(dir_u and self._is_collision(pt_r))
            or (dir_d and self._is_collision(pt_l))
            or (dir_l and self._is_collision(pt_u))
            or (dir_r and self._is_collision(pt_d)),
            danger_left=(dir_d and self._is_collision(pt_r))
            or (dir_u and self._is_collision(pt_l))
            or (dir_r and self._is_collision(pt_u))
            or (dir_l and self._is_collision(pt_d)),
            # Direction Sensors
            moving_left=dir_l,
            moving_right=dir_r,
            moving_up=dir_u,
            moving_down=dir_d,
            # Food Sensors
            food_left=self.food.x < self.head.x,
            food_right=self.food.x > self.head.x,
            food_up=self.food.y < self.head.y,
            food_down=self.food.y > self.head.y,
        )
