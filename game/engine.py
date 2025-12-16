import random
from typing import Optional, Tuple


from .entities import Action, Direction, State, Point
from .config import GameConfig


class SnakeGameEngine:
    """
    Snake game engine.
    """

    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize game engine.

        Args:
            config: Game configuration (uses defaults if None)
        """
        self.config = config or GameConfig()
        self.grid_size = self.config.grid_size

        # Game state variables
        self.snake: list[Point] = []
        self.food: Optional[Point] = None
        self.direction: Direction = Direction.RIGHT
        self.score: int = 0
        self.frame_iteration: int = 0
        self.game_over: bool = False

        # Initialize game
        self.reset()

    def reset(self) -> State:
        """
        Reset game to initial state.

        Returns:
            Initial State
        """
        # Reset game variables
        self.score = 0
        self.frame_iteration = 0
        self.game_over = False

        # Initialize snake in center, facing right
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2

        # Create initial snake (length 3, horizontal)
        self.snake = [
            Point(center_x, center_y),  # Head
            Point(center_x - 1, center_y),  # Body segment 1
            Point(center_x - 2, center_y),  # Body segment 2
        ]

        # Initial direction
        self.direction = Direction.RIGHT

        # Place food
        self._place_food()

        # Return initial state
        return self.get_state()

    def _place_food(self) -> None:
        """
        Place food randomly on grid, avoiding snake positions.
        """
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            food = Point(x, y)

            # Ensure food is not on snake
            if food not in self.snake:
                self.food = food
                break

    def step(self, action: Action) -> Tuple[float, bool, int]:
        """
        Execute one game step based on action.

        Args:
            action: Absolute action (UP, RIGHT, DOWN, LEFT)

        Returns:
            Tuple of (reward, game_over, score):
                - reward: Immediate reward for this transition
                - game_over: Whether episode has terminated
                - score: Current score (number of apples eaten)
        """
        self.frame_iteration += 1

        # 1. Determine effective action (handle invalid 180° turns)
        effective_action = self._get_effective_action(action)

        # 2. Update direction and move snake
        self.direction = Direction.from_action(effective_action)
        self._move_snake()

        # 3. Check for collisions
        reward = 0.0

        # Check timeout (prevent infinite loops)
        if self.frame_iteration > self.config.max_steps_multiplier * len(self.snake):
            self.game_over = True
            reward = self.config.reward_collision
            return reward, self.game_over, self.score

        # Check wall collision or self-collision
        if self._is_collision():
            self.game_over = True
            reward = self.config.reward_collision
            return reward, self.game_over, self.score

        # 4. Check if food was eaten
        if self.snake[0] == self.food:
            self.score += 1
            reward = self.config.reward_food
            self._place_food()
            # Snake grows (don't remove tail)
        else:
            # Normal move (remove tail, snake stays same length)
            self.snake.pop()
            reward = self.config.reward_step

        return reward, self.game_over, self.score

    def _get_effective_action(self, action: Action) -> Action:
        """
        Get effective action, preventing 180° turns.

        If the action would cause the snake to reverse direction
        (e.g., currently moving RIGHT, action is LEFT), the snake
        continues in its current direction instead.

        Args:
            action: Requested action

        Returns:
            Effective action (either requested action or current direction)
        """
        requested_direction = Direction.from_action(action)

        # Check if this would be a 180° turn
        if requested_direction == self.direction.opposite():
            # Invalid: Continue in current direction
            return Action(self.direction.value)

        # Valid action
        return action

    def _move_snake(self) -> None:
        """
        Move snake one step in current direction.

        Adds new head position, tail is removed later if no food eaten.
        """
        head = self.snake[0]
        dx, dy = self.direction.to_vector()

        new_head = Point(head.x + dx, head.y + dy)
        self.snake.insert(0, new_head)

    def _is_collision(self, point: Optional[Point] = None) -> bool:
        """
        Check if point collides with wall or snake body.

        Args:
            point: Point to check (defaults to current head)

        Returns:
            True if collision, False otherwise
        """
        if point is None:
            point = self.snake[0]

        # Wall collision
        if (
            point.x < 0
            or point.x >= self.grid_size
            or point.y < 0
            or point.y >= self.grid_size
        ):
            return True

        # Self-collision (check if head hit body)
        # Note: snake[0] is head, snake[1:] is body
        if point in self.snake[1:]:
            return True

        return False

    def get_state(self) -> State:
        """
        Get current game state as State (full MDP representation).

        Returns:
            State with 3-channel grid and direction
        """
        assert self.food is not None
        return State.from_game_state(
            grid_size=self.grid_size,
            snake_positions=self.snake,
            food_position=self.food,
            direction=self.direction,
        )

    def is_valid_position(self, point: Point) -> bool:
        """
        Check if a point is within grid bounds.

        Args:
            point: Point to check

        Returns:
            True if within bounds, False otherwise
        """
        return 0 <= point.x < self.grid_size and 0 <= point.y < self.grid_size

    def get_available_food_positions(self) -> list[Point]:
        """
        Get all valid positions for food (not occupied by snake).

        Returns:
            List of Points where food can be placed
        """
        available = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                point = Point(x, y)
                if point not in self.snake:
                    available.append(point)
        return available

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SnakeGame(size={self.grid_size}, "
            f"score={self.score}, "
            f"snake_len={len(self.snake)}, "
            f"steps={self.frame_iteration})"
        )
