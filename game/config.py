from dataclasses import dataclass


@dataclass
class GameConfig:
    """Configuration for Snake game with Reward Shaping and Milestones."""

    grid_size: int = 10
    initial_snake_length: int = 3

    reward_food: float = 10.0
    reward_collision: float = -10.0
    reward_step: float = 0.0

    reward_win: float = 1000.0
    reward_timeout: float = -5.0

    # Milestone lengths and rewards
    milestones = {snake: snake // 2 for snake in range(10, 101, 10)}

    # Timeout = multiplier * current_snake_length
    max_steps_multiplier: int = 100

    def __post_init__(self):
        """Validate configuration."""
        if self.grid_size < 5:
            raise ValueError("Grid size must be at least 5")
        if self.initial_snake_length < 3:
            raise ValueError("Initial snake length must be at least 3")
        if self.initial_snake_length >= self.grid_size:
            raise ValueError("Initial snake length must be less than grid size")

    @property
    def max_capacity(self) -> int:
        """Total number of cells in the grid."""
        return self.grid_size * self.grid_size

    def get_milestone_reward(self, snake_length: int) -> int:
        """Get reward for reaching a milestone based on snake length."""
        return self.milestones.get(snake_length, 0)
