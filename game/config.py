from dataclasses import dataclass


@dataclass
class GameConfig:
    """Configuration for Snake game."""

    grid_size: int = 20
    initial_snake_length: int = 3

    # Reward structure
    reward_food: float = 10.0
    reward_collision: float = -10.0
    reward_step: float = 0.0

    # Timeout (prevent infinite loops)
    max_steps_multiplier: int = 100  # max_steps = multiplier * snake_length

    def __post_init__(self):
        """Validate configuration."""
        if self.grid_size < 5:
            raise ValueError("Grid size must be at least 5")
        if self.initial_snake_length < 3:
            raise ValueError("Initial snake length must be at least 3")
        if self.initial_snake_length >= self.grid_size:
            raise ValueError("Initial snake length must be less than grid size")
