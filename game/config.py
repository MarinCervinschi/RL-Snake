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

    # Level 1: "Human Competence" (~30% of map)
    milestone_expert_ratio: float = 0.3
    reward_milestone_expert: float = 50.0

    # Level 2: "Mastery" (~50% of map)
    milestone_master_ratio: float = 0.5
    reward_milestone_master: float = 100.0

    # Level 3: "Near Perfection" (~90% of map)
    milestone_grandmaster_ratio: float = 0.9
    reward_milestone_grandmaster: float = 500.0

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

        # Validation for ratios
        if not (
            0
            < self.milestone_expert_ratio
            < self.milestone_master_ratio
            < self.milestone_grandmaster_ratio
            < 1.0
        ):
            raise ValueError("Milestone ratios must be ascending and between 0 and 1")

    @property
    def max_capacity(self) -> int:
        """Total number of cells in the grid."""
        return self.grid_size * self.grid_size

    def get_milestone_length(self, ratio: float) -> int:
        """Calculate exact snake length required for a milestone."""
        return int(self.max_capacity * ratio)
