from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TrainingMetrics:
    """
    Tracks and visualizes training metrics.
    """

    def __init__(self, save_dir: str = "results"):
        """
        Initialize metrics tracker.

        Args:
            save_dir: Directory to save metrics and plots
            save_interval: Save metrics to disk every N episodes
        """

        self.save_dir = Path(save_dir)

        # Episode data
        self.episodes: List[int] = []
        self.scores: List[int] = []
        self.rewards: List[float] = []

        # Derived metrics
        self.steps_per_apple: List[float] = []
        self.moving_avg_scores: List[float] = []

    def record_episode(
        self, episode: int, score: int, steps: int, total_reward: Optional[float] = None
    ) -> None:
        """
        Record metrics for a completed episode.

        Args:
            episode: Episode number
            score: Number of apples eaten
            steps: Number of steps taken
            total_reward: Total reward accumulated (optional)
        """
        self.episodes.append(episode)
        self.scores.append(score)

        if total_reward is not None:
            self.rewards.append(total_reward)

        # Calculate efficiency (steps per apple)
        if score > 0:
            efficiency = steps / score
        else:
            efficiency = 0.0  # Or could use steps as penalty
        self.steps_per_apple.append(efficiency)

        # Calculate moving average (window=100)
        if len(self.scores) >= 100:
            moving_avg = np.mean(self.scores[-100:])
        else:
            moving_avg = np.mean(self.scores)

        moving_avg = float(moving_avg)
        self.moving_avg_scores.append(moving_avg)

    def get_recent_average_score(self, window: int = 100) -> float:
        """
        Get average score over recent episodes.

        Args:
            window: Number of recent episodes to average

        Returns:
            Average score
        """
        if not self.scores:
            return 0.0

        recent = self.scores[-window:]
        return float(np.mean(recent))

    def get_recent_average_efficiency(self, window: int = 100) -> float:
        """
        Get average efficiency over recent episodes.

        Args:
            window: Number of recent episodes to average

        Returns:
            Average steps per apple
        """
        if not self.steps_per_apple:
            return 0.0

        # Filter out zero-score episodes
        non_zero = [eff for eff in self.steps_per_apple[-window:] if eff > 0]
        if not non_zero:
            return 0.0

        return float(np.mean(non_zero))

    def plot(self, show: bool = True, save: bool = True) -> None:
        """
        Generate and optionally display/save training plots.

        Args:
            show: Whether to display plots
            save: Whether to save plots to disk
        """
        if not self.episodes:
            print("⚠️  No data to plot")
            return
        
        print("📈 Generating training plots...")


        # Create figure with 2 subplots in a single row
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

        # Plot 1: Score over time
        ax1 = axes[0]
        ax1.plot(self.episodes, self.scores, alpha=0.3, color="blue", label="Raw Score")

        if len(self.scores) > 50:
            # Moving average
            window = min(50, len(self.scores) // 10)
            moving_avg = self._moving_average(self.scores, window)
            ax1.plot(
                self.episodes,
                moving_avg,
                color="darkblue",
                linewidth=2,
                label=f"Avg ({window} ep)",
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score (Apples)")
        ax1.set_title("Learning Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Efficiency (steps per apple)
        ax2 = axes[1]

        # Filter out zero-score episodes for efficiency plot
        valid_indices = [i for i, s in enumerate(self.scores) if s > 0]
        if valid_indices:
            valid_episodes = [self.episodes[i] for i in valid_indices]
            valid_efficiency = [self.steps_per_apple[i] for i in valid_indices]

            ax2.plot(
                valid_episodes,
                valid_efficiency,
                alpha=0.3,
                color="orange",
                label="Steps/Apple",
            )

            if len(valid_efficiency) > 50:
                window = min(50, len(valid_efficiency) // 10)
                moving_avg = self._moving_average(valid_efficiency, window)
                ax2.plot(
                    valid_episodes,
                    moving_avg,
                    color="darkorange",
                    linewidth=2,
                    label=f"Avg ({window} ep)",
                )

            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Steps per Apple")
            ax2.set_title("Pathfinding Efficiency")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No successful episodes yet",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Pathfinding Efficiency")

        plt.tight_layout()

        # Save to file
        if save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            plot_path = self.save_dir / "training_metrics.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"📊 Plot saved to: {plot_path}")

        # Display
        if show:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()

    def _moving_average(self, data: List[float] | List[int], window: int) -> np.ndarray:
        """
        Calculate moving average.

        Args:
            data: Data to smooth
            window: Window size

        Returns:
            Smoothed data
        """
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window) / window, mode="same")

    def print_summary(self) -> None:
        """Print summary statistics."""
        if not self.scores:
            print("No metrics recorded yet")
            return

        print("\n" + "=" * 50)
        print("📊 Training Summary")
        print("=" * 50)
        print(f"Total Episodes:     {len(self.episodes)}")
        print(
            f"Average Score:      {np.mean(self.scores):.2f} ± {np.std(self.scores):.2f}"
        )
        print(f"Best Score:         {max(self.scores)}")
        print(f"Worst Score:        {min(self.scores)}")
        print(f"Recent Avg (100):   {self.get_recent_average_score(100):.2f}")

        non_zero_eff = [e for e in self.steps_per_apple if e > 0]
        if non_zero_eff:
            print(f"Avg Efficiency:     {np.mean(non_zero_eff):.1f} steps/apple")

        if self.rewards:
            print(f"Avg Total Reward:   {np.mean(self.rewards):.2f}")

        print("=" * 50 + "\n")
