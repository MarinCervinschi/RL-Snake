from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TrainingMetrics:
    """
    Tracks and visualizes training metrics.
    """

    def __init__(self, save_dir: str = "plots"):
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

        self.epsilons: List[float] = []

        # Derived metrics
        self.moving_avg_scores: List[float] = []

    @property
    def record(self) -> int:
        """Highest score achieved."""
        return max(self.scores) if self.scores else 0

    def reset(self) -> None:
        """Reset all recorded metrics."""
        self.episodes.clear()
        self.scores.clear()
        self.rewards.clear()
        self.moving_avg_scores.clear()
        self.epsilons.clear()

    def record_episode(
        self,
        episode: int,
        score: int,
        total_reward: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """
        Record metrics for a completed episode.

        Args:
            episode: Episode number
            score: Number of apples eaten
            steps: Number of steps taken
            total_reward: Total reward accumulated (optional)
            epsilon: Epsilon value (optional)
        """
        self.episodes.append(episode)
        self.scores.append(score)

        if total_reward is not None:
            self.rewards.append(total_reward)

        if epsilon is not None:
            self.epsilons.append(epsilon)

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

    def plot(self, show: bool = True, save: bool = True) -> None:
        """
        Generate and optionally display/save training plots.

        Args:
            show: Whether to display plots
            save: Whether to save plots to disk
            plot_epsilon: Whether to plot epsilon decay (if tracked)
        """
        if not self.episodes:
            print("⚠️  No data to plot")
            return

        print("📈 Generating training plots...")

        if self.epsilons:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
            fig.suptitle(
                "Training Metrics & Epsilon Decay", fontsize=16, fontweight="bold"
            )

            # Score plot
            ax1.plot(
                self.episodes, self.scores, alpha=0.3, color="blue", label="Raw Score"
            )
            if len(self.scores) > 50:
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

            # Epsilon plot
            ax2.plot(
                self.episodes[: len(self.epsilons)],
                self.epsilons,
                color="orange",
                label="Epsilon",
            )
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Epsilon")
            ax2.set_title("Epsilon Decay")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout(rect=(0, 0, 1, 0.97))

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
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
            fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")
            ax1.plot(
                self.episodes, self.scores, alpha=0.3, color="blue", label="Raw Score"
            )
            if len(self.scores) > 50:
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
            plt.tight_layout()
            if save:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                plot_path = self.save_dir / "training_metrics.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                print(f"📊 Plot saved to: {plot_path}")
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

    def print_summary(self, play: bool = False) -> None:
        """Print summary statistics."""
        if not self.scores:
            print("No metrics recorded yet")
            return

        print("\n" + "=" * 50)
        print("📊 Training Summary" if not play else "📊 Playback Summary")
        print("=" * 50)
        print(f"Total Episodes:     {len(self.episodes)}")
        print(f"Best Score:         {max(self.scores)}")
        if not play:
            print(f"Recent Avg (100):   {self.get_recent_average_score(100):.2f}")
        else:
            print(f"Worst Score:        {min(self.scores)}")
            print(f"Avg Score:          {np.mean(self.scores):.2f}")

        if self.rewards and not play:
            print(f"Avg Total Reward:   {np.mean(self.rewards):.2f}")

        print("=" * 50 + "\n")
