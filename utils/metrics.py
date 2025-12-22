from pathlib import Path
from typing import List, Optional, Union

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
        """
        self.save_dir = Path(save_dir)

        # Episode data
        self.episodes: List[int] = []
        self.scores: List[int] = []
        self.steps: List[int] = []
        self.rewards: List[float] = []
        self.epsilons: List[float] = []

    @property
    def record(self) -> int:
        """Highest score achieved."""
        return max(self.scores) if self.scores else 0

    def reset(self) -> None:
        """Reset all recorded metrics."""
        self.episodes.clear()
        self.scores.clear()
        self.steps.clear()
        self.rewards.clear()
        self.epsilons.clear()

    def record_episode(
        self,
        episode: int,
        score: int,
        steps: Optional[int] = None,
        reward: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """
        Record metrics for a completed episode.

        Args:
            episode: Episode number
            score: Number of apples eaten
            steps: Number of steps taken (Survival Time)
            reward: Reward obtained
            epsilon: Epsilon value
        """
        self.episodes.append(episode)
        self.scores.append(score)

        if steps is not None:
            self.steps.append(steps)

        if reward is not None:
            self.rewards.append(reward)

        if epsilon is not None:
            self.epsilons.append(epsilon)

    def get_recent_average_score(self, window: int = 100) -> float:
        """Get average score over recent episodes."""
        if not self.scores:
            return 0.0
        recent = self.scores[-window:]
        return float(np.mean(recent))

    # -------------------------------------------------------------------------
    # Plotting Helpers
    # -------------------------------------------------------------------------

    def _moving_average(
        self, data: Union[List[int], List[float]], window: int
    ) -> np.ndarray:
        """Calculate moving average for smoothing plots."""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window) / window, mode="same")

    def _setup_plot(self, title: str):
        """Helper to create a standard figure."""
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig, ax

    def _finalize_plot(self, path_name: str, show: bool, save: bool):
        """Helper to save and show plots."""
        plt.tight_layout()

        if save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            plot_path = self.save_dir / path_name
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"📊 Plot saved to: {plot_path}")

        if show:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()

    # -------------------------------------------------------------------------
    # Individual Plots
    # -------------------------------------------------------------------------

    def plot_learning_curve(
        self, show: bool = True, save: bool = True, ax=None
    ) -> None:
        """Plot scores over episodes."""
        if not self.episodes:
            return

        # Handle internal call (ax provided) vs external call (new figure)
        if ax is None:
            fig, ax = self._setup_plot("Learning Curve")
            is_standalone = True
        else:
            is_standalone = False

        # Raw data
        ax.plot(self.episodes, self.scores, alpha=0.3, color="blue", label="Raw Score")

        # Trend line
        if len(self.scores) > 50:
            window = min(50, len(self.scores) // 10)
            ma = self._moving_average(self.scores, window)
            ax.plot(
                self.episodes,
                ma,
                color="darkblue",
                linewidth=2,
                label=f"Avg ({window} ep)",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if is_standalone:
            self._finalize_plot("learning_curve.png", show, save)

    def plot_epsilon_decay(self, show: bool = True, save: bool = True, ax=None) -> None:
        """Plot epsilon values over episodes."""
        if not self.epsilons:
            return

        if ax is None:
            fig, ax = self._setup_plot("Epsilon Decay")
            is_standalone = True
        else:
            is_standalone = False

        ax.plot(
            self.episodes[: len(self.epsilons)],
            self.epsilons,
            color="orange",
            linewidth=2,
            label="Epsilon",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if is_standalone:
            self._finalize_plot("epsilon_decay.png", show, save)

    def plot_survival_time(self, show: bool = True, save: bool = True, ax=None) -> None:
        """Plot number of steps per episode."""
        if not self.steps:
            return

        if ax is None:
            fig, ax = self._setup_plot("Survival Time (Steps)")
            is_standalone = True
        else:
            is_standalone = False

        ax.plot(self.episodes, self.steps, alpha=0.4, color="green", label="Steps")

        # Trend line for steps
        if len(self.steps) > 50:
            window = min(50, len(self.steps) // 10)
            ma = self._moving_average(self.steps, window)
            ax.plot(
                self.episodes,
                ma,
                color="darkgreen",
                linewidth=2,
                label=f"Avg ({window} ep)",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps Taken")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if is_standalone:
            self._finalize_plot("survival_time.png", show, save)

    def plot_rewards(self, show: bool = True, save: bool = True, ax=None) -> None:
        """Plot total reward per episode."""
        if not self.rewards:
            return

        if ax is None:
            fig, ax = self._setup_plot("Total Reward per Episode")
            is_standalone = True
        else:
            is_standalone = False

        ax.plot(
            self.episodes[: len(self.rewards)],
            self.rewards,
            alpha=0.4,
            color="purple",
            label="Reward",
        )

        if len(self.rewards) > 50:
            window = min(50, len(self.rewards) // 10)
            ma = self._moving_average(self.rewards, window)
            ax.plot(
                self.episodes[: len(self.rewards)],
                ma,
                color="indigo",
                linewidth=2,
                label=f"Avg ({window} ep)",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if is_standalone:
            self._finalize_plot("rewards.png", show, save)

    # -------------------------------------------------------------------------
    # Main Dashboard
    # -------------------------------------------------------------------------

    def plot(self, show: bool = True, save: bool = True) -> None:
        """
        Generate a 2x2 grid containing all 4 plots.
        """
        if not self.episodes:
            print("⚠️  No data to plot")
            return

        print("📈 Generating training dashboard...")

        # Create 2x2 Grid
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Metrics Dashboard", fontsize=16, fontweight="bold")

        # Call individual methods, passing the specific axis
        # Top Left: Learning Curve
        self.plot_learning_curve(show=False, save=False, ax=axs[0, 0])
        axs[0, 0].set_title("Learning Curve (Scores)")

        # Top Right: Epsilon
        if self.epsilons:
            self.plot_epsilon_decay(show=False, save=False, ax=axs[0, 1])
            axs[0, 1].set_title("Epsilon Decay")
        else:
            axs[0, 1].text(0.5, 0.5, "No Epsilon Data", ha="center")

        # Bottom Left: Survival Time
        self.plot_survival_time(show=False, save=False, ax=axs[1, 0])
        axs[1, 0].set_title("Survival Time (Steps)")

        # Bottom Right: Rewards
        if self.rewards:
            self.plot_rewards(show=False, save=False, ax=axs[1, 1])
            axs[1, 1].set_title("Accumulated Rewards")
        else:
            axs[1, 1].text(0.5, 0.5, "No Reward Data", ha="center")

        plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle

        if save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            plot_path = self.save_dir / "training_dashboard.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"📊 Dashboard saved to: {plot_path}")

        if show:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()

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

        if not play and self.scores:
            print(f"Recent Avg Score:   {self.get_recent_average_score(100):.2f}")

        if self.steps:
            print(f"Avg Steps/Episode:  {np.mean(self.steps):.1f}")

        if self.rewards and not play:
            print(f"Avg Total Reward:   {np.mean(self.rewards):.2f}")

        print("=" * 50 + "\n")
