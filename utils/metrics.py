import matplotlib.pyplot as plt
import numpy as np

from core.interfaces import IMetricsLogger


class TrainingMetrics(IMetricsLogger):
    def __init__(self, save_interval=100, save_plot=False):
        self.episodes = []
        self.scores = []
        self.steps_per_apple = []
        self.save_interval = save_interval
        self.save_plot = save_plot

    def record_episode(self, episode: int, score: int, steps: int):
        self.episodes.append(episode)
        self.scores.append(score)

        # Calculate Efficiency: Steps / Apple
        # If score is 0, we treat efficiency as the total steps survived (or 0)
        efficiency = steps / score if score > 0 else 0
        self.steps_per_apple.append(efficiency)

    def plot(self):
        """
        Plots two graphs:
        1. Learning Curve (Score over time)
        2. Efficiency Curve (Steps taken to find an apple)
        """
        # We assume this is called at the end of training or periodically

        plt.figure(figsize=(12, 5))

        # --- Graph 1: Score (The "Result") ---
        plt.subplot(1, 2, 1)
        plt.plot(self.episodes, self.scores, label="Raw Score", color="cyan", alpha=0.3)
        # Add a moving average trendline to make it readable
        if len(self.scores) > 10:
            trend = self._moving_average(self.scores)
            plt.plot(self.episodes, trend, label="Avg Score (50 eps)", color="blue")

        plt.title("Agent Learning Curve")
        plt.xlabel("Episodes")
        plt.ylabel("Score (Apples)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # --- Graph 2: Efficiency (The "Intelligence") ---
        plt.subplot(1, 2, 2)
        # We filter out 0-score games to avoid noise in efficiency graph
        valid_indices = [i for i, s in enumerate(self.scores) if s > 0]
        if valid_indices:
            valid_eps = [self.episodes[i] for i in valid_indices]
            valid_eff = [self.steps_per_apple[i] for i in valid_indices]

            plt.plot(
                valid_eps, valid_eff, label="Steps per Apple", color="orange", alpha=0.3
            )

            if len(valid_eff) > 10:
                trend_eff = self._moving_average(valid_eff)
                plt.plot(valid_eps, trend_eff, label="Avg Efficiency", color="red")

        plt.title("Pathfinding Efficiency")
        plt.xlabel("Episodes")
        plt.ylabel("Steps needed per Apple")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Updates window without freezing code

        if self.save_plot:
            plt.savefig(f"docs/training_metrics_episode_{self.episodes[-1]}.png")

    def _moving_average(self, data, window_size=50):
        """Helper to smooth out the jittery graphs."""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode="same")
