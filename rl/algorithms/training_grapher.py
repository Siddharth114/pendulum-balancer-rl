from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rl.algorithms.base_algorithm import TrainingResults


@dataclass(frozen=True)
class GraphingConfig:
    reward_color: str = "#4b72b0"
    moving_average_window: int = 10
    moving_average_color: str = "#c44e52"

    episode_length_color: str = "#dd8452"

    results_dir: str = "results"  # relative to where you run from


class TrainingGrapher:
    def __init__(self, config: GraphingConfig) -> None:
        self._config = config

    def _slugify(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def plot_results(self, results: TrainingResults, name: str) -> Path:
        # fresh figure per call so repeated plots don't draw on top of each other
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        episodes = range(len(results.episode_rewards))

        moving_avg = (
            pd.Series(results.episode_rewards)
            .rolling(window=self._config.moving_average_window)
            .mean()
        )

        sns.lineplot(
            x=episodes,
            y=results.episode_rewards,
            ax=axes[0],
            color=self._config.reward_color,
            alpha=0.7,
            label="Episode Reward",
        )
        sns.lineplot(
            x=episodes,
            y=moving_avg,
            ax=axes[0],
            color=self._config.moving_average_color,
            linewidth=2,
            label=f"{self._config.moving_average_window}-Episode Moving Avg",
        )

        axes[0].set_title(f"{name} - Episode Rewards")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].legend(loc="center right")

        sns.lineplot(
            x=episodes,
            y=results.episode_lengths,
            ax=axes[1],
            color=self._config.episode_length_color,
            label="Episode Lengths",
        )

        axes[1].set_title(f"{name} - Episode Lengths")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Episode Length (steps)")

        fig.tight_layout()

        results_dir = Path(self._config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        path = results_dir / f"{self._slugify(name)}_results_{timestamp}.png"
        fig.savefig(path)
        plt.close(fig)
        return path
