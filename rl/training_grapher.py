from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rl.algorithms.base_algorithm import EpisodeResult


@dataclass(frozen=True)
class GraphingConfig:
    moving_average_window: int = 10


# graphs will be a 2x2 layout:
# First panel - longest balance per episode
# Second panel - total reward per episode
# Third panel - per step reward with all weights
# Fourth panel - command and command change per episode


class TrainingGrapher:
    def __init__(self, config: GraphingConfig) -> None:
        self._config = config

    def _slugify(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def _balance_graph(self, results: list[EpisodeResult], axes):
        """plot of longest balance per episode"""
        longest_balances = [r.longest_balance for r in results]

        moving_average = (
            pd.Series(longest_balances)
            .rolling(window=self._config.moving_average_window)
            .mean()
        )

        episodes = [r.number for r in results]
        sns.lineplot(x=episodes, y=longest_balances, ax=axes, label="Longest Balance")
        sns.lineplot(
            x=episodes,
            y=moving_average,
            ax=axes,
            label=f"Moving Average ({self._config.moving_average_window})",
        )
        axes.set_title("Longest Balance")
        axes.set_xlabel("Episode")
        axes.set_ylabel("Longest Balance (steps)")

    def _reward_graph(self, results: list[EpisodeResult], axes):
        """plot of total reward per episode"""
        total_rewards = [r.total_reward for r in results]

        moving_average = (
            pd.Series(total_rewards)
            .rolling(window=self._config.moving_average_window)
            .mean()
        )

        max_reward_reference = [r.max_reward for r in results]

        episodes = [r.number for r in results]
        sns.lineplot(x=episodes, y=total_rewards, ax=axes, label="Total Reward")
        sns.lineplot(
            x=episodes,
            y=moving_average,
            ax=axes,
            label=f"Moving Average ({self._config.moving_average_window})",
        )
        sns.lineplot(
            x=episodes,
            y=max_reward_reference,
            ax=axes,
            linestyle="--",
            color="gray",
            label="Max Reward",
        )
        axes.set_title("Total Reward")
        axes.set_xlabel("Episode")
        axes.set_ylabel("Total Reward")

    def _per_step_reward_graph(self, results: list[EpisodeResult], axes):
        zero_line = [0] * len(results)
        episodes = [r.number for r in results]

        avg_total_rewards = [r.avg_total_reward for r in results]
        avg_upright_rewards = [r.avg_upright_reward for r in results]
        avg_rail_centre_rewards = [r.avg_rail_centre_reward for r in results]
        avg_spin_rewards = [r.avg_spin_reward for r in results]

        sns.lineplot(
            x=episodes, y=zero_line, ax=axes, color="gray", linestyle="--", label="Zero"
        )
        sns.lineplot(x=episodes, y=avg_total_rewards, ax=axes, label="Total")
        sns.lineplot(x=episodes, y=avg_upright_rewards, ax=axes, label="Upright")
        sns.lineplot(
            x=episodes, y=avg_rail_centre_rewards, ax=axes, label="Rail Centre"
        )
        sns.lineplot(x=episodes, y=avg_spin_rewards, ax=axes, label="Spin")

        axes.set_title("Per Step Rewards")
        axes.set_xlabel("Episode")
        axes.set_ylabel("Average Reward")

    def _command_graph(self, results: list[EpisodeResult], axes):
        episodes = [r.number for r in results]

        avg_commands = [r.avg_command for r in results]
        avg_command_changes = [r.avg_command_change for r in results]

        sns.lineplot(x=episodes, y=avg_commands, ax=axes, label="Command")
        sns.lineplot(x=episodes, y=avg_command_changes, ax=axes, label="Command Change")

        axes.set_title("Command and Command Change")
        axes.set_xlabel("Episode")
        axes.set_ylabel("Average Command")

    def plot_results(self, results: list[EpisodeResult], filename: Path) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        self._balance_graph(results, axes[0, 0])
        self._reward_graph(results, axes[0, 1])
        self._per_step_reward_graph(results, axes[1, 0])
        self._command_graph(results, axes[1, 1])

        fig.savefig(filename)
        plt.close(fig)
