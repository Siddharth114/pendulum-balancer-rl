from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.typing import RcKeyType

from rl.algorithms.episode import Episode

STYLE_OVERRIDES: dict[RcKeyType, Any] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "legend.fontsize": 8,
}

COLORS = {
    "metric": "#2a78d6",  # the panel's main quantity
    "command_change": "#eb6834",
    "upright": "#2a78d6",
    "rail_centre": "#eb6834",
    "spin": "#1baf7a",
    "total": "#52514e",
    "reference": "#9a9a94",
}


@dataclass(frozen=True)
class GrapherConfig:
    trend_window_fraction: float = 0.05  # fraction of the run for moving average
    dpi: int = 150


# graphs are a 2x2 layout:
# top left - longest balance per episode
# top right - total reward per episode
# bottom left - per step reward, split by term
# bottom right - command and command change per episode


class Grapher:
    def __init__(self, config: GrapherConfig) -> None:
        self._config = config

    def _trend_window(self, num_episodes: int) -> int:
        return max(1, round(num_episodes * self._config.trend_window_fraction))

    def _smooth(self, values: list[float]) -> np.ndarray:
        """trailing moving average that starts at the first episode, averaging
        whatever is available until the window fills up"""
        window = self._trend_window(len(values))
        cumulative = np.concatenate(([0.0], np.cumsum(values)))

        end = np.arange(1, len(values) + 1)
        start = np.maximum(0, end - window)

        return (cumulative[end] - cumulative[start]) / (end - start)

    def _plot_raw_and_trend(
        self,
        axes: Axes,
        episodes: list[int],
        values: list[float],
        color: str,
        label: str | None = None,
    ) -> None:
        """faint dot per episode, bold moving average on top"""
        axes.scatter(episodes, values, s=8, alpha=0.35, color=color, linewidths=0)
        axes.plot(episodes, self._smooth(values), color=color, linewidth=2, label=label)

    def _reference_line(self, axes: Axes, y: float, episodes: list[int], text: str):
        axes.axhline(y, color=COLORS["reference"], linestyle="--", linewidth=1)
        axes.annotate(
            text,
            xy=(episodes[-1], y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color=COLORS["reference"],
        )

    def _balance_graph(self, results: list[Episode], axes: Axes, trend: str):
        episodes = [r.number for r in results]

        self._plot_raw_and_trend(
            axes,
            episodes,
            [r.metrics.longest_balance for r in results],
            COLORS["metric"],
        )

        axes.set_title(f"Longest balance{trend}")
        axes.set_ylabel("steps")
        axes.set_ylim(bottom=0)

    def _reward_graph(self, results: list[Episode], axes: Axes, trend: str):
        episodes = [r.number for r in results]
        max_reward = results[0].metrics.max_reward

        returns = [r.metrics.total_reward for r in results]

        self._plot_raw_and_trend(axes, episodes, returns, COLORS["metric"])
        self._reference_line(axes, max_reward, episodes, f"max {max_reward:g}")

        axes.set_title(f"Return per episode{trend}")
        axes.set_ylabel("reward")

        axes.set_ylim(min(0, min(returns) * 1.05), max_reward * 1.05)

    def _per_step_reward_graph(self, results: list[Episode], axes: Axes):
        episodes = [r.number for r in results]

        axes.axhline(0, color=COLORS["reference"], linewidth=1)

        for label, values, color in [
            (
                "Upright",
                [r.metrics.avg_upright_reward for r in results],
                COLORS["upright"],
            ),
            (
                "Rail centre",
                [r.metrics.avg_rail_centre_reward for r in results],
                COLORS["rail_centre"],
            ),
            ("Spin", [r.metrics.avg_spin_reward for r in results], COLORS["spin"]),
        ]:
            axes.plot(
                episodes, self._smooth(values), color=color, linewidth=1.25, label=label
            )

        axes.plot(
            episodes,
            self._smooth([r.metrics.avg_total_reward for r in results]),
            color=COLORS["total"],
            linewidth=2.25,
            label="Total",
        )

        axes.set_title("Reward per step, by term")
        axes.set_xlabel("Episode")
        axes.set_ylabel("reward / step")
        axes.legend(loc="best")

    def _command_graph(self, results: list[Episode], axes: Axes):
        episodes = [r.number for r in results]

        self._plot_raw_and_trend(
            axes,
            episodes,
            [r.metrics.avg_command for r in results],
            COLORS["metric"],
            label="|command|",
        )
        self._plot_raw_and_trend(
            axes,
            episodes,
            [r.metrics.avg_command_change for r in results],
            COLORS["command_change"],
            label="|command change|",
        )

        axes.set_title("Control effort")
        axes.set_xlabel("Episode")
        axes.set_ylabel("mean per step")
        axes.set_ylim(bottom=0)
        axes.legend(loc="best")

    def plot_results(self, results: list[Episode], filename: Path, title: str) -> None:
        window = self._trend_window(len(results))
        # title suffix, empty when the run is too short to smooth
        trend = f" · {window}-ep avg" if window > 1 else ""

        with sns.axes_style("whitegrid"), plt.rc_context(STYLE_OVERRIDES):
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(14, 10),
                sharex=True,
                layout=ConstrainedLayoutEngine(
                    w_pad=0.3, h_pad=0.3, wspace=0.08, hspace=0.12
                ),
            )

            self._balance_graph(results, axes[0, 0], trend)
            self._reward_graph(results, axes[0, 1], trend)
            self._per_step_reward_graph(results, axes[1, 0])
            self._command_graph(results, axes[1, 1])

            fig.suptitle(title, fontsize=13, fontweight="bold")

            fig.savefig(
                filename, dpi=self._config.dpi, bbox_inches="tight", pad_inches=0.4
            )
            plt.close(fig)
