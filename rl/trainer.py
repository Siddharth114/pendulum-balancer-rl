from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console

from replay.replay_animator import ReplayAnimator
from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.grapher import Grapher
from rl.metrics_writer import MetricsWriter
from rl.recorder import Recorder


@dataclass(frozen=True)
class TrainerConfig:
    results_dir: Path = Path("results")


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        algorithm: BaseAlgorithm,
        training_mode: str,
        metrics_writer: MetricsWriter,
        recorder: Recorder,
        grapher: Grapher,
        replay_animator: ReplayAnimator,
    ) -> None:
        self._config = config
        self._algorithm = algorithm
        self._training_mode = training_mode
        self._metrics_writer = metrics_writer
        self._recorder = recorder
        self._grapher = grapher
        self._replay_animator = replay_animator

    def _slugify(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def _create_run_dir(self, algorithm_name: str, training_mode: str) -> Path:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(
            f"{self._config.results_dir}/{self._slugify(algorithm_name)}_{training_mode}_{timestamp}"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        return results_dir

    def run(self, num_episodes: int) -> Path:

        episodes = self._algorithm.train(num_episodes)

        results_dir = self._create_run_dir(
            algorithm_name=self._algorithm.name(),
            training_mode=self._training_mode,
        )

        with Console().status("saving metrics, recording and graph"):
            self._metrics_writer.write_metrics(
                results_dir=results_dir,
                episodes=episodes,
            )

            self._recorder.write_recordings(
                results_dir=results_dir,
                episodes=episodes,
                env_config=self._algorithm.env_config,
            )

            self._grapher.plot_results(
                results=episodes,
                filename=results_dir / "results.png",
                title=f"{self._algorithm.name()} · {self._training_mode} · {num_episodes} episodes",
            )

        self._replay_animator.render(results_dir)

        return results_dir
