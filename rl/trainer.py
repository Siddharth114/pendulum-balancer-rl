from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console

from replay.replay_animator import ReplayAnimator
from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.algorithms.episode import Episode
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

        results_dir = self._create_run_dir(
            algorithm_name=self._algorithm.name(),
            training_mode=self._training_mode,
        )

        with (
            Console().status("training"),
            self._metrics_writer.open(results_dir=results_dir) as write_metrics,
            self._recorder.open(
                results_dir=results_dir, env_config=self._algorithm.env_config
            ) as record_metrics,
            self._grapher.open(
                filename=results_dir / "results.png",
                title=f"{self._algorithm.name()} · {self._training_mode} · {num_episodes} episodes",
            ) as graph_metrics,
        ):

            def save(episode: Episode) -> None:
                write_metrics(episode)
                record_metrics(episode)
                graph_metrics(episode)

            self._algorithm.set_on_episode(save)
            self._algorithm.train(num_episodes)

        self._replay_animator.render(results_dir)

        return results_dir
