from pathlib import Path

from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.run_writer import RunWriter
from rl.training_grapher import TrainingGrapher


class Trainer:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        writer: RunWriter,
        grapher: TrainingGrapher,
        mode: str,
    ) -> None:
        self._algorithm = algorithm
        self._grapher = grapher
        self._writer = writer
        self._mode = mode

    def run(self, num_episodes: int) -> Path:
        results = self._algorithm.train(num_episodes)
        results_dir = self._writer.create_run_dir(self._algorithm.name(), self._mode)
        self._writer.write_results(results_dir, results)
        self._grapher.plot_results(results, results_dir / "graph.png")
        return results_dir
