from pathlib import Path

from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.algorithms.training_grapher import TrainingGrapher


class Trainer:
    def __init__(self, algorithm: BaseAlgorithm, grapher: TrainingGrapher) -> None:
        self._algorithm = algorithm
        self._grapher = grapher

    def run(self, num_episodes: int) -> Path:
        results = self._algorithm.train(num_episodes)
        return self._grapher.plot_results(results, self._algorithm.name())
