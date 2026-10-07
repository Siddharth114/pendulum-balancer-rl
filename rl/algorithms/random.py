from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.algorithms.episode import Episode
from rl.env.config import RLEnvConfig
from rl.env.state import Action, RLEnvState


class Random(BaseAlgorithm):
    def __init__(self, config: RLEnvConfig) -> None:
        super().__init__(config)

    def name(self) -> str:
        return "Random Baseline"

    def action(self, state: RLEnvState) -> Action:
        return self._env.sample_action_space()

    def train(self, num_episodes: int) -> list[Episode]:
        return [self._run_episode(ind + 1) for ind in range(num_episodes)]
