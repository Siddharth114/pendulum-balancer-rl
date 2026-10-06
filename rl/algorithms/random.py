from rl.algorithms.base_algorithm import BaseAlgorithm, EpisodeResult
from rl.env import Action, RLEnvConfig, RLEnvState


class Random(BaseAlgorithm):
    def __init__(self, config: RLEnvConfig) -> None:
        super().__init__(config)

    def name(self) -> str:
        return "Random Baseline"

    def action(self, state: RLEnvState) -> Action:
        return self._env.sample_action_space()

    def train(self, num_episodes: int) -> list[EpisodeResult]:
        episode_results = []

        for ind in range(num_episodes):
            episode_results.append(self._run_episode(ind + 1))

        return episode_results
