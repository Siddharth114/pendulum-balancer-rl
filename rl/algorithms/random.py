from rl.algorithms.base_algorithm import BaseAlgorithm, TrainingResults
from rl.env import Action, RLEnvConfig, RLEnvState


class Random(BaseAlgorithm):
    def __init__(self, config: RLEnvConfig) -> None:
        super().__init__(config)

    def name(self) -> str:
        return "Random Agent - Baseline"

    def action(self, state: RLEnvState) -> Action:
        return self._env.sample_action_space()

    def train(self, num_episodes: int) -> TrainingResults:
        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            total_reward = 0
            episode_length = 0
            state = self._env.reset()

            while True:
                episode_length += 1

                action = self.action(state)

                state = self._env.step(action)

                total_reward += state.reward

                if state.terminated or state.truncated:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)

        return TrainingResults(episode_rewards, episode_lengths)
