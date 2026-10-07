from abc import ABC, abstractmethod

from rl.algorithms.episode import Episode, EpisodeMetrics, EpisodeRecording
from rl.env.config import RLEnvConfig
from rl.env.rl_env import RLEnv
from rl.env.state import Action, RLEnvState


class BaseAlgorithm(ABC):
    def __init__(self, config: RLEnvConfig) -> None:
        self._env = RLEnv(config)
        super().__init__()

    @property
    def env_config(self) -> RLEnvConfig:
        return self._env.config

    @abstractmethod
    def action(self, state: RLEnvState) -> Action:
        raise NotImplementedError

    def _run_episode(self, episode_number: int) -> Episode:
        max_reward = self._env.get_max_reward()

        total_reward = 0
        episode_length = 0
        streak = 0
        best = 0
        state = self._env.reset()

        total_upright_reward = 0
        total_rail_centre_reward = 0
        total_spin_reward = 0

        total_command = 0
        prev_command = None
        total_command_change = 0

        states = [state.to_simulation_state()]
        actions = []

        while True:
            episode_length += 1

            action = self.action(state)

            state = self._env.step(action)

            if state.is_bob_above_rail:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0

            step_reward = state.reward

            total_reward += step_reward.total_reward
            total_upright_reward += step_reward.upright_reward
            total_rail_centre_reward += step_reward.rail_centre_reward
            total_spin_reward += step_reward.spin_reward

            states.append(state.to_simulation_state())
            actions.append(action)

            total_command += abs(action.command)
            if prev_command is not None:
                total_command_change += abs(action.command - prev_command)
            prev_command = action.command

            if state.terminated or state.truncated:
                break

        avg_upright_reward = total_upright_reward / episode_length
        avg_rail_centre_reward = total_rail_centre_reward / episode_length
        avg_spin_reward = total_spin_reward / episode_length
        avg_total_reward = total_reward / episode_length
        avg_command = total_command / episode_length
        avg_command_change = total_command_change / max(episode_length - 1, 1)

        return Episode(
            number=episode_number,
            metrics=EpisodeMetrics(
                length=episode_length,
                longest_balance=best,
                total_reward=total_reward,
                max_reward=max_reward,
                avg_upright_reward=avg_upright_reward,
                avg_rail_centre_reward=avg_rail_centre_reward,
                avg_spin_reward=avg_spin_reward,
                avg_total_reward=avg_total_reward,
                avg_command=avg_command,
                avg_command_change=avg_command_change,
            ),
            recording=EpisodeRecording(
                states=states,
                actions=actions,
                terminated=state.terminated,
            ),
        )

    @abstractmethod
    def train(self, num_episodes: int) -> list[Episode]:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
