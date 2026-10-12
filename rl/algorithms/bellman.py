import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rich.progress import track

from engine.state import SimulationState
from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.algorithms.episode import Episode
from rl.env.rl_env import RLEnv
from rl.env.state import Action


@dataclass()
class BellmanConfig:
    action_space_items: int = 5

    min_arm_angle: float = 0
    max_arm_angle: float = 2 * np.pi
    arm_angle_intervals: int = 36

    min_block_position: float = 0
    max_block_position: float = (
        0  # will be filled after initialisation of algo with the env config
    )
    block_position_intervals: int = 4

    # hueristic based
    min_block_velocity: float = -2
    max_block_velocity: float = 2
    block_velocity_intervals: int = 4

    # hueristic based
    min_angular_velocity: float = -3
    max_angular_velocity: float = 3
    angular_velocity_intervals: int = 4

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_episodes_fraction: float = 0.8

    learning_rate: float = 0.1  # alpha
    discount_factor: float = 0.99  # gamma


class Bellman(BaseAlgorithm):
    cli_name = "bellman"

    def __init__(self, env: RLEnv, config: BellmanConfig | None = None) -> None:
        super().__init__(env)
        self._config = config or BellmanConfig()

        self._config.max_block_position = self.env_config.simulation.params.rail_length

        self._action_space = np.linspace(
            self._env._action_low,
            self._env._action_high,
            self._config.action_space_items,
        ).tolist()

        # Q Table is a map of vector of bin of state vars -> reward for each discrete action
        # (x, xdot, theta, thetadot) -> (reward for each discrete action)
        self._q_table = defaultdict(lambda: [0.0] * self._config.action_space_items)

        self._last_action_index = 0

        self._rng = np.random.default_rng()

        self._epsilon = self._config.epsilon_start

    def name(self) -> str:
        return "Bellman Q Learning"

    def _identify_bin(
        self, min_value: float, max_value: float, value: float, num_bins: int
    ) -> int:
        fraction = (value - min_value) / (max_value - min_value)

        clipped_bin = min(max(0, math.floor(fraction * num_bins)), num_bins - 1)

        return clipped_bin

    def _wrap_angle(self, angle: float):
        return angle % (2 * np.pi)

    def _lookup_q_table(self, state: SimulationState) -> list[float]:
        block_position_bin = self._identify_bin(
            min_value=self._config.min_block_position,
            max_value=self._config.max_block_position,
            value=state.block_position_along_rail,
            num_bins=self._config.block_position_intervals,
        )

        block_velocity_bin = self._identify_bin(
            min_value=self._config.min_block_velocity,
            max_value=self._config.max_block_velocity,
            value=state.block_velocity,
            num_bins=self._config.block_velocity_intervals,
        )

        arm_angle_bin = self._identify_bin(
            min_value=self._config.min_arm_angle,
            max_value=self._config.max_arm_angle,
            value=self._wrap_angle(state.arm_angle),
            num_bins=self._config.arm_angle_intervals,
        )

        angular_velocity_bin = self._identify_bin(
            min_value=self._config.min_angular_velocity,
            max_value=self._config.max_angular_velocity,
            value=state.angular_velocity,
            num_bins=self._config.angular_velocity_intervals,
        )

        q_table_vector = self._q_table[
            (
                block_position_bin,
                block_velocity_bin,
                arm_angle_bin,
                angular_velocity_bin,
            )
        ]

        return q_table_vector

    def _update(
        self,
        current_state: SimulationState,
        action_taken: Action,
        new_state: SimulationState,
        reward: float,
        terminated: bool,
    ) -> None:
        """
        Q(s, a) ← Q(s, a) + α · [r + γ · max over a’ of Q(s’, a’) − Q(s, a)]
        """

        current_q_vector = self._lookup_q_table(current_state)

        # Q(s, a)
        current_q_value = current_q_vector[self._last_action_index]

        # max over a’ of Q(s’, a’)
        max_new_q_value = max(self._lookup_q_table(new_state))

        if not terminated:
            updated_current_q_value = current_q_value + self._config.learning_rate * (
                reward
                + (self._config.discount_factor * max_new_q_value)
                - current_q_value
            )
        else:
            updated_current_q_value = current_q_value + self._config.learning_rate * (
                reward - current_q_value
            )

        current_q_vector[self._last_action_index] = updated_current_q_value

    def action(self, state: SimulationState) -> Action:
        q_table_vector = np.array(self._lookup_q_table(state))

        if self._rng.random() < self._epsilon:
            self._last_action_index = self._rng.choice(range(len(self._action_space)))
        else:
            self._last_action_index = self._rng.choice(
                np.flatnonzero(q_table_vector == q_table_vector.max())
            )  # not using argmax for randomised tiebreaks

        return Action(command=self._action_space[self._last_action_index])

    def _decayed_epsilon(self, episode_index: int, num_episodes: int) -> float:
        # linear epsilon decay
        decay_episodes = max(1, num_episodes * self._config.decay_episodes_fraction)
        progress = min(1.0, episode_index / decay_episodes)

        return self._config.epsilon_start + progress * (
            self._config.epsilon_end - self._config.epsilon_start
        )

    def train(self, num_episodes: int) -> list[Episode]:
        episodes = []

        for ind in track(
            range(num_episodes),
            description="training",
            auto_refresh=False,
        ):
            self._epsilon = self._decayed_epsilon(ind, num_episodes)
            episodes.append(self._run_episode(ind + 1))

        return episodes
