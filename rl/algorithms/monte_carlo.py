from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from rich.progress import track

from engine.state import SimulationState
from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.env.rl_env import RLEnv
from rl.env.state import Action


@dataclass()
class MonteCarloConfig:
    action_space_items: int = 2

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

    discount_factor: float = 0.99  # gamma


class MonteCarlo(BaseAlgorithm):
    cli_name = "monte-carlo"

    def __init__(self, env: RLEnv, config: MonteCarloConfig | None = None) -> None:
        super().__init__(env)

        self._config: MonteCarloConfig = config or MonteCarloConfig()

        self._config.max_block_position = self.env_config.simulation.params.rail_length

        self._action_space: list[float] = np.linspace(
            self._env._action_low,
            self._env._action_high,
            self._config.action_space_items,
        ).tolist()

        # Q Table is a map of vector of bin of state vars -> reward for each discrete action
        # (x, xdot, theta, thetadot) -> (reward for each discrete action)
        self._q_table = defaultdict(lambda: [0.0] * self._config.action_space_items)

        self._visit_count = defaultdict(lambda: [0] * self._config.action_space_items)
        self._trajectory: list[
            tuple[tuple[int, int, int, int], int, float]
        ] = []  # (state bin, last action index, reward for action)

        self._final_state_bin: tuple[int, int, int, int] | None = None
        self._terminated: bool = False

        self._last_action_index: int = 0

        self._rng = np.random.default_rng()

        self._epsilon = self._config.epsilon_start

    def name(self) -> str:
        return "Monte Carlo"

    def _identify_bin(
        self, min_value: float, max_value: float, value: float, num_bins: int
    ) -> int:
        fraction = (value - min_value) / (max_value - min_value)

        clipped_bin = min(max(0, math.floor(fraction * num_bins)), num_bins - 1)

        return clipped_bin

    def _wrap_angle(self, angle: float):
        return angle % (2 * np.pi)

    def _get_state_bins(self, state: SimulationState) -> tuple[int, int, int, int]:
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

        return (
            block_position_bin,
            block_velocity_bin,
            arm_angle_bin,
            angular_velocity_bin,
        )

    def action(self, state: SimulationState) -> Action:
        q_table_vector = np.array(self._q_table[self._get_state_bins(state)])
        if self._rng.random() < self._epsilon:
            self._last_action_index = int(
                self._rng.choice(range(len(self._action_space)))
            )
        else:
            self._last_action_index = int(
                self._rng.choice(np.flatnonzero(q_table_vector == q_table_vector.max()))
            )

        return Action(command=self._action_space[self._last_action_index])

    def _update(
        self,
        current_state: SimulationState,
        action_taken: Action,
        new_state: SimulationState,
        reward: float,
        terminated: bool,
    ) -> None:

        self._trajectory.append(
            (self._get_state_bins(current_state), self._last_action_index, reward)
        )
        self._final_state_bin = self._get_state_bins(new_state)
        self._terminated = terminated

    def _learn_from_episode(self) -> None:
        if self._terminated:
            g = 0
        else:
            g = max(
                self._q_table.get(
                    self._final_state_bin, [0] * self._config.action_space_items
                )
            )

        for key, action_index, reward in reversed(self._trajectory):
            g = reward + self._config.discount_factor * g

            self._visit_count[key][action_index] += 1

            q = self._q_table[key][action_index]

            # Q[key][a] += (g − Q[key][a]) / N[key][a]

            self._q_table[key][action_index] += (g - q) / self._visit_count[key][
                action_index
            ]

        self._trajectory.clear()

    def _decayed_epsilon(self, episode_index: int, num_episodes: int) -> float:
        # linear epsilon decay
        decay_episodes = max(1, num_episodes * self._config.decay_episodes_fraction)
        progress = min(1.0, episode_index / decay_episodes)

        return self._config.epsilon_start + progress * (
            self._config.epsilon_end - self._config.epsilon_start
        )

    def train(self, num_episodes: int) -> None:
        for ind in track(
            range(num_episodes), description="training", auto_refresh=False
        ):
            self._epsilon = self._decayed_epsilon(ind, num_episodes)
            self._run_episode(ind + 1)
            self._learn_from_episode()
