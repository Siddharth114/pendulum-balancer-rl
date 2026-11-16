from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rich.progress import track

from engine.state import SimulationState
from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.env.rl_env import RLEnv
from rl.env.state import Action


@dataclass(frozen=True)
class Discretisation:
    # representing the bins through normalised deviation from 0 -> 1
    # positive and negative
    arm_angle_edges: tuple[float, ...] = (
        0,
        0.0056,
        0.0111,
        0.0222,
        0.0444,
        0.0833,
        0.1667,
        0.3333,
    )

    angular_velocity_edges: tuple[float, ...] = (
        0,
        0.033,
        0.1,
        0.233,
    )

    # 4 intervals
    block_position_edges: tuple[float, ...] = (0, 0.33)

    # 4 intervals
    block_velocity_edges: tuple[float, ...] = (0, 0.33)

    def _identify_bin(
        self, edges: tuple[float, ...], min_value: float, max_value: float, value: float
    ) -> int:

        all_edges = [-i for i in edges[1:]][::-1] + list(edges)

        centre = (min_value + max_value) / 2
        span = max_value - min_value

        true_edges = (
            [min_value] + [centre + edge * span for edge in all_edges] + [max_value]
        )

        ind = 0

        for ind, edge in enumerate(true_edges):
            if value < edge:
                return ind

        return ind + 1

    def get_bins(
        self, state: SimulationState, rail_length: float
    ) -> tuple[int, int, int, int]:
        # (block_position_bin, block_velocity_bin, arm_angle_bin, angular_velocity_bin,)

        arm_angle_bin = self._identify_bin(
            edges=self.arm_angle_edges,
            min_value=0,
            max_value=2 * np.pi,
            value=state.arm_angle % (2 * np.pi),
        )

        angular_velocity_bin = self._identify_bin(
            edges=self.angular_velocity_edges,
            min_value=-3,
            max_value=3,
            value=state.angular_velocity,
        )

        block_position_bin = self._identify_bin(
            edges=self.block_position_edges,
            min_value=0,
            max_value=rail_length,
            value=state.block_position_along_rail,
        )

        block_velocity_bin = self._identify_bin(
            edges=self.block_velocity_edges,
            min_value=-2,
            max_value=2,
            value=state.block_velocity,
        )

        return (
            block_position_bin,
            block_velocity_bin,
            arm_angle_bin,
            angular_velocity_bin,
        )


@dataclass()
class BellmanConfig:
    action_space_items: int = 2

    discretisation: Discretisation = Discretisation()

    max_block_position: float = 0

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

    def _lookup_q_table(self, state: SimulationState) -> list[float]:

        (
            block_position_bin,
            block_velocity_bin,
            arm_angle_bin,
            angular_velocity_bin,
        ) = self._config.discretisation.get_bins(
            state=state, rail_length=self._config.max_block_position
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

    def train(self, num_episodes: int) -> None:

        for ind in track(
            range(num_episodes),
            description="training",
            auto_refresh=False,
        ):
            self._epsilon = self._decayed_epsilon(ind, num_episodes)
            self._run_episode(ind + 1)
