from dataclasses import dataclass
import math
import numpy as np
from rich.progress import track
from engine.state import SimulationState
from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.env.rl_env import RLEnv
from rl.env.state import Action


@dataclass()
class LinearQLearningConfig:
    action_space_items: int = 3

    min_block_velocity: float = -2
    max_block_velocity: float = 2

    min_angular_velocity: float = -3
    max_angular_velocity: float = 3

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_episodes_fraction: float = 0.8

    learning_rate: float = 0.01
    discount_factor: float = 0.99


class LinearQLearning(BaseAlgorithm):
    cli_name = "linear-q-learning"

    def __init__(self, env: RLEnv, config: LinearQLearningConfig | None = None) -> None:
        super().__init__(env)

        self._config = config or LinearQLearningConfig()

        self._action_space = np.linspace(
            self._env._action_low,
            self._env._action_high,
            self._config.action_space_items,
        ).tolist()

        self._initial_features = self._get_features(
            self.env_config.simulation.initial_state
        )

        self._weights = np.zeros(
            (self._config.action_space_items, len(self._initial_features))
        )

        self._rng = np.random.default_rng()

        self._epsilon = self._config.epsilon_start

        self._last_action_index = 0

    def name(self) -> str:
        return "Linear Q Learning"

    def _get_features(self, state: SimulationState) -> np.ndarray:
        # φ = [1, sin θ, cos θ, x̃, ẋ̃, θ̇̃, x̃², θ̇̃², sin θ·θ̇̃, x̃·ẋ̃]
        # Features:
        # sin theta, cos theta, block position centred and scaled to [-1,1]
        # block velocity divided by rough max
        # angular velocity divided by rough max

        sin_theta = math.sin(state.arm_angle)
        cos_theta = math.cos(state.arm_angle)

        scaled_block_position = -1 + (
            (2 * state.block_position_along_rail)
            / self.env_config.simulation.params.rail_length
        )

        scaled_block_velocity = -1 + (
            ((state.block_velocity - self._config.min_block_velocity) * 2)
            / (self._config.max_block_velocity - self._config.min_block_velocity)
        )

        scaled_angular_velocity = -1 + (
            ((state.angular_velocity - self._config.min_angular_velocity) * 2)
            / (self._config.max_angular_velocity - self._config.min_angular_velocity)
        )

        return np.array(
            [
                1,
                sin_theta,
                cos_theta,
                scaled_block_position,
                scaled_block_velocity,
                scaled_angular_velocity,
                scaled_block_position**2,
                scaled_angular_velocity**2,
                sin_theta * scaled_angular_velocity,
                scaled_block_position * scaled_block_velocity,
            ]
        )

    def _get_q_table(
        self,
        state: SimulationState,
    ) -> np.ndarray:

        features = self._get_features(state)

        return self._weights @ features

    def _decayed_epsilon(self, episode_index: int, num_episodes: int) -> float:
        # linear epsilon decay
        decay_episodes = max(1, num_episodes * self._config.decay_episodes_fraction)
        progress = min(1.0, episode_index / decay_episodes)

        return self._config.epsilon_start + progress * (
            self._config.epsilon_end - self._config.epsilon_start
        )

    def action(self, state: SimulationState) -> Action:

        # shape is (no of actions, 1)
        q_table_vector = self._get_q_table(state)

        if self._rng.random() < self._epsilon:
            self._last_action_index = self._rng.choice(range(len(self._action_space)))
        else:
            self._last_action_index = self._arg_max_q(q_table_vector=q_table_vector)
        return Action(command=self._action_space[self._last_action_index])

    def _arg_max_q(self, q_table_vector: np.ndarray) -> int:
        return int(
            self._rng.choice(np.flatnonzero(q_table_vector == q_table_vector.max()))
        )

    def _update(
        self,
        current_state: SimulationState,
        action_taken: Action,
        new_state: SimulationState,
        reward: float,
        terminated: bool,
    ) -> None:

        target = reward

        current_features = self._get_features(current_state)
        last_action_index = self._last_action_index

        if not terminated:
            target += self._config.discount_factor * self._get_q_table(new_state).max()

        error = target - self._weights[last_action_index] @ current_features

        self._weights[last_action_index] += (
            self._config.learning_rate * error * current_features
        )

    def train(self, num_episodes: int) -> None:

        for ind in track(
            range(num_episodes),
            description="training",
            auto_refresh=False,
        ):
            self._epsilon = self._decayed_epsilon(ind, num_episodes)
            self._run_episode(ind + 1)
