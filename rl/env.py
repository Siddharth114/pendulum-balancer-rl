import math
from dataclasses import dataclass

import numpy as np

from engine.config import (
    DEFAULT_SIMULATION_CONFIG,
    EASY_SIMULATION_CONFIG,
    SimulationConfig,
    SimulationStepConfig,
)
from engine.simulation_engine import SimulationEngine
from engine.state import SimulationState


@dataclass(frozen=True)
class RewardConfig:
    upright_weight: float = 1.0
    rail_centre_weight: float = 0.1
    spin_weight: float = 0.1


DEFAULT_REWARD_CONFIG = RewardConfig()


@dataclass(frozen=True)
class RLEnvConfig:
    simulation: SimulationConfig = EASY_SIMULATION_CONFIG
    reward: RewardConfig = DEFAULT_REWARD_CONFIG

    simulation_dt: float = 1 / 240

    substeps: int = 8
    max_steps: int = 300  # max steps before truncating
    terminate_on_fall: bool = True


RL_ENV_MODES = {
    "easy": RLEnvConfig(
        simulation=EASY_SIMULATION_CONFIG, max_steps=300, terminate_on_fall=True
    ),
    "normal": RLEnvConfig(
        simulation=DEFAULT_SIMULATION_CONFIG, max_steps=600, terminate_on_fall=False
    ),
}


@dataclass(frozen=True)
class Action:
    command: (
        float  # in the range [-1, 1] i.e., left to right. a fraction of the max force
    )

    def __post_init__(self):
        if not (-1 <= self.command <= 1):
            raise ValueError("command must be between -1 and 1")


@dataclass(frozen=True)
class RLEnvState(SimulationState):
    reward: float = 0
    terminated: bool = False
    truncated: bool = False


class RLEnv:
    def __init__(self, config: RLEnvConfig) -> None:
        self._config = config

        self._simulation_engine = SimulationEngine(config.simulation)

        self._state = self._simulation_engine.reset()

        self._steps = 0

        self._action_low = -1
        self._action_high = 1

    def _get_reward(self, state: SimulationState) -> float:
        upright_reward = self._config.reward.upright_weight * (
            (1 - math.cos(state.arm_angle)) / 2
        )

        rail_length = self._config.simulation.params.rail_length

        rail_centre_reward = self._config.reward.rail_centre_weight * (
            ((state.block_position_along_rail - rail_length / 2) / (rail_length / 2))
            ** 2
        )

        z = state.angular_velocity**2 / (
            2
            * self._config.simulation.params.g
            / self._config.simulation.params.arm_length
        )

        spinning_reward = self._config.reward.spin_weight * (z / (1 + z))

        return upright_reward - rail_centre_reward - spinning_reward

    def step(self, action: Action) -> RLEnvState:
        simulation_step_config = SimulationStepConfig(
            time_delta=self._config.simulation_dt,
            command=action.command,
        )

        for _ in range(self._config.substeps):
            self._state = self._simulation_engine.step(simulation_step_config)

        self._steps += 1

        return RLEnvState(
            **self._state.__dict__,
            reward=self._get_reward(self._state),
            terminated=self._config.terminate_on_fall
            and not self._state.is_bob_above_rail,
            truncated=self._steps >= self._config.max_steps,
        )

    def reset(self) -> RLEnvState:
        self._state = self._simulation_engine.reset()

        self._steps = 0

        return RLEnvState(
            **self._state.__dict__,
            reward=0,
            terminated=False,
        )

    def sample_action_space(self) -> Action:
        return Action(command=np.random.uniform(self._action_low, self._action_high))
