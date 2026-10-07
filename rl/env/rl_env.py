import math

import numpy as np

from engine.config import SimulationStepConfig
from engine.simulation_engine import SimulationEngine
from engine.state import SimulationState
from rl.env.config import RLEnvConfig
from rl.env.state import Action, RLEnvState, StepReward


class RLEnv:
    def __init__(self, config: RLEnvConfig) -> None:
        self._config = config

        self._simulation_engine = SimulationEngine(config.simulation)

        self._state = self._simulation_engine.reset()

        self._steps = 0

        self._action_low = -1
        self._action_high = 1

    def _get_reward(self, state: SimulationState) -> StepReward:
        upright_reward = self._config.reward.upright_weight * (
            (1 - math.cos(state.arm_angle)) / 2
        )

        rail_length = self._config.simulation.params.rail_length

        rail_centre_reward = -self._config.reward.rail_centre_weight * (
            ((state.block_position_along_rail - rail_length / 2) / (rail_length / 2))
            ** 2
        )

        z = state.angular_velocity**2 / (
            2
            * self._config.simulation.params.g
            / self._config.simulation.params.arm_length
        )

        spinning_reward = -self._config.reward.spin_weight * (z / (1 + z))

        return StepReward(
            upright_reward=upright_reward,
            rail_centre_reward=rail_centre_reward,
            spin_reward=spinning_reward,
            total_reward=upright_reward + rail_centre_reward + spinning_reward,
        )

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
            reward=StepReward(),
            terminated=False,
        )

    @property
    def config(self) -> RLEnvConfig:
        return self._config

    def sample_action_space(self) -> Action:
        return Action(command=np.random.uniform(self._action_low, self._action_high))

    def get_max_reward(self) -> float:
        return self._config.max_steps * self._config.reward.upright_weight
