from dataclasses import dataclass

from engine.config import (
    DEFAULT_SIMULATION_CONFIG,
    EASY_SIMULATION_CONFIG,
    SimulationConfig,
)


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

    @property
    def step_dt(self) -> float:
        """seconds of simulated time per env step (one decision)"""
        return self.simulation_dt * self.substeps


RL_ENV_MODES = {
    "easy": RLEnvConfig(
        simulation=EASY_SIMULATION_CONFIG, max_steps=300, terminate_on_fall=True
    ),
    "normal": RLEnvConfig(
        simulation=DEFAULT_SIMULATION_CONFIG, max_steps=600, terminate_on_fall=False
    ),
}
