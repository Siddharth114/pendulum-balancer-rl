import math
from dataclasses import dataclass, replace

from engine.state import SimulationState


@dataclass(frozen=True)
class SimulationParameters:
    block_mass: float  # kg
    bob_mass: float  # kg
    arm_length: float  # m
    rail_length: float  # m
    max_force: float  # Newtons
    rail_friction_coefficient: float = 0
    g: float = 9.81  # m/s^2

    def __post_init__(self):
        if self.max_force <= 0:
            raise ValueError("Max force must be positive")
        if self.rail_length <= 0:
            raise ValueError("Rail length must be positive")
        if self.block_mass <= 0:
            raise ValueError("Block mass must be positive")
        if self.bob_mass <= 0:
            raise ValueError("Bob mass must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    params: SimulationParameters
    initial_state: SimulationState

    def __post_init__(self):
        if (
            self.initial_state.block_position_along_rail < 0
            or self.initial_state.block_position_along_rail > self.params.rail_length
        ):
            raise ValueError("Block position should be within rail limits")


@dataclass(frozen=True)
class SimulationStepConfig:
    time_delta: float  # seconds
    command: float  # in the range [-1, 1] i.e., left to right. a fraction of the force left or right

    def __post_init__(self):
        if self.command < -1 or self.command > 1:
            raise ValueError("Command must be in the range [-1, 1]")

        if self.time_delta <= 0:
            raise ValueError("Time delta must be positive")


_RAIL_LENGTH = 10  # metres

DEFAULT_SIMULATION_CONFIG = SimulationConfig(
    params=SimulationParameters(
        block_mass=15,
        bob_mass=5,
        arm_length=2,
        rail_length=_RAIL_LENGTH,
        max_force=100,
        rail_friction_coefficient=20,
    ),
    initial_state=SimulationState(
        arm_angle=0,
        angular_velocity=0,
        block_position_along_rail=_RAIL_LENGTH / 2,
        block_velocity=0,
    ),
)

# starts nearly upright with a slight lean
EASY_SIMULATION_CONFIG = replace(
    DEFAULT_SIMULATION_CONFIG,
    initial_state=replace(
        DEFAULT_SIMULATION_CONFIG.initial_state, arm_angle=math.pi - 0.05
    ),
)

SIMULATION_MODES = {
    "normal": DEFAULT_SIMULATION_CONFIG,
    "easy": EASY_SIMULATION_CONFIG,
}
