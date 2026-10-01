from dataclasses import dataclass

from engine.state import SimulationState


@dataclass
class SimulationParameters:
    block_mass: float  # kg
    bob_mass: float  # kg
    arm_length: float  # m
    rail_friction_coefficient: float = 0
    rail_length: float = 0
    max_force: float = 0  # Newtons

    def __post_init__(self):
        if self.max_force <= 0:
            raise ValueError("Max force must be non-negative")
        if self.rail_length <= 0:
            raise ValueError("Rail length must be non-negative")
        if self.block_mass <= 0:
            raise ValueError("Block mass must be non-negative")
        if self.bob_mass <= 0:
            raise ValueError("Bob mass must be non-negative")


@dataclass
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
