import math
from dataclasses import dataclass


@dataclass
class SimulationStateDerivative:
    angular_velocity: float  # radians per second
    angular_acceleration: float  # radians per second squared
    block_velocity: float  # m/s
    block_acceleration: float  # m/s^2

    def __mul__(self, constant: float) -> SimulationStateDerivative:
        return SimulationStateDerivative(
            angular_velocity=self.angular_velocity * constant,
            angular_acceleration=self.angular_acceleration * constant,
            block_velocity=self.block_velocity * constant,
            block_acceleration=self.block_acceleration * constant,
        )

    def __add__(self, other: SimulationStateDerivative) -> SimulationStateDerivative:
        return SimulationStateDerivative(
            angular_velocity=self.angular_velocity + other.angular_velocity,
            angular_acceleration=self.angular_acceleration + other.angular_acceleration,
            block_velocity=self.block_velocity + other.block_velocity,
            block_acceleration=self.block_acceleration + other.block_acceleration,
        )


@dataclass(frozen=True)
class SimulationState:
    arm_angle: float = 0  # radians
    angular_velocity: float = 0  # radians per second
    block_position_along_rail: float = 0  # from left corner of rail
    block_velocity: float = 0  # m/s

    def __add__(self, derivative: SimulationStateDerivative) -> SimulationState:
        return SimulationState(
            arm_angle=self.arm_angle + derivative.angular_velocity,
            angular_velocity=self.angular_velocity + derivative.angular_acceleration,
            block_position_along_rail=self.block_position_along_rail
            + derivative.block_velocity,
            block_velocity=self.block_velocity + derivative.block_acceleration,
        )

    @property
    def is_bob_above_rail(self) -> bool:
        return math.cos(self.arm_angle) < 0


@dataclass(frozen=True)
class CartesianCoordinates:
    block_coordinates: tuple[float, float]
    bob_coordinates: tuple[float, float]
