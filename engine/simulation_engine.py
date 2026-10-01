import math
from collections.abc import Callable
from dataclasses import astuple
from enum import StrEnum

from engine.config import SimulationConfig, SimulationStepConfig
from engine.state import SimulationState, SimulationStateDerivative


class PinnedWallEnum(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class SimulationEngine:
    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self._block_mass = config.params.block_mass
        self._bob_mass = config.params.bob_mass
        self._arm_length = config.params.arm_length
        self._rail_friction_coefficient = config.params.rail_friction_coefficient
        self._g = 9.81  # m/s^2
        self._state = config.initial_state
        self._rail_length = config.params.rail_length
        self._max_force = config.params.max_force

        self._total_mass = self._block_mass + self._bob_mass

    def _f(self, state: SimulationState, force: float):
        """
        Derivation:
            F_n = F − c_r·ẋ                       net horizontal force on the block

            A = M + m                             total mass
            B = m·L·cosθ                          coupling between block and swing
            C = m·L²                              rotational inertia of the bob about the pivot

            R1 = F_n + m·L·θ̇²·sinθ                right side of the block equation
            R2 = −m·g·L·sinθ                      right side of the swing equation

            det = A·C − B²                        equals m·L²·(M + m·sin²θ), always positive when M > 0

            ẍ  = (C·R1 − B·R2) / det
            θ̈  = (A·R2 − B·R1) / det
        """
        (
            current_arm_angle,  # theta
            current_angular_velocity,  # theta dot
            current_block_position_along_rail,  # x
            current_block_velocity,  # x dot
        ) = astuple(state)

        net_horizontal_force = (
            force - self._rail_friction_coefficient * current_block_velocity
        )

        # [A, B] [x     dot dot] = [R1]
        # [B, C] [theta dot dot] = [R2]
        #
        # solving using cramer's rule

        A = self._total_mass
        B = self._bob_mass * self._arm_length * math.cos(current_arm_angle)
        C = self._bob_mass * (self._arm_length**2)

        R1 = net_horizontal_force + (
            self._bob_mass
            * self._arm_length
            * (current_angular_velocity**2)
            * math.sin(current_arm_angle)
        )

        R2 = -self._bob_mass * self._g * self._arm_length * math.sin(current_arm_angle)

        determinant = A * C - B**2

        block_acceleration = (C * R1 - B * R2) / determinant  # x dot dot
        angular_acceleration = (A * R2 - B * R1) / determinant  # theta dot dot

        return SimulationStateDerivative(
            angular_velocity=current_angular_velocity,
            angular_acceleration=angular_acceleration,
            block_velocity=current_block_velocity,
            block_acceleration=block_acceleration,
        )

    def _apply_wall_impact(self, state: SimulationState) -> SimulationState:
        (
            current_arm_angle,  # theta
            current_angular_velocity,  # theta dot
            current_block_position_along_rail,  # x
            current_block_velocity,  # x dot
        ) = astuple(state)

        if current_block_position_along_rail > self._rail_length:
            wall_x = self._rail_length
            moving_inward = current_block_velocity > 0
        elif current_block_position_along_rail < 0:
            wall_x = 0
            moving_inward = current_block_velocity < 0
        else:
            return state

        if moving_inward:
            current_angular_velocity += (
                current_block_velocity / self._arm_length
            ) * math.cos(current_arm_angle)
            current_block_velocity = 0

        current_block_position_along_rail = wall_x

        return SimulationState(
            arm_angle=current_arm_angle,
            angular_velocity=current_angular_velocity,
            block_position_along_rail=current_block_position_along_rail,
            block_velocity=current_block_velocity,
        )

    def _pinned_wall(self, state: SimulationState, force: float) -> PinnedWallEnum:
        free_acceleration = self._f(state, force).block_acceleration
        if (
            state.block_position_along_rail == self._rail_length
            and state.block_velocity == 0
            and free_acceleration > 0
        ):
            return PinnedWallEnum.RIGHT
        if (
            state.block_position_along_rail == 0
            and state.block_velocity == 0
            and free_acceleration < 0
        ):
            return PinnedWallEnum.LEFT
        return PinnedWallEnum.NONE

    def _pinned_f(self, state: SimulationState) -> SimulationStateDerivative:
        return SimulationStateDerivative(
            angular_velocity=state.angular_velocity,
            angular_acceleration=-(self._g / self._arm_length)
            * math.sin(state.arm_angle),
            block_velocity=0,
            block_acceleration=0,
        )

    def _rk4(
        self,
        state: SimulationState,
        derivative_fn: Callable[[SimulationState], SimulationStateDerivative],
        dt: float,
    ) -> SimulationState:
        """
        k1 = f(s,               F)
        k2 = f(s + (dt/2)·k1,   F)
        k3 = f(s + (dt/2)·k2,   F)
        k4 = f(s + dt·k3,       F)

        s_new = s + (dt/6)·(k1 + 2·k2 + 2·k3 + k4)
        """

        k1 = derivative_fn(state)

        k2 = derivative_fn(state + k1 * (dt / 2))

        k3 = derivative_fn(state + k2 * (dt / 2))

        k4 = derivative_fn(state + k3 * dt)

        return state + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)

    def step(self, config: SimulationStepConfig) -> SimulationState:

        force = self._max_force * config.command

        current_state = self._state

        if self._pinned_wall(current_state, force) != PinnedWallEnum.NONE:
            new_state = self._rk4(current_state, self._pinned_f, config.time_delta)
        else:
            new_state = self._rk4(
                current_state, lambda st: self._f(st, force), config.time_delta
            )
            new_state = self._apply_wall_impact(new_state)

        self._state = new_state
        return new_state
