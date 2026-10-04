from dataclasses import dataclass

from engine.config import SimulationConfig, SimulationParameters
from engine.state import SimulationState
from renderer.config import RendererConfig

_RAIL_LENGTH = 10  # metres

DEFAULT_RENDERER_CONFIG = RendererConfig()

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


@dataclass(frozen=True)
class OrchestratorConfig:
    simulation: SimulationConfig = DEFAULT_SIMULATION_CONFIG
    renderer: RendererConfig = DEFAULT_RENDERER_CONFIG

    window_title: str = "Pendulum Balancer"
    framerate: int = 60

    simulation_dt: float = 1 / 240  # fixed time step seen by the simulation engine
    max_frame_time: float = (
        0.25  # seconds. how much one frame of game loop can add to sim time
    )
