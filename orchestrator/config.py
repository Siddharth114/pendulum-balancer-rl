from dataclasses import dataclass

from engine.config import DEFAULT_SIMULATION_CONFIG, SimulationConfig
from renderer.config import RendererConfig
from scoring.config import ScoringConfig

DEFAULT_RENDERER_CONFIG = RendererConfig()

DEFAULT_SCORING_CONFIG = ScoringConfig()


@dataclass(frozen=True)
class OrchestratorConfig:
    simulation: SimulationConfig = DEFAULT_SIMULATION_CONFIG
    renderer: RendererConfig = DEFAULT_RENDERER_CONFIG
    scoring: ScoringConfig = DEFAULT_SCORING_CONFIG

    window_title: str = "Pendulum Balancer"
    framerate: int = 60

    simulation_dt: float = 1 / 240  # fixed time step seen by the simulation engine
    max_frame_time: float = (
        0.25  # seconds. how much one frame of game loop can add to sim time
    )
