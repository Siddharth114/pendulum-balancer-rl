from dataclasses import dataclass

import pygame

from engine.config import SimulationConfig, SimulationParameters, SimulationStepConfig
from engine.simulation_engine import SimulationEngine
from engine.state import SimulationState
from player.human_player import HumanPlayer
from renderer import Renderer, RendererConfig, RenderFrameArgs


@dataclass(frozen=True)
class OrchestratorConfig:
    window_width: int = 800  # pixels
    window_height: int = 600  # pixels
    window_title: str = "Pendulum Balancer"

    framerate: int = 60

    rail_margin: int = 150  # pixels
    rail_thickness: int = 10  # pixels
    rail_length: int = 10  # metres

    arm_length: int = 2  # metres

    block_radius: int = 15  # pixels
    bob_radius: int = 10  # pixels
    rod_thickness: int = 5  # pixels

    simulation_dt: float = 1 / 240  # fixed time step seen by the simulation engine
    max_frame_time: float = (
        0.25  # seconds. how much one frame of game loop can add to sim time
    )


class Orchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config

        self._renderer_config = RendererConfig(
            window_width=self._config.window_width,
            window_height=self._config.window_height,
            rail_thickness=self._config.rail_thickness,
            rail_length=self._config.rail_length,
            rail_margin=self._config.rail_margin,
            block_radius=self._config.block_radius,
            bob_radius=self._config.bob_radius,
            rod_thickness=self._config.rod_thickness,
        )

        self._player = HumanPlayer()

        self._time_accumulator = 0.0

        self._simulation_config = SimulationConfig(
            params=SimulationParameters(
                block_mass=15,
                bob_mass=5,
                arm_length=self._config.arm_length,
                rail_friction_coefficient=20,
                rail_length=self._config.rail_length,
                max_force=100,
            ),
            initial_state=SimulationState(
                arm_angle=0,
                angular_velocity=0,
                block_position_along_rail=self._config.rail_length / 2,
                block_velocity=0,
            ),
        )

        self._simulation_engine = SimulationEngine(self._simulation_config)

        self._state = self._simulation_config.initial_state

        self._renderer = Renderer(self._renderer_config)

        pygame.init()
        self._screen = pygame.display.set_mode(
            (self._config.window_width, self._config.window_height)
        )
        pygame.display.set_caption(self._config.window_title)

        self._clock = pygame.time.Clock()
        self._running = True

    def run(self) -> None:
        while self._running:
            real_time = min(
                self._clock.tick(self._config.framerate) / 1000,
                self._config.max_frame_time,
            )

            self._time_accumulator += real_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

            move = self._player.move(self._state)

            simulation_step_config = SimulationStepConfig(
                time_delta=self._config.simulation_dt,
                command=move.command,
            )

            while self._time_accumulator >= self._config.simulation_dt:
                self._state = self._simulation_engine.step(simulation_step_config)
                self._time_accumulator -= self._config.simulation_dt

            cartesian_coordinates = self._simulation_engine.get_cartesian_coordinates(
                self._state
            )

            render_frame_args = RenderFrameArgs(
                block_position=cartesian_coordinates.block_coordinates,
                bob_position=cartesian_coordinates.bob_coordinates,
            )

            self._renderer.render_frame(surface=self._screen, args=render_frame_args)
            pygame.display.flip()


if __name__ == "__main__":
    orchestrator = Orchestrator(OrchestratorConfig())
    orchestrator.run()
