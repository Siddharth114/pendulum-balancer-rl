import pygame

from engine.config import SimulationStepConfig
from engine.simulation_engine import SimulationEngine
from orchestrator.config import OrchestratorConfig
from player.human_player import HumanPlayer
from renderer.config import RenderFrameArgs
from renderer.renderer import Renderer


class Orchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        pygame.init()

        self._config = config

        self._player = HumanPlayer()

        self._time_accumulator = 0.0

        self._simulation_engine = SimulationEngine(self._config.simulation)

        self._state = self._config.simulation.initial_state

        self._renderer = Renderer(
            self._config.renderer,
            rail_length=self._config.simulation.params.rail_length,
        )

        self._screen = pygame.display.set_mode(
            (self._config.renderer.window_width, self._config.renderer.window_height)
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
