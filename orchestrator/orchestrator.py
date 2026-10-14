import pygame

from engine.config import SimulationStepConfig
from engine.simulation_engine import SimulationEngine
from engine.state import CartesianCoordinates
from orchestrator.config import OrchestratorConfig
from renderer.config import RenderFrameArgs
from renderer.renderer import Renderer
from scoring.scorekeeper import Score, ScoreKeeper


class Orchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        simulation_engine: SimulationEngine,
        renderer: Renderer,
        scorekeeper: ScoreKeeper,
    ) -> None:
        # window and keyboard only. pygame.init() would also open the unused audio device
        pygame.display.init()

        self._config = config

        self._time_accumulator = 0.0

        self._simulation_engine = simulation_engine

        self._state = self._simulation_engine.reset()

        self._renderer = renderer

        self._scorekeeper = scorekeeper

        self._current_score = Score(current_score=0, top_n=(), is_bob_above_rail=False)

        self._screen = pygame.display.set_mode(
            (self._config.renderer.window_width, self._config.renderer.window_height)
        )
        pygame.display.set_caption(self._config.window_title)

        self._clock = pygame.time.Clock()
        self._running = True

    def _read_command(self) -> float:
        keys = pygame.key.get_pressed()
        return keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]

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

            simulation_step_config = SimulationStepConfig(
                time_delta=self._config.simulation_dt,
                command=self._read_command(),
            )

            while self._time_accumulator >= self._config.simulation_dt:
                self._state = self._simulation_engine.step(simulation_step_config)

                self._current_score = self._scorekeeper.update_score(
                    self._state, self._config.simulation_dt
                )

                self._time_accumulator -= self._config.simulation_dt

            cartesian_coordinates = CartesianCoordinates.from_state(
                self._state, self._config.simulation.params.arm_length
            )

            render_frame_args = RenderFrameArgs(
                block_position=cartesian_coordinates.block_coordinates,
                bob_position=cartesian_coordinates.bob_coordinates,
                score=self._current_score,
            )

            self._renderer.render_frame(surface=self._screen, args=render_frame_args)
            pygame.display.flip()
