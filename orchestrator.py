from dataclasses import dataclass

import pygame

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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

            render_frame_args = RenderFrameArgs(
                block_position=(self._config.rail_length // 2, 0),
                bob_position=(self._config.rail_length // 2, self._config.arm_length),
            )

            self._renderer.render_frame(surface=self._screen, args=render_frame_args)
            pygame.display.flip()
            self._clock.tick(self._config.framerate)


if __name__ == "__main__":
    orchestrator = Orchestrator(OrchestratorConfig())
    orchestrator.run()
