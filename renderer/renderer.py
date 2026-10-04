import pygame

from renderer.config import RendererConfig, RenderFrameArgs


class Renderer:
    def __init__(self, config: RendererConfig, rail_length: float) -> None:
        """rail_length is in metres, taken from the simulation so the two can't disagree"""
        self._config = config
        # pixels per metre
        self._scale = (config.window_width - (2 * config.rail_margin)) / rail_length

    def _system_coordinates(self, x: float, y: float) -> tuple[float, float]:
        """converts system coordinates (metres) to window coordinates (pixels)"""
        return (
            self._config.rail_margin + (self._scale * x),
            self._config.window_height // 2 + (self._scale * y),
        )

    def _draw_rail(self, surface: pygame.Surface):
        rail_y = self._config.window_height // 2

        pygame.draw.line(
            surface=surface,
            color=self._config.rail_color,
            start_pos=(self._config.rail_margin, rail_y),
            end_pos=(self._config.window_width - self._config.rail_margin, rail_y),
            width=self._config.rail_thickness,
        )

    def _draw_pendulum(self, surface: pygame.Surface, args: RenderFrameArgs):
        block_x, block_y = args.block_position
        bob_x, bob_y = args.bob_position

        scaled_block_x, scaled_block_y = self._system_coordinates(block_x, block_y)
        scaled_bob_x, scaled_bob_y = self._system_coordinates(bob_x, bob_y)

        pygame.draw.line(
            surface=surface,
            color=self._config.rod_color,
            start_pos=(scaled_block_x, scaled_block_y),
            end_pos=(scaled_bob_x, scaled_bob_y),
            width=self._config.rod_thickness,
        )

        pygame.draw.circle(
            surface=surface,
            color=self._config.block_color,
            radius=self._config.block_radius,
            center=(scaled_block_x, scaled_block_y),
        )

        pygame.draw.circle(
            surface=surface,
            color=self._config.bob_color,
            radius=self._config.bob_radius,
            center=(scaled_bob_x, scaled_bob_y),
        )

    def render_frame(self, surface: pygame.Surface, args: RenderFrameArgs):
        surface.fill(self._config.window_background_color)

        self._draw_rail(surface)

        self._draw_pendulum(surface, args)
