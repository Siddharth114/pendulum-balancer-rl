from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class RendererConfig:
    window_width: int  # pixels
    window_height: int  # pixels

    rail_margin: int  # distance from the window edge to the rail, in pixels
    rail_thickness: int  # pixels

    # in metres. this is used to calculate the scale for rendering
    #  scale = window_width- (2*rail_margin) / rail_length  in pixels per metre
    rail_length: int

    block_radius: int  # pixels
    bob_radius: int  # pixels
    rod_thickness: int  # pixels

    window_background_color: tuple[int, int, int] = (222, 222, 209)
    rail_color: tuple[int, int, int] = (197, 199, 188)
    block_color: tuple[int, int, int] = (182, 174, 159)
    rod_color: tuple[int, int, int] = (82, 70, 70)
    bob_color: tuple[int, int, int] = (182, 174, 159)

    @property
    def scale(self) -> float:
        """in pixels per metre"""
        return (self.window_width - (2 * self.rail_margin)) / self.rail_length


@dataclass(frozen=True)
class RenderFrameArgs:
    """always relative to the pendulum system. so 0,0 is the top left corner of the rail, not of the window"""

    block_position: tuple[int, int]
    bob_position: tuple[int, int]


class Renderer:
    def __init__(self, config: RendererConfig) -> None:
        self._config = config

    def _system_coordinates(self, x: int, y: int) -> tuple[float, float]:
        """converts system coordinates (metres) to window coordinates (pixels)"""
        return (
            self._config.rail_margin + (self._config.scale * x),
            self._config.window_height // 2 + (self._config.scale * y),
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
