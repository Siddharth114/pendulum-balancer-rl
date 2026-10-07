import pygame

from renderer.config import RendererConfig, RenderFrameArgs


class Renderer:
    def __init__(self, config: RendererConfig, rail_length: float) -> None:
        """rail_length is in metres, taken from the simulation so the two can't disagree"""
        self._config = config
        # pixels per metre
        self._scale = (config.window_width - (2 * config.rail_margin)) / rail_length

        # idempotent, lets the renderer be built before the caller runs pygame.init()
        pygame.font.init()
        self._hud_font = pygame.font.Font(None, self._config.hud_font_size)

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

    def _draw_pendulum_shapes(
        self,
        surface: pygame.Surface,
        block: tuple[float, float],
        bob: tuple[float, float],
    ):
        """rod, block and bob at the given pixel positions on surface"""
        pygame.draw.line(
            surface=surface,
            color=self._config.rod_color,
            start_pos=block,
            end_pos=bob,
            width=self._config.rod_thickness,
        )

        pygame.draw.circle(
            surface=surface,
            color=self._config.block_color,
            radius=self._config.block_radius,
            center=block,
        )

        pygame.draw.circle(
            surface=surface,
            color=self._config.bob_color,
            radius=self._config.bob_radius,
            center=bob,
        )

    def draw_background(self, surface: pygame.Surface):
        """clears the frame and draws the rail"""
        surface.fill(self._config.window_background_color)
        self._draw_rail(surface)

    def draw_pendulum(
        self,
        surface: pygame.Surface,
        block_position: tuple[float, float],
        bob_position: tuple[float, float],
        alpha: int = 255,
    ):
        """positions in system coordinates (metres). alpha is 0 (invisible) to 255 (opaque)"""
        block = self._system_coordinates(*block_position)
        bob = self._system_coordinates(*bob_position)

        if alpha == 255:
            self._draw_pendulum_shapes(surface, block, bob)
            return

        padding = max(self._config.block_radius, self._config.bob_radius) + 1
        left = min(block[0], bob[0]) - padding
        top = min(block[1], bob[1]) - padding
        width = abs(block[0] - bob[0]) + 2 * padding
        height = abs(block[1] - bob[1]) + 2 * padding

        layer = pygame.Surface((width, height), pygame.SRCALPHA)
        self._draw_pendulum_shapes(
            layer,
            (block[0] - left, block[1] - top),
            (bob[0] - left, bob[1] - top),
        )
        layer.set_alpha(alpha)
        surface.blit(layer, (left, top))

    def draw_text_lines(
        self, surface: pygame.Surface, lines: list[tuple[str, tuple[int, int, int]]]
    ):
        """(text, colour) pairs stacked down from the top right corner"""
        x = self._config.window_width - self._config.hud_margin
        y = self._config.hud_margin

        for text, color in lines:
            text_surface = self._hud_font.render(text, True, color)
            surface.blit(text_surface, text_surface.get_rect(topright=(x, y)))
            y += self._hud_font.get_linesize()

    def _score_lines(
        self, args: RenderFrameArgs
    ) -> list[tuple[str, tuple[int, int, int]]]:
        return [
            (
                f"Current Score: {args.score.current_score:.2f}s",
                self._config.score_up_color
                if args.score.is_bob_above_rail
                else self._config.score_down_color,
            ),
            ("Best Scores:", self._config.hud_default_color),
            *[
                (f"{ind}. {score:.2f}s", self._config.hud_default_color)
                for ind, score in enumerate(args.score.top_n, start=1)
            ],
        ]

    def render_frame(self, surface: pygame.Surface, args: RenderFrameArgs):
        """one full game frame: background, the pendulum, and the score hud"""
        self.draw_background(surface)
        self.draw_pendulum(surface, args.block_position, args.bob_position)
        self.draw_text_lines(surface, self._score_lines(args))
