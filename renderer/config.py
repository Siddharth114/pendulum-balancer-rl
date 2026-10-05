from dataclasses import dataclass

from scoring.scorekeeper import Score


@dataclass(frozen=True)
class RendererConfig:
    window_width: int = 800  # pixels
    window_height: int = 600  # pixels

    rail_margin: int = 150  # distance from the window edge to the rail, in pixels
    rail_thickness: int = 10  # pixels

    block_radius: int = 15  # pixels
    bob_radius: int = 10  # pixels
    rod_thickness: int = 5  # pixels

    window_background_color: tuple[int, int, int] = (222, 222, 209)
    rail_color: tuple[int, int, int] = (197, 199, 188)
    block_color: tuple[int, int, int] = (182, 174, 159)
    rod_color: tuple[int, int, int] = (82, 70, 70)
    bob_color: tuple[int, int, int] = (182, 174, 159)

    hud_margin: int = 50
    hud_font_size: int = 24
    hud_default_color: tuple[int, int, int] = (76, 69, 65)
    score_up_color: tuple[int, int, int] = (132, 177, 121)
    score_down_color: tuple[int, int, int] = (189, 68, 68)


@dataclass(frozen=True)
class RenderFrameArgs:
    """always relative to the pendulum system. so 0,0 is the top left corner of the rail, not of the window"""

    block_position: tuple[float, float]
    bob_position: tuple[float, float]

    score: Score
