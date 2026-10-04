from dataclasses import dataclass


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


@dataclass(frozen=True)
class RenderFrameArgs:
    """always relative to the pendulum system. so 0,0 is the top left corner of the rail, not of the window"""

    block_position: tuple[float, float]
    bob_position: tuple[float, float]
