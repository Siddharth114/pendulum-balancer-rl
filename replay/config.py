from dataclasses import dataclass, replace

from renderer.config import RendererConfig

REPLAY_RENDERER_CONFIG = replace(
    RendererConfig(),
    block_color=(128, 118, 104),
    bob_color=(92, 80, 76),
)


@dataclass(frozen=True)
class ReplayConfig:
    renderer: RendererConfig = REPLAY_RENDERER_CONFIG

    running_alpha: int | None = None  # 0-255, None picks one from the episode count
    ended_alpha_fraction: float = 0.5  # finished episodes drawn at this fraction of it

    playback_speed: float = 1.0  # 2.0 plays twice as fast, still showing every step
    end_hold_seconds: float = 1.0  # keep the final frame on screen this long

    file_name: str = "replay.mp4"
