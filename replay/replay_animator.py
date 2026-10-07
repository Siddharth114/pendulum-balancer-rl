import os
import subprocess
from pathlib import Path

import numpy as np
import pygame

from engine.state import CartesianCoordinates, SimulationState
from renderer.renderer import Renderer
from replay.config import ReplayConfig
from rl.recorder import Recorder


class ReplayAnimator:
    """renders every episode of a run, overlaid and translucent, into an mp4"""

    def __init__(self, config: ReplayConfig, recorder: Recorder) -> None:
        self._config = config
        self._recorder = recorder

    def _running_alpha(self, num_episodes: int) -> int:
        if self._config.running_alpha is not None:
            return self._config.running_alpha

        # few episodes look nearly solid, many still build up density instead of a blob
        return int(np.clip(round(255 * 10 / num_episodes), 40, 230))

    def render(self, run_dir: Path) -> Path:
        recording = self._recorder.read_recording(run_dir)
        metadata = recording.metadata

        # one (length + 1, 4) array per episode, so frames don't query the DataFrame
        episodes = [
            group.sort_values("step")[metadata.state_columns].to_numpy()
            for _, group in recording.steps.groupby("episode_number")
        ]
        last_frame = max(len(states) for states in episodes) - 1

        running_alpha = self._running_alpha(len(episodes))
        ended_alpha = round(running_alpha * self._config.ended_alpha_fraction)

        # headless: frames are drawn onto an off-screen surface, no window opens.
        # set here rather than at import, so importing this never affects the game window
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        renderer_config = self._config.renderer
        width, height = renderer_config.window_width, renderer_config.window_height
        surface = pygame.Surface((width, height))
        renderer = Renderer(renderer_config, rail_length=metadata.rail_length)

        fps = round(self._config.playback_speed / metadata.dt)
        output_path = run_dir / self._config.file_name

        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                # input: raw rgb frames on stdin
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                # output: yuv420p so it plays everywhere, needs even width and height
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
        )
        assert process.stdin is not None

        try:
            for frame in range(last_frame + 1):
                renderer.draw_background(surface)

                running = [frame < len(states) - 1 for states in episodes]

                # finished episodes first, so the running ones are drawn on top of them
                for index in sorted(range(len(episodes)), key=lambda i: running[i]):
                    states = episodes[index]
                    row = states[min(frame, len(states) - 1)]
                    state = SimulationState(
                        **dict(zip(metadata.state_columns, row.tolist()))
                    )
                    coordinates = CartesianCoordinates.from_state(
                        state, metadata.arm_length
                    )
                    renderer.draw_pendulum(
                        surface,
                        coordinates.block_coordinates,
                        coordinates.bob_coordinates,
                        alpha=running_alpha if running[index] else ended_alpha,
                    )

                status = (
                    f"t = {frame * metadata.dt:.2f} s · "
                    f"{sum(running)} / {len(episodes)} still running"
                )
                renderer.draw_text_lines(
                    surface,
                    [
                        (run_dir.name, renderer_config.hud_default_color),
                        (status, renderer_config.hud_default_color),
                    ],
                )

                process.stdin.write(pygame.image.tobytes(surface, "RGB"))

            final_frame = pygame.image.tobytes(surface, "RGB")
            for _ in range(round(self._config.end_hold_seconds * fps)):
                process.stdin.write(final_frame)
        finally:
            process.stdin.close()

        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with code {process.returncode}")

        return output_path
