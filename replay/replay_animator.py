import math
import subprocess
from pathlib import Path

import numpy as np
import pygame
from matplotlib.colors import LinearSegmentedColormap
from rich.progress import track

from engine.state import CartesianCoordinates, SimulationState
from replay.config import ReplayConfig
from rl.recorder import Recorder


class ReplayAnimator:
    """renders every episode of a run overlaid into an mp4. colour is episode number
    (cool early, warm late), pendulums are matte and translucent, only the force each
    one applies glows and leaves a trail. ends on a photo of every frame summed together"""

    def __init__(self, config: ReplayConfig, recorder: Recorder) -> None:
        self._config = config
        self._recorder = recorder

        # idempotent, and the only pygame subsystem the replay needs
        pygame.font.init()
        self._value_font = pygame.font.SysFont(config.font, config.value_font_size)
        self._label_font = pygame.font.SysFont(config.font, config.label_font_size)

    def _episode_colors(self, num_episodes: int) -> np.ndarray:
        """(num_episodes, 3) rgb, 0-255"""
        colormap = LinearSegmentedColormap.from_list(
            "episodes", [np.array(color) / 255 for color in self._config.episode_colors]
        )
        return colormap(np.linspace(0, 1, num_episodes))[:, :3] * 255

    def _draw_pendulum(
        self,
        surfaces: list[pygame.Surface],
        block: tuple[float, float],
        bob: tuple[float, float],
        color: np.ndarray,
        opacity: float,
    ) -> None:
        """blends one solid pendulum onto each surface. positions in pixels"""
        config = self._config
        padding = max(config.bob_radius, config.cart_size[0]) + 1
        left = min(block[0], bob[0]) - padding
        top = min(block[1], bob[1]) - padding
        layer = pygame.Surface(
            (abs(block[0] - bob[0]) + 2 * padding, abs(block[1] - bob[1]) + 2 * padding)
        )
        layer.set_colorkey((0, 0, 0))  # only the shapes are blended, not the box
        layer.set_alpha(round(255 * opacity))
        block = (block[0] - left, block[1] - top)
        bob = (bob[0] - left, bob[1] - top)

        core = tuple(int(v) for v in np.clip(color, 1, 255))
        pygame.draw.line(layer, core, block, bob, config.rod_thickness)
        pygame.draw.circle(layer, core, bob, config.bob_radius)
        cart = pygame.Rect((0, 0), config.cart_size)
        cart.center = (round(block[0]), round(block[1]))
        pygame.draw.rect(layer, core, cart, border_radius=3)

        for surface in surfaces:
            surface.blit(layer, (left, top))

    def _draw_flare(
        self,
        surface: pygame.Surface,
        block: tuple[float, float],
        command: float,
        color: np.ndarray,
    ) -> None:
        """adds an exhaust's light behind the cart, on the side away from the push and
        longer the harder it pushes. block in pixels, command in [-1, 1]"""
        config = self._config
        core = tuple(int(v) for v in np.clip(color, 0, 255))
        if not command or not any(core):
            return

        back = -math.copysign(1, command)
        edge = block[0] + back * config.cart_size[0] / 2
        length = abs(command) * config.flare_length
        half = config.cart_size[1] / 2
        padding = config.cart_size[0] + 2 * config.flare_length
        left, top = block[0] - padding, block[1] - padding
        # black, so adding it onto surface only adds light
        layer = pygame.Surface((2 * padding, 2 * padding))

        # a wide dim plume under a narrow bright one reads as glow
        for spread, reach, dim in ((1.6, 1.5, 3), (1.0, 1.0, 1)):
            tip = edge + back * length * reach
            pygame.draw.polygon(
                layer,
                tuple(v // dim for v in core),
                [
                    (edge - left, block[1] - half * spread - top),
                    (edge - left, block[1] + half * spread - top),
                    (tip - left, block[1] - top),
                ],
            )

        surface.blit(layer, (left, top), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_rig(self, surface: pygame.Surface, rail_length: float) -> None:
        """rail with a ruler (a tick every half metre, longer on whole metres) and end stops"""
        config = self._config
        scale = (config.width - 2 * config.rail_margin) / rail_length
        rail_y = round(config.height * config.rail_height)
        left, right = config.rail_margin, config.width - config.rail_margin

        pygame.draw.line(surface, config.ink_color, (left, rail_y), (right, rail_y), 2)
        for tick in range(math.floor(rail_length * 2) + 1):
            x = left + scale * tick / 2
            length = 14 if tick % 2 == 0 else 9
            pygame.draw.line(
                surface, config.ink_color, (x, rail_y + 5), (x, rail_y + length)
            )
        for x in (left, right):
            pygame.draw.rect(surface, config.ink_color, (x - 4, rail_y - 10, 8, 20), 1)

    def _draw_strip(self, surface: pygame.Surface, points: list[tuple[float, float]]):
        """the share of episodes still up, filled in up to the current frame"""
        config = self._config
        left, right = config.rail_margin, config.width - config.rail_margin
        bottom = config.height - 40

        pygame.draw.line(surface, config.ink_color, (left, bottom), (right, bottom))
        if len(points) > 1:
            pygame.draw.polygon(
                surface,
                config.strip_fill_color,
                [(left, bottom), *points, (points[-1][0], bottom)],
            )
            pygame.draw.lines(surface, config.text_color, False, points, 2)
        label = self._label_font.render("still alive", True, config.label_color)
        surface.blit(label, (left, bottom - 74))

    def _draw_readouts(
        self, surface: pygame.Surface, readouts: list[tuple[str, str]], caption: str
    ) -> None:
        """(label, value) columns over the rail's left end, caption over its right end"""
        config = self._config
        top = 40
        value_top = top + self._label_font.get_linesize() + 4

        for column, (label, value) in enumerate(readouts):
            x = config.rail_margin + column * config.readout_spacing
            surface.blit(
                self._label_font.render(label, True, config.label_color), (x, top)
            )
            surface.blit(
                self._value_font.render(value, True, config.text_color), (x, value_top)
            )

        rendered = self._label_font.render(caption, True, config.label_color)
        surface.blit(
            rendered,
            rendered.get_rect(topright=(config.width - config.rail_margin, top)),
        )

    def render(self, run_dir: Path) -> Path:
        config = self._config
        recording = self._recorder.read_recording(run_dir)
        metadata = recording.metadata

        steps = recording.steps.sort_values(["episode_number", "step"])
        # one contiguous array split into (length + 1, 4) views per episode, so frames
        # don't query the DataFrame and there's no per-episode copy of it
        splits = np.flatnonzero(np.diff(steps["episode_number"].to_numpy())) + 1
        episodes = np.split(steps[metadata.state_columns].to_numpy(), splits)
        # the last state has no action after it, recorded as NaN
        commands = np.split(
            np.nan_to_num(steps["command"].to_numpy(dtype=float)), splits
        )
        del recording, steps  # the arrays are all the frames need
        ends = np.array([len(states) - 1 for states in episodes])
        last_frame = int(ends.max())
        flare_colors = self._episode_colors(len(episodes))
        luminance = flare_colors @ np.array([0.299, 0.587, 0.114])
        pendulum_colors = luminance[:, None] + config.pendulum_saturation * (
            flare_colors - luminance[:, None]
        )

        fps = round(config.playback_speed / metadata.dt)
        fade_frames = max(1, round(config.fade_seconds / metadata.dt))
        decay = round(255 * math.exp(-metadata.dt / config.trail_seconds))

        # headless: frames are drawn onto off-screen surfaces, which need no pygame
        # subsystem (fonts are already up). pygame.init() would also open the audio
        # device, which then underruns while this loop keeps the cpu busy
        width, height = config.width, config.height
        scale = (width - 2 * config.rail_margin) / metadata.rail_length
        rail_y = round(height * config.rail_height)

        def to_pixels(x: float, y: float) -> tuple[float, float]:
            return config.rail_margin + scale * x, rail_y + scale * y

        rig = pygame.Surface((width, height))
        self._draw_rig(rig, metadata.rail_length)

        still_up = [int(np.sum(ends > frame)) for frame in range(last_frame + 1)]
        strip_top, strip_bottom = height - 90, height - 40
        strip_points = [
            (
                config.rail_margin
                + (width - 2 * config.rail_margin) * frame / last_frame,
                strip_bottom - (strip_bottom - strip_top) * up / len(episodes),
            )
            for frame, up in enumerate(still_up)
        ]

        frame_light = pygame.Surface((width, height))  # this frame's flares only
        exposure = pygame.Surface((width, height))  # flares plus their fading trails
        pendulums = pygame.Surface(
            (width, height)
        )  # this frame's pendulums, for the photo
        photo = np.zeros(
            (width, height, 3), np.float32
        )  # every frame's pendulums, summed
        surface = pygame.Surface((width, height))
        output_path = run_dir / config.file_name

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
            # past the last fall, so the final pendulums fade instead of vanishing
            for frame in track(
                range(last_frame + fade_frames + 1), description="rendering replay"
            ):
                exposure.fill((decay,) * 3, special_flags=pygame.BLEND_RGB_MULT)
                frame_light.fill((0, 0, 0))
                pendulums.fill((0, 0, 0))

                since_end = frame - ends
                weights = np.clip(1 - since_end / fade_frames, 0, 1) ** 2
                weights[since_end <= 0] = 1
                # survivors stand out more as the rest fall, so the last few don't vanish
                crowd = max(weights.sum(), 1) ** config.crowd_dimming
                opacity = float(np.clip(config.opacity / crowd, config.min_opacity, 1))
                flare_gain = config.flare_brightness / crowd

                visible = np.flatnonzero(weights)
                pixels = []
                for index in visible:
                    step = min(frame, ends[index])
                    state = SimulationState(
                        **dict(
                            zip(metadata.state_columns, episodes[index][step].tolist())
                        )
                    )
                    coordinates = CartesianCoordinates.from_state(
                        state, metadata.arm_length
                    )
                    block = to_pixels(*coordinates.block_coordinates)
                    pixels.append((block, to_pixels(*coordinates.bob_coordinates)))
                    self._draw_flare(
                        frame_light,
                        block,
                        commands[index][step],
                        flare_colors[index] * weights[index] * flare_gain,
                    )

                exposure.blit(frame_light, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

                surface.fill(config.background_color)
                surface.blit(exposure, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
                surface.blit(rig, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
                # pendulums over the glow, so the flares sit behind their carts
                for index, (block, bob) in zip(visible, pixels):
                    self._draw_pendulum(
                        [surface, pendulums],
                        block,
                        bob,
                        pendulum_colors[index],
                        opacity * weights[index],
                    )
                # pixels3d is a view, so no per-frame copy; it unlocks once the statement ends
                np.add(photo, pygame.surfarray.pixels3d(pendulums), out=photo)

                self._draw_strip(surface, strip_points[: min(frame, last_frame) + 1])
                self._draw_readouts(
                    surface,
                    [
                        ("time", f"{min(frame, last_frame) * metadata.dt:.2f} s"),
                        (
                            "still alive",
                            f"{still_up[min(frame, last_frame)]} / {len(episodes)}",
                        ),
                    ],
                    run_dir.name,
                )

                process.stdin.write(pygame.image.tobytes(surface, "RGB"))

            # end card: tone mapped so dense regions saturate softly and keep their colour
            brightest = photo.max(axis=2)
            reference = np.percentile(brightest[brightest > 0], 99)
            mapped = 255 * (1 - np.exp(-config.photo_exposure * photo / reference))

            end_card = pygame.Surface((width, height))
            end_card.fill(config.background_color)
            end_card.blit(
                pygame.surfarray.make_surface(mapped.astype(np.uint8)),
                (0, 0),
                special_flags=pygame.BLEND_RGB_ADD,
            )
            end_card.blit(rig, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            self._draw_strip(end_card, strip_points)
            self._draw_readouts(
                end_card,
                [
                    ("episodes", str(len(episodes))),
                    ("longest", f"{last_frame * metadata.dt:.2f} s"),
                ],
                run_dir.name,
            )

            crossfade_frames = round(config.crossfade_seconds * fps)
            last_surface = surface.copy()
            for index in range(1, crossfade_frames + 1):
                surface.blit(last_surface, (0, 0))
                end_card.set_alpha(round(255 * index / crossfade_frames))
                surface.blit(end_card, (0, 0))
                process.stdin.write(pygame.image.tobytes(surface, "RGB"))

            end_card.set_alpha(None)
            final_frame = pygame.image.tobytes(end_card, "RGB")
            for _ in range(round(config.end_hold_seconds * fps)):
                process.stdin.write(final_frame)
        finally:
            process.stdin.close()

        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with code {process.returncode}")

        return output_path
