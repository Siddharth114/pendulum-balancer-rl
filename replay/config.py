from dataclasses import dataclass

Color = tuple[int, int, int]


@dataclass(frozen=True)
class ReplayConfig:
    width: int = 1280  # pixels, even so yuv420p accepts it
    height: int = 720  # pixels, even so yuv420p accepts it
    # frame edge to rail end, in pixels. wider than the arm, so a bob past an end stays in frame
    rail_margin: int = 220
    rail_height: float = 0.62  # rail's y as a fraction of the frame height

    cart_size: tuple[int, int] = (28, 14)  # pixels
    bob_radius: int = 8  # pixels
    rod_thickness: int = 3  # pixels
    flare_length: int = 40  # pixels, exhaust behind the cart at full command

    # first one installed wins, falls back to pygame's default. monospaced so numbers don't jitter
    font: str = "sfnsmono,jetbrainsmono,menlo,dejavusansmono"
    value_font_size: int = 22
    label_font_size: int = 14
    readout_spacing: int = 200  # pixels between readout columns

    background_color: Color = (10, 10, 14)
    ink_color: Color = (90, 92, 104)  # rail, ruler, end stops, strip axis
    text_color: Color = (180, 182, 192)  # readout values
    label_color: Color = (120, 122, 134)  # readout labels, run name
    strip_fill_color: Color = (40, 40, 52)
    # first episode to last, interpolated: cool early on, warm once it has learned
    episode_colors: tuple[Color, ...] = ((61, 123, 255), (180, 77, 255), (255, 138, 42))

    pendulum_saturation: float = 0.55  # 1 is the full episode colour, 0 is grey
    opacity: float = 8.0  # pendulum opacity with one up, shared out as more are up
    min_opacity: float = 0.06  # so each of a thousand stacked pendulums still shows
    flare_brightness: float = 2.5  # flare light with one up, shared out as more are up
    # 1 holds the total constant as pendulums fall, 0 never brightens the survivors
    crowd_dimming: float = 0.8
    trail_seconds: float = 0.15  # flare glow decays to 1/e in this much simulated time
    fade_seconds: float = 0.6  # a fallen pendulum fades out over this, simulated

    photo_exposure: float = 1.6  # brightness of the end card that sums every frame
    crossfade_seconds: float = 0.6  # from the last frame to the end card
    end_hold_seconds: float = 3.0  # keep the end card on screen this long

    playback_speed: float = 1.0  # 2.0 plays twice as fast, still showing every step

    file_name: str = "replay.mp4"
