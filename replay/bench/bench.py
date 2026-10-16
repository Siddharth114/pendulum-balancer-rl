import base64
import json
import subprocess
import tempfile
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl.recorder import Recorder

BUILD_SCRIPT = Path(__file__).with_name("build.mjs")
ANGLE_SCALE = 32767 / np.pi  # wrapped angle to int16; the page unwraps it again


@dataclass(frozen=True)
class BenchConfig:
    file_name: str = "bench.html"
    # episodes as small scripts next to the page, which loads only the one it needs, so
    # the file opens from disk without a server and nothing holds every episode
    chunk_dir: str = "bench"
    episodes_per_chunk: int = 200
    light: bool = False  # the light theme instead of the dark one
    open_browser: bool = True


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


class _Chunks:
    """packs finished episodes and writes every episodes_per_chunk of them as one script"""

    def __init__(self, folder: Path, size: int, rail_length: float) -> None:
        self._folder = folder
        self._size = size
        self._position_scale = 65535 / rail_length
        self._episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.written = 0
        folder.mkdir(exist_ok=True)
        for stale in folder.glob("chunk-*.js"):
            stale.unlink()

    @property
    def position_scale(self) -> float:
        return self._position_scale

    def add(self, angle: np.ndarray, position: np.ndarray, command: np.ndarray) -> None:
        self._episodes.append((angle, position, command))
        if len(self._episodes) == self._size:
            self.flush()

    def flush(self) -> None:
        if not self._episodes:
            return
        angle, position, command = (
            np.concatenate(part) for part in zip(*self._episodes)
        )
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        payload = b"".join(
            [
                np.round(wrapped * ANGLE_SCALE).astype("<i2").tobytes(),
                np.round(np.clip(position * self._position_scale, 0, 65535))
                .astype("<u2")
                .tobytes(),
                np.round(np.nan_to_num(command)).astype(np.int8).tobytes(),
            ]
        )
        script = (
            f'__benchChunk({self.written},"{base64.b64encode(payload).decode()}");\n'
        )
        (self._folder / f"chunk-{self.written:05d}.js").write_text(script)
        self.written += 1
        self._episodes = []


def longest_balance(angle: np.ndarray) -> int:
    """the longest run of steps with the bob above the rail, counted after each step as
    training counts it (BaseAlgorithm's longest_balance), so the reset state is skipped"""
    above = np.concatenate([[False], np.cos(angle[1:]) < 0, [False]])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max(initial=0))


def export_bench(recorder: Recorder, run_dir: Path, config: BenchConfig) -> dict:
    """streams the recording once, a batch at a time: every episode goes into a chunk
    script, and only its length, balance and fall flag are kept for the page's timeline"""
    numbers: list[int] = []
    steps: list[int] = []
    balance: list[int] = []
    fell: list[bool] = []
    open_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    with recorder.open_reader(run_dir) as (metadata, reader):
        chunks = _Chunks(
            run_dir / config.chunk_dir,
            config.episodes_per_chunk,
            metadata.rail_length,
        )

        def close() -> None:
            angle, position, command = (
                np.concatenate(part) for part in zip(*open_parts)
            )
            steps.append(len(angle) - 1)  # the last state has no action after it
            balance.append(longest_balance(angle))
            chunks.add(angle, position, command)
            open_parts.clear()

        for batch in reader:
            episode = batch.column("episode_number").to_numpy()
            angle = batch.column("arm_angle").to_numpy()
            position = batch.column("block_position_along_rail").to_numpy()
            command = batch.column("command").to_numpy(zero_copy_only=False)
            terminated = batch.column("terminated").to_numpy(zero_copy_only=False)
            # rows of one episode are contiguous, and one episode can run across batches
            starts = np.concatenate([[0], np.flatnonzero(np.diff(episode)) + 1])
            ends = np.append(starts[1:], len(episode))
            for start, end in zip(starts, ends):
                number = int(episode[start])
                if not numbers or number != numbers[-1]:
                    if open_parts:
                        close()
                    numbers.append(number)
                    fell.append(bool(terminated[start]))
                open_parts.append(
                    (angle[start:end], position[start:end], command[start:end])
                )
        if open_parts:
            close()
        chunks.flush()

    numbers_array = np.array(numbers, dtype=np.int64)
    packed = np.minimum(np.array(steps), 0x7FFF).astype(np.uint16)
    packed |= np.array(fell, dtype=np.uint16) << 15
    contiguous = bool(np.all(np.diff(numbers_array) == 1))
    return {
        "run": run_dir.name,
        "meta": asdict(metadata),
        "episodes": len(numbers),
        "first": numbers[0],
        "last": numbers[-1],
        "packed": _b64(packed),  # two bytes per episode: its length, and a fall bit
        # what the drum and timeline chart: with terminate_on_fall off every episode
        # runs the full length, so length alone shows no learning
        "balance": _b64(np.minimum(np.array(balance), 0xFFFF).astype(np.uint16)),
        "numbers": None if contiguous else _b64(numbers_array.astype(np.int32)),
        "chunks": {
            "dir": config.chunk_dir,
            "size": config.episodes_per_chunk,
            "angleScale": ANGLE_SCALE,
            "positionScale": chunks.position_scale,
        },
    }


class BenchWriter:
    """writes a run as an interactive pendulum bench that opens from disk: the page holds
    every episode's length for the timeline and loads an episode's chunk when it is picked"""

    def __init__(self, config: BenchConfig, recorder: Recorder) -> None:
        self._config = config
        self._recorder = recorder

    def write(self, run_dir: Path) -> Path:
        summary = export_bench(self._recorder, run_dir, self._config)
        page = run_dir / self._config.file_name
        with tempfile.NamedTemporaryFile("w", suffix=".json") as data:
            json.dump(summary, data, separators=(",", ":"))
            data.flush()
            # the figure is drawn by node, with the isometric kit vendored next to it
            subprocess.run(
                ["node", str(BUILD_SCRIPT), data.name, str(page)]
                + (["--light"] if self._config.light else []),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        if self._config.open_browser:
            webbrowser.open(page.resolve().as_uri())
        return page
