"""uv run python -m replay.bench.test_bench"""

import base64
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from replay.bench.bench import ANGLE_SCALE, BenchConfig, export_bench, longest_balance
from rl.recorder import (
    METADATA_KEY,
    SCHEMA,
    Recorder,
    RecorderConfig,
    RecordingMetadata,
)


def _write(run_dir: Path, batches: list[list[tuple[int, int]]]) -> None:
    """batches of (episode, rows) runs, so an episode can straddle two batches"""
    metadata = RecordingMetadata(
        dt=0.1,
        arm_length=2,
        rail_length=10,
        max_steps=10,
        terminate_on_fall=True,
        state_columns=[
            "arm_angle",
            "angular_velocity",
            "block_position_along_rail",
            "block_velocity",
        ],
    )
    schema = SCHEMA.with_metadata({METADATA_KEY: json.dumps(asdict(metadata))})
    with pa.OSFile(str(run_dir / RecorderConfig().file_name), "wb") as sink:
        writer = pa.ipc.new_stream(sink, schema)
        step = {}
        for runs in batches:
            columns = {name: [] for name in schema.names}
            for episode, rows in runs:
                start = step.get(episode, 0)
                step[episode] = start + rows
                columns["episode_number"] += [episode] * rows
                columns["step"] += list(range(start, start + rows))
                # each episode its own angle, so a slice from the wrong rows shows
                columns["arm_angle"] += [np.pi + episode / 10] * rows
                columns["angular_velocity"] += [0.0] * rows
                columns["block_position_along_rail"] += [5.0] * rows
                columns["block_velocity"] += [0.0] * rows
                columns["command"] += [1.0] * rows
                columns["terminated"] += [episode != 2] * rows
            writer.write_batch(pa.RecordBatch.from_pydict(columns, schema=schema))
        writer.close()


def _chunk(folder: Path, number: int) -> bytes:
    script = (folder / f"chunk-{number:05d}.js").read_text()
    assert script.startswith(f"__benchChunk({number},")
    return base64.b64decode(script.split('"')[1])


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        run_dir = Path(folder)
        # episode 1 is split across the two batches: 4 + 3 rows
        _write(run_dir, [[(0, 5), (1, 4)], [(1, 3), (2, 11)]])
        config = BenchConfig(episodes_per_chunk=2)
        summary = export_bench(Recorder(RecorderConfig()), run_dir, config)
        chunks = run_dir / config.chunk_dir
        files = sorted(path.name for path in chunks.iterdir())
        first, second = _chunk(chunks, 0), _chunk(chunks, 1)

    assert summary["episodes"] == 3 and summary["numbers"] is None
    packed = np.frombuffer(base64.b64decode(summary["packed"]), np.uint16)
    assert (packed & 0x7FFF).tolist() == [4, 6, 10]
    assert (packed >> 15).tolist() == [1, 1, 0]

    # two episodes to a chunk: 5 + 7 rows, then 11. angles, positions, then pushes
    assert files == ["chunk-00000.js", "chunk-00001.js"]
    assert len(first) == 12 * 5 and len(second) == 11 * 5
    angle = np.frombuffer(first, "<i2", 12) / ANGLE_SCALE
    expected = np.repeat([np.pi, np.pi + 0.1], [5, 7])
    assert np.allclose(angle, (expected + np.pi) % (2 * np.pi) - np.pi, atol=1e-3)
    position = np.frombuffer(first, "<u2", 12, 24) / summary["chunks"]["positionScale"]
    assert np.allclose(position, 5.0, atol=1e-3)
    assert np.frombuffer(first, np.int8, 12, 48).tolist() == [1] * 12
    # every test episode is held upright: balance counts every state after the reset
    balance = np.frombuffer(base64.b64decode(summary["balance"]), np.uint16)
    assert balance.tolist() == [4, 6, 10]

    # down, up, up, down, up, up, up: the reset state never counts, the longest run is 3
    angles = np.array([np.pi, 0, np.pi, np.pi, 0, np.pi, np.pi, np.pi])
    assert longest_balance(angles) == 3
    assert longest_balance(np.zeros(5)) == 0
    print("bench export ok")


if __name__ == "__main__":
    main()
