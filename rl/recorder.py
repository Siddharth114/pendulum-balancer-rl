import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from rl.algorithms.episode import Episode
from rl.env.config import RLEnvConfig

STATE_COLUMNS = (
    "arm_angle",
    "angular_velocity",
    "block_position_along_rail",
    "block_velocity",
)


SCHEMA = pa.schema(
    [
        ("episode_number", pa.int64()),
        ("step", pa.int64()),
        *[(column, pa.float64()) for column in STATE_COLUMNS],
        ("command", pa.float64()),
        ("terminated", pa.bool_()),
    ]
)


METADATA_KEY = b"recording"


@dataclass(frozen=True)
class RecordingMetadata:
    """what a reader needs beyond the rows, to replay or learn from them"""

    dt: float  # seconds between consecutive steps
    arm_length: float  # metres
    rail_length: float  # metres
    max_steps: int  # step limit before truncation
    terminate_on_fall: bool
    state_columns: list[str]  # STATE_COLUMNS at write time

    @classmethod
    def from_env_config(cls, env_config: RLEnvConfig) -> RecordingMetadata:
        return cls(
            dt=env_config.step_dt,
            arm_length=env_config.simulation.params.arm_length,
            rail_length=env_config.simulation.params.rail_length,
            max_steps=env_config.max_steps,
            terminate_on_fall=env_config.terminate_on_fall,
            state_columns=list(STATE_COLUMNS),
        )

    @classmethod
    def from_schema(cls, schema: pa.Schema) -> RecordingMetadata:
        return cls(**json.loads((schema.metadata or {})[METADATA_KEY]))


@dataclass(frozen=True)
class Recording:
    """a recording read back from disk"""

    steps: pd.DataFrame
    metadata: RecordingMetadata


@dataclass(frozen=True)
class RecorderConfig:
    file_name: str = "recording.arrows"
    rows_per_write: int = 100_000


class Recorder:
    def __init__(self, config: RecorderConfig):
        self._config = config

    @contextmanager
    def open(
        self, results_dir: Path, env_config: RLEnvConfig
    ) -> Generator[Callable[[Episode], None]]:
        recordings_path = self.path_of(results_dir)

        metadata = RecordingMetadata.from_env_config(env_config)
        schema = SCHEMA.with_metadata({METADATA_KEY: json.dumps(asdict(metadata))})

        sink = pa.OSFile(str(recordings_path), "wb")
        writer = pa.ipc.new_stream(
            sink, schema, options=pa.ipc.IpcWriteOptions(compression="zstd")
        )
        columns: dict[str, list] = {name: [] for name in schema.names}

        def flush() -> None:
            if columns["step"]:
                writer.write_batch(pa.RecordBatch.from_pydict(columns, schema=schema))
                for values in columns.values():
                    values.clear()

        def write_episode(episode: Episode) -> None:
            recording = episode.recording
            for step, state in enumerate(recording.states):
                columns["episode_number"].append(episode.number)
                columns["step"].append(step)
                for column in STATE_COLUMNS:
                    columns[column].append(getattr(state, column))
                columns["command"].append(
                    recording.actions[step].command
                    if step < len(recording.actions)
                    else None
                )
                columns["terminated"].append(recording.terminated)

            if len(columns["step"]) >= self._config.rows_per_write:
                flush()

        try:
            yield write_episode
        finally:
            flush()
            writer.close()
            sink.close()

    def path_of(self, results_dir: Path) -> Path:
        return results_dir / self._config.file_name

    @contextmanager
    def open_reader(
        self, results_dir: Path
    ) -> Generator[tuple[RecordingMetadata, pa.ipc.RecordBatchStreamReader]]:
        """iterate the reader for one batch at a time, for runs too big to load whole.
        a plain file rather than a memory map, so pages already read don't stay resident"""
        with pa.OSFile(str(self.path_of(results_dir)), "r") as source:
            reader = pa.ipc.open_stream(source)
            yield RecordingMetadata.from_schema(reader.schema), reader

    def read_recording(self, results_dir: Path) -> Recording:
        with self.open_reader(results_dir) as (metadata, reader):
            steps = reader.read_all().to_pandas()

        return Recording(steps=steps, metadata=metadata)
