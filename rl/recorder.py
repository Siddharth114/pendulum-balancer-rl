import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rl.algorithms.episode import Episode
from rl.env.config import RLEnvConfig

STATE_COLUMNS = (
    "arm_angle",
    "angular_velocity",
    "block_position_along_rail",
    "block_velocity",
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


@dataclass(frozen=True)
class Recording:
    """a recording read back from disk"""

    steps: pd.DataFrame
    metadata: RecordingMetadata


@dataclass(frozen=True)
class RecorderConfig:
    file_name: str = "recording.parquet"


class Recorder:
    def __init__(self, config: RecorderConfig):
        self._config = config

    def write_recordings(
        self, results_dir: Path, episodes: list[Episode], env_config: RLEnvConfig
    ) -> None:
        recordings_path = results_dir / self._config.file_name

        columns: dict[str, list] = {
            "episode_number": [],
            "step": [],
            **{column: [] for column in STATE_COLUMNS},
            "command": [],
            "terminated": [],
        }

        for episode in episodes:
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

        table = pa.Table.from_pandas(pd.DataFrame(columns), preserve_index=False)

        metadata = RecordingMetadata.from_env_config(env_config)
        table = table.replace_schema_metadata(
            {
                **(table.schema.metadata or {}),
                METADATA_KEY: json.dumps(asdict(metadata)).encode(),
            }
        )

        pq.write_table(table, recordings_path)

    def read_recording(self, results_dir: Path) -> Recording:
        table = pq.read_table(results_dir / self._config.file_name)

        metadata = RecordingMetadata(
            **json.loads((table.schema.metadata or {})[METADATA_KEY])
        )

        return Recording(steps=table.to_pandas(), metadata=metadata)
