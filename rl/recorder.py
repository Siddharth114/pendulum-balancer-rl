from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from rl.algorithms.episode import Episode


@dataclass(frozen=True)
class RecorderConfig:
    file_name: str = "recording.parquet"


class Recorder:
    def __init__(self, config: RecorderConfig):
        self._config = config

    def write_recordings(self, results_dir: Path, episodes: list[Episode]) -> None:
        recordings_path = results_dir / self._config.file_name

        df = pd.DataFrame([asdict(episode.recording) for episode in episodes])
        df.to_parquet(recordings_path, index=False)
