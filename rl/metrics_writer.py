import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from rl.algorithms.episode import Episode, EpisodeMetrics


@dataclass(frozen=True)
class MetricsWriterConfig:
    file_name: str = "metrics.csv"


class MetricsWriter:
    def __init__(self, config: MetricsWriterConfig) -> None:
        self._config = config

    def write_metrics(self, results_dir: Path, episodes: list[Episode]) -> None:
        metrics_path = results_dir / self._config.file_name

        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["episode_number"]
                + [field.name for field in fields(EpisodeMetrics)],
            )

            writer.writeheader()
            writer.writerows(
                {"episode_number": episode.number, **asdict(episode.metrics)}
                for episode in episodes
            )
