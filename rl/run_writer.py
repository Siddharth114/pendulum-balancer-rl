import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from rl.algorithms.base_algorithm import EpisodeResult


@dataclass(frozen=True)
class RunWriterConfig:
    results_dir: str = "results"  # relative to where you run from


class RunWriter:
    def __init__(self, config: RunWriterConfig) -> None:
        self._config = config

    def _slugify(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def create_run_dir(self, name: str, mode: str) -> Path:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(
            f"{self._config.results_dir}/{self._slugify(name)}_{mode}_{timestamp}"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        return results_dir

    def write_results(self, run_dir: Path, results: list[EpisodeResult]) -> Path:
        """one row per episode, columns in EpisodeResult field order"""
        path = run_dir / "episodes.csv"

        with path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=[field.name for field in fields(EpisodeResult)]
            )
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)

        return path
