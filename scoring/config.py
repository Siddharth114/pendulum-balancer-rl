from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    top_n: int = 5
