from dataclasses import dataclass

from engine.state import SimulationState
from scoring.config import ScoringConfig


@dataclass(frozen=True)
class Score:
    current_score: float
    is_bob_above_rail: bool
    top_n: tuple[float, ...]


class ScoreKeeper:
    def __init__(self, config: ScoringConfig) -> None:
        self._config = config
        self._current_score = 0
        self._top_n = []

    def update_score(self, simulation_state: SimulationState, dt: float) -> Score:
        if simulation_state.is_bob_above_rail:
            self._current_score += dt

            return Score(
                current_score=self._current_score,
                top_n=tuple(self._top_n),
                is_bob_above_rail=True,
            )

        else:
            if self._current_score > 0:
                self._top_n.append(self._current_score)
                self._top_n.sort(reverse=True)
                self._top_n = self._top_n[: self._config.top_n]
            self._current_score = 0
            return Score(
                current_score=0, top_n=tuple(self._top_n), is_bob_above_rail=False
            )
