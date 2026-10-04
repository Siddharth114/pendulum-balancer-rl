from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.state import SimulationState


@dataclass(frozen=True)
class Move:
    command: float

    def __post_init__(self):
        if not (-1 <= self.command <= 1):
            raise ValueError("command must be between -1 and 1")


class BasePlayer(ABC):
    @abstractmethod
    def move(self, state: SimulationState) -> Move:
        raise NotImplementedError
