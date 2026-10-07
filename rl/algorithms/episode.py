from dataclasses import dataclass

from engine.state import SimulationState
from rl.env.state import Action


@dataclass(frozen=True)
class EpisodeMetrics:
    """per-episode summary numbers, what goes into the metrics and the graph"""

    length: int  # step units

    longest_balance: int  # step units

    total_reward: float
    max_reward: float

    avg_upright_reward: float
    avg_rail_centre_reward: float
    avg_spin_reward: float
    avg_total_reward: float

    avg_command: float
    avg_command_change: float


@dataclass(frozen=True)
class EpisodeRecording:
    """per-step data, what goes into the results dir for the replay and the model"""

    states: list[SimulationState]  # length + 1, starts with the reset state
    actions: list[Action]  # length
    terminated: bool


@dataclass(frozen=True)
class Episode:
    number: int  # episode index within the run

    metrics: EpisodeMetrics
    recording: EpisodeRecording
