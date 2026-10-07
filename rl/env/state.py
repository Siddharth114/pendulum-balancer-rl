from dataclasses import dataclass, fields

from engine.state import SimulationState


@dataclass(frozen=True)
class Action:
    command: (
        float  # in the range [-1, 1] i.e., left to right. a fraction of the max force
    )

    def __post_init__(self):
        if not (-1 <= self.command <= 1):
            raise ValueError("command must be between -1 and 1")


@dataclass(frozen=True)
class StepReward:
    upright_reward: float = 0
    rail_centre_reward: float = 0
    spin_reward: float = 0
    total_reward: float = 0


@dataclass(frozen=True)
class RLEnvState(SimulationState):
    reward: StepReward = StepReward()
    terminated: bool = False
    truncated: bool = False

    def to_simulation_state(self) -> SimulationState:
        """drops the reward and episode flags, keeping only the physics"""
        return SimulationState(
            **{
                field.name: getattr(self, field.name)
                for field in fields(SimulationState)
            }
        )
