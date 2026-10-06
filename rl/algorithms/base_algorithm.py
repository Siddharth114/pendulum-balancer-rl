from abc import ABC, abstractmethod
from dataclasses import dataclass

from rl.env import Action, RLEnv, RLEnvConfig, RLEnvState


@dataclass(frozen=True)
class TrainingResults:
    episode_rewards: list[float]
    episode_lengths: list[int]


class BaseAlgorithm(ABC):
    def __init__(self, config: RLEnvConfig) -> None:
        self._env = RLEnv(config)
        super().__init__()

    @abstractmethod
    def action(self, state: RLEnvState) -> Action:
        raise NotImplementedError

    @abstractmethod
    def train(self, num_episodes: int) -> TrainingResults:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
