from rich.progress import track

from engine.state import SimulationState
from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.env.rl_env import RLEnv
from rl.env.state import Action


class Random(BaseAlgorithm):
    cli_name = "random"

    def __init__(self, env: RLEnv) -> None:
        super().__init__(env)

    def name(self) -> str:
        return "Random Baseline"

    def _update(
        self,
        current_state: SimulationState,
        action_taken: Action,
        new_state: SimulationState,
        reward: float,
        terminated: bool,
    ) -> None:
        pass

    def action(self, state: SimulationState) -> Action:
        return self._env.sample_action_space()

    def train(self, num_episodes: int) -> None:

        for ind in track(
            range(num_episodes), description="training", auto_refresh=False
        ):
            self._run_episode(ind + 1)
