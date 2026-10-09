import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
from pathlib import Path

from engine.config import SIMULATION_MODES
from engine.simulation_engine import SimulationEngine
from orchestrator.config import OrchestratorConfig
from orchestrator.orchestrator import Orchestrator
from renderer.renderer import Renderer
from replay.config import ReplayConfig
from replay.replay_animator import ReplayAnimator
from rl.algorithms.base_algorithm import BaseAlgorithm
from rl.env.config import RL_ENV_MODES
from rl.env.rl_env import RLEnv
from rl.grapher import Grapher, GrapherConfig
from rl.metrics_writer import MetricsWriter, MetricsWriterConfig
from rl.recorder import Recorder, RecorderConfig
from rl.trainer import Trainer, TrainerConfig
from scoring.scorekeeper import ScoreKeeper

# uv run pendulum play easy
# uv run pendulum train random --mode normal --episodes 200
# uv run pendulum replay results/<run folder>


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser("play", help="play the game yourself")
    play_parser.add_argument(
        "mode", nargs="?", default="normal", choices=SIMULATION_MODES.keys()
    )

    train_parser = subparsers.add_parser("train", help="train an agent headless")
    train_parser.add_argument("algorithm", choices=BaseAlgorithm.registry.keys())
    train_parser.add_argument("--mode", default="easy", choices=RL_ENV_MODES.keys())
    train_parser.add_argument("--episodes", type=int, default=100)

    replay_parser = subparsers.add_parser(
        "replay", help="render a training run's episodes into an mp4"
    )
    replay_parser.add_argument("run_dir", type=Path)
    replay_parser.add_argument("--playback-speed", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "play":
        orchestrator_config = OrchestratorConfig(simulation=SIMULATION_MODES[args.mode])

        Orchestrator(
            config=orchestrator_config,
            simulation_engine=SimulationEngine(orchestrator_config.simulation),
            renderer=Renderer(
                orchestrator_config.renderer,
                rail_length=orchestrator_config.simulation.params.rail_length,
            ),
            scorekeeper=ScoreKeeper(orchestrator_config.scoring),
        ).run()
    elif args.command == "train":
        env_config = RL_ENV_MODES[args.mode]
        env = RLEnv(env_config, SimulationEngine(env_config.simulation))
        recorder = Recorder(RecorderConfig())

        trainer = Trainer(
            config=TrainerConfig(),
            algorithm=BaseAlgorithm.registry[args.algorithm](env),
            training_mode=args.mode,
            metrics_writer=MetricsWriter(MetricsWriterConfig()),
            recorder=recorder,
            grapher=Grapher(GrapherConfig()),
            replay_animator=ReplayAnimator(ReplayConfig(), recorder),
        )

        path = trainer.run(args.episodes)
        print(f"run saved to {path}")
    elif args.command == "replay":
        replay_animator = ReplayAnimator(
            ReplayConfig(playback_speed=args.playback_speed),
            Recorder(RecorderConfig()),
        )

        path = replay_animator.render(args.run_dir)
        print(f"replay saved to {path}")


if __name__ == "__main__":
    main()
