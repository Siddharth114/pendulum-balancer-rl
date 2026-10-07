import argparse

from engine.config import SIMULATION_MODES
from orchestrator.config import OrchestratorConfig
from rl.algorithms.random import Random
from rl.env.config import RL_ENV_MODES
from rl.grapher import Grapher, GrapherConfig
from rl.metrics_writer import MetricsWriter, MetricsWriterConfig
from rl.recorder import Recorder, RecorderConfig
from rl.trainer import Trainer, TrainerConfig

ALGORITHMS = {
    "random": Random,
}

# uv run main.py play easy
# uv run main.py train random --mode normal --episodes 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser("play", help="play the game yourself")
    play_parser.add_argument(
        "mode", nargs="?", default="normal", choices=SIMULATION_MODES.keys()
    )

    train_parser = subparsers.add_parser("train", help="train an agent headless")
    train_parser.add_argument("algorithm", choices=ALGORITHMS.keys())
    train_parser.add_argument("--mode", default="easy", choices=RL_ENV_MODES.keys())
    train_parser.add_argument("--episodes", type=int, default=100)

    args = parser.parse_args()

    if args.command == "play":
        from orchestrator.orchestrator import (
            Orchestrator,  # importing here to not display pygame banner in console
        )

        Orchestrator(OrchestratorConfig(simulation=SIMULATION_MODES[args.mode])).run()
    elif args.command == "train":
        algorithm = ALGORITHMS[args.algorithm](RL_ENV_MODES[args.mode])
        grapher = Grapher(GrapherConfig())
        writer = MetricsWriter(MetricsWriterConfig())
        recorder = Recorder(RecorderConfig())

        trainer = Trainer(
            config=TrainerConfig(),
            algorithm=algorithm,
            training_mode=args.mode,
            metrics_writer=writer,
            recorder=recorder,
            grapher=grapher,
        )

        path = trainer.run(args.episodes)
        print(f"run saved to {path}")
