import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import sys
from pathlib import Path

import questionary

from engine.config import SIMULATION_MODES
from engine.simulation_engine import SimulationEngine
from orchestrator.config import OrchestratorConfig
from orchestrator.orchestrator import Orchestrator
from renderer.renderer import Renderer
from replay.bench.bench import BenchConfig, BenchWriter
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
# uv run pendulum bench results/<run folder> [--light]   (interactive bench.html, opens from disk)
# uv run pendulum                  (no args: pick everything from menus)


def _ask_args() -> list[str]:
    command = questionary.select(
        "what do you want to do?", choices=["train", "replay", "bench", "play"]
    ).unsafe_ask()

    if command == "play":
        mode = questionary.select(
            "mode", choices=list(SIMULATION_MODES), default="normal"
        ).unsafe_ask()
        return [command, mode]

    if command == "train":
        algorithm = questionary.select(
            "algorithm", choices=list(BaseAlgorithm.registry)
        ).unsafe_ask()
        mode = questionary.select(
            "mode", choices=list(RL_ENV_MODES), default="easy"
        ).unsafe_ask()
        episodes = questionary.text(
            "episodes",
            default="100",
            validate=lambda text: text.isdigit() and int(text) > 0,
        ).unsafe_ask()
        return [command, algorithm, "--mode", mode, "--episodes", episodes]

    results_dir = TrainerConfig().results_dir
    runs = sorted(
        (
            run_dir
            for run_dir in (results_dir.iterdir() if results_dir.exists() else [])
            if (run_dir / RecorderConfig().file_name).exists()
        ),
        key=lambda run_dir: run_dir.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        sys.exit("no recorded runs to replay yet, train one first")

    run_dir = questionary.select(
        "run", choices=[str(run_dir) for run_dir in runs]
    ).unsafe_ask()
    return [command, run_dir]


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

    bench_parser = subparsers.add_parser(
        "bench", help="write a training run as an interactive pendulum bench page"
    )
    bench_parser.add_argument("run_dir", type=Path)
    bench_parser.add_argument("--light", action="store_true")
    bench_parser.add_argument("--no-browser", action="store_true")

    try:
        argv = None if len(sys.argv) > 1 else _ask_args()
    except KeyboardInterrupt:
        return

    args = parser.parse_args(argv)

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
            bench_writer=BenchWriter(BenchConfig(open_browser=False), recorder),
        )

        path = trainer.run(args.episodes)
        print(
            f"run saved to {path}, open {path / BenchConfig().file_name} to replay it"
        )
    elif args.command == "replay":
        replay_animator = ReplayAnimator(
            ReplayConfig(playback_speed=args.playback_speed),
            Recorder(RecorderConfig()),
        )

        path = replay_animator.render(args.run_dir)
        print(f"replay saved to {path}")
    elif args.command == "bench":
        path = BenchWriter(
            BenchConfig(light=args.light, open_browser=not args.no_browser),
            Recorder(RecorderConfig()),
        ).write(args.run_dir)
        print(f"bench saved to {path}")


if __name__ == "__main__":
    main()
