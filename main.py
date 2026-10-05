import argparse

from orchestrator.config import SIMULATION_MODES, OrchestratorConfig
from orchestrator.orchestrator import Orchestrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", default="normal", choices=SIMULATION_MODES.keys()
    )
    args = parser.parse_args()

    Orchestrator(OrchestratorConfig(simulation=SIMULATION_MODES[args.mode])).run()
