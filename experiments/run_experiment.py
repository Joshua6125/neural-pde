#!/usr/bin/env python
"""
CLI entry point for running experiments.

Usage:
    python run_experiment.py --experiment wave_1d
    python run_experiment.py --config config/experiment_1.yaml
    python run_experiment.py --config config/experiment_1.yaml --override training.epochs=200
    python run_experiment.py --validate config/experiment_1.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config_loader import ConfigLoader
from experiment_base import BaseExperiment
from registry import get_experiment_config


def main():
    parser = argparse.ArgumentParser(description="Run an experiment")

    # Experiment selection
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name from registry (e.g., wave_1d)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )

    # Options
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Configuration overrides (dot-notation, e.g., training.epochs=200)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate config without running experiment",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Explicit run ID (defaults to timestamp)",
    )

    args = parser.parse_args()

    # Determine config source
    if args.experiment:
        print(f"Loading experiment: {args.experiment}")
        config = get_experiment_config(args.experiment)
    elif args.config:
        loader = ConfigLoader("experiments/config")
        config_file = Path(args.config).name
        config = loader.load(config_file)
    else:
        print("Error: specify --experiment or --config")
        parser.print_help()
        sys.exit(1)

    # Apply overrides
    if args.override:
        overrides = {}
        for override in args.override:
            key, value = override.split("=", 1)
            # Try to parse as number
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            overrides[key] = value

        loader = ConfigLoader("experiments/config")
        config = loader._build_config(loader._apply_overrides(config.__dict__, overrides))

    # Validate
    config.validate()

    if args.validate:
        print("\n✓ Configuration valid!")
        print(f"  Name: {config.name}")
        print(f"  Domain: {config.domain}")
        print(f"  Methods: {[m.name for m in config.methods]}")
        print(f"  Models: {[m.name for m in config.models]}")
        print(f"  Training epochs: {config.training.epochs}")
        sys.exit(0)

    # Run experiment
    print("\nRunning experiment...")
    experiment = BaseExperiment(config=config)
    experiment.execute(run_id=args.run_id)
    experiment.print_summary()


if __name__ == "__main__":
    main()
