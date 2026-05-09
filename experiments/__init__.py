"""
Experiment management system.

Main entry points:
- run_experiment.py: CLI for running experiments
- BaseExperiment: Base class for custom experiments
- registry.py: Pre-registered experiment discovery
- analysis/: Tools for comparing multiple runs
"""

from .experiment_base import BaseExperiment
from .registry import get_experiment_config, list_known_experiments, register_experiment
from .config_loader import ConfigLoader
from .results_manager import ResultsManager, RunManager
from .domains import get_domain, list_domains

__all__ = [
    "BaseExperiment",
    "get_experiment_config",
    "list_known_experiments",
    "register_experiment",
    "ConfigLoader",
    "ResultsManager",
    "RunManager",
    "get_domain",
    "list_domains",
]
