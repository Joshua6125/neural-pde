"""
Experiment registry: maps experiment names to configs and domains.

Pre-defines commonly used experiments for easy discovery and reuse.

Usage:
    config = get_experiment_config("wave_1d")

    list_known_experiments()

    register_experiment("my_new_exp", "my_config.yaml", "wave_equation")
"""

import yaml
from pathlib import Path
from typing import Dict, Tuple
from config_loader import ConfigLoader
from config.schema import ExperimentConfig


class ExperimentRegistry:
    """Registry for pre-defined experiments."""

    def __init__(self, config_dir: str = "experiments/config"):
        self.config_dir = Path(config_dir)
        self.loader = ConfigLoader(config_dir)
        self._registry: Dict[str, Tuple[str, str]] = {}  # name -> (config_file, domain)
        self._load_known_experiments()

    def _load_known_experiments(self) -> None:
        """Load experiments from known_experiments.yaml."""
        known_path = self.config_dir / "known_experiments.yaml"
        if not known_path.exists():
            # Create defaults if not present
            self._create_default_registry()
            return

        with open(known_path, "r") as f:
            data = yaml.safe_load(f) or {}

        for exp_name, exp_info in data.items():
            config_file = exp_info.get("config_file")
            domain = exp_info.get("domain")
            if config_file and domain:
                self._registry[exp_name] = (config_file, domain)

    def _create_default_registry(self) -> None:
        """Create default known_experiments.yaml."""
        defaults = {
            "experiment_1": {
                "config_file": "experiment_1.yaml",
                "domain": "wave_equation",
                "description": "Wave equation: methods vs models comparison",
            }
        }

        known_path = self.config_dir / "known_experiments.yaml"
        with open(known_path, "w") as f:
            yaml.dump(defaults, f, default_flow_style=False)

        self._registry = {name: (v["config_file"], v["domain"]) for name, v in defaults.items()}

    def get_experiment(self, name: str) -> ExperimentConfig:
        """Get experiment config by name.

        Args:
            name: Experiment name

        Returns:
            ExperimentConfig

        Raises:
            ValueError: If experiment not found
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(
                f"Unknown experiment '{name}'. Available: {available}"
            )

        config_file, domain = self._registry[name]
        return self.loader.load(config_file)

    def list_experiments(self) -> Dict[str, Dict[str, str]]:
        """List all known experiments.

        Returns:
            Dict mapping experiment names to (config_file, domain)
        """
        result = {}
        for name, (config_file, domain) in self._registry.items():
            result[name] = {"config_file": config_file, "domain": domain}
        return result

    def register(self, name: str, config_file: str, domain: str) -> None:
        """Register a new experiment.

        Args:
            name: Experiment name for lookup
            config_file: YAML config filename
            domain: Domain name
        """
        self._registry[name] = (config_file, domain)
        self._save_registry()

    def _save_registry(self) -> None:
        """Save registry to known_experiments.yaml."""
        known_path = self.config_dir / "known_experiments.yaml"
        data = {}
        for name, (config_file, domain) in self._registry.items():
            data[name] = {
                "config_file": config_file,
                "domain": domain,
            }

        with open(known_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Global singleton registry
_registry = None


def get_registry() -> ExperimentRegistry:
    """Get or create global registry."""
    global _registry
    if _registry is None:
        _registry = ExperimentRegistry("experiments/config")
    return _registry


def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment config by name from global registry.

    Args:
        name: Experiment name

    Returns:
        ExperimentConfig
    """
    return get_registry().get_experiment(name)


def list_known_experiments() -> Dict[str, Dict[str, str]]:
    """List all known experiments."""
    return get_registry().list_experiments()


def register_experiment(name: str, config_file: str, domain: str) -> None:
    """Register a new experiment in global registry."""
    get_registry().register(name, config_file, domain)
