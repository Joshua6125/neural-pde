"""
Configuration loader: loads YAML configs and merges with defaults.

Usage:
    config = ConfigLoader("config/experiment_1.yaml").load()

    # With Python overrides:
    config = ConfigLoader("config/experiment_1.yaml").load(
        overrides={"training.epochs": 200, "methods[0].name": "pinn"}
    )
"""

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional, Dict, cast

import optax
import yaml
from experiments.config.schema import (
    ExperimentConfig,
    MethodSpec,
    ModelSpec,
)

from src.integration import QuadratureConfig, MonteCarloConfig
from src.train import TrainConfig


class ConfigLoader:
    """Load and merge experiment configurations."""

    def __init__(self, config_dir: str = "config"):
        """Initialize loader with config directory.

        Args:
            config_dir: Directory containing YAML config files
        """
        self.config_dir = Path(config_dir)
        self.defaults_path = self.config_dir / "defaults.yaml"

        if not self.defaults_path.exists():
            raise FileNotFoundError(f"defaults.yaml not found in {self.config_dir}")

    def load(
        self,
        config_file: str = "experiment_1.yaml",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ExperimentConfig:
        """Load and merge configuration.

        Args:
            config_file: Name of experiment config (relative to config_dir)
            overrides: Dict of dot-notation overrides (e.g., {"training.epochs": 200})

        Returns:
            Merged ExperimentConfig

        Raises:
            FileNotFoundError: If config files don't exist
            ValueError: If configuration is invalid
        """
        # Load defaults
        with open(self.defaults_path, "r") as f:
            defaults = yaml.safe_load(f) or {}

        # Load experiment config
        experiment_path = self.config_dir / config_file
        if not experiment_path.exists():
            raise FileNotFoundError(f"Config file not found: {experiment_path}")

        with open(experiment_path, "r") as f:
            experiment_yaml = yaml.safe_load(f) or {}

        # Merge: experiment overrides defaults, but nested mappings merge deeply.
        merged = self._deep_merge(defaults, experiment_yaml)

        # Apply programmatic overrides
        if overrides:
            merged = self._apply_overrides(merged, overrides)

        self.old_config = merged

        # Convert to dataclass
        config = self._build_config(merged, source_spec=deepcopy(merged))
        config.validate()

        return config

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge two dictionaries without mutating the inputs."""
        merged = deepcopy(base)

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)

        return merged

    @staticmethod
    def _apply_overrides(config: dict, overrides: dict) -> dict:
        """Apply dot-notation overrides to config dict.

        Examples:
            "training.epochs" -> config["training"]["epochs"]
            "methods[0].name" -> config["methods"][0]["name"]
        """
        for key_path, value in overrides.items():
            keys = key_path.split(".")
            current = config

            # Navigate to parent
            for key in keys[:-1]:
                # Handle list indexing: "methods[0]" -> "methods", 0
                if "[" in key:
                    list_name, index_str = key.split("[")
                    index = int(index_str.rstrip("]"))
                    current = current[list_name][index]
                else:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

            # Set final value
            final_key = keys[-1]
            if "[" in final_key:
                list_name, index_str = final_key.split("[")
                index = int(index_str.rstrip("]"))
                current[list_name][index] = value
            else:
                current[final_key] = value

        return config

    def _build_config(self, data: dict, source_spec: Optional[dict[str, Any]] = None) -> ExperimentConfig:
        """Build ExperimentConfig from dict."""
        return ExperimentConfig(
            name=data.get("name", "unnamed"),
            domain=data.get("domain", ""),
            problem_params=data.get("problem_params", {}),
            methods=[MethodSpec.from_mapping(method) for method in data.get("methods", [])],
            models=[ModelSpec.from_mapping(model) for model in data.get("models", [])],
            training=self._build_training_config(data.get("training", {})),
            integration=self._build_integration_config(data.get("integration", {})),
            test_data_params=data.get("test_data_params", {}),
            source_spec=source_spec or {},
        )

    def _field_names(self, cls):
        return {f.name for f in fields(cls)}

    def _build_integration_config(self, data: dict):
        integration_type = (
            data.get("strategy")
            or data.get("kind")
            or data.get("integration_method")
            or "monte_carlo"
        )
        specific_data = data.get(integration_type, {})
        if not isinstance(specific_data, dict):
            specific_data = {}

        # choose config class and allowed keys
        if integration_type == "monte_carlo":
            ConfigCls = MonteCarloConfig
        elif integration_type == "quadrature":
            ConfigCls = QuadratureConfig
        else:
            raise ValueError(f"Unrecognised integration type: {integration_type!r}")

        allowed = self._field_names(ConfigCls) | {"strategy", "kind", "integration_method", "monte_carlo", "quadrature"}
        unknown = set(data.keys()) - allowed
        if unknown:
            raise AttributeError(f"Unknown attributes in integration config: {unknown!r}")

        config_kwargs = {k: v for k, v in data.items() if k in self._field_names(ConfigCls)}
        config_kwargs.update({k: v for k, v in specific_data.items() if k in self._field_names(ConfigCls)})
        return ConfigCls(**config_kwargs)

    def _build_learning_rate_schedule(self, spec: Any) -> optax.Schedule:
        """Build an optax schedule from a scalar or schedule specification."""
        if callable(spec):
            return cast(optax.Schedule, spec)

        if isinstance(spec, (int, float)):
            return optax.constant_schedule(float(spec))

        if not isinstance(spec, dict):
            raise TypeError(
                "learning_rate must be a scalar, callable, or a schedule specification dict"
            )

        kind = str(spec.get("kind", "constant")).lower()

        if kind == "constant":
            return optax.constant_schedule(float(spec.get("value", spec.get("init_value", 1e-3))))

        if kind == "exponential_decay":
            return optax.exponential_decay(
                init_value=float(spec.get("init_value", 1e-3)),
                transition_steps=int(spec.get("transition_steps", 1000)),
                decay_rate=float(spec.get("decay_rate", 0.95)),
                staircase=bool(spec.get("staircase", True)),
                end_value=spec.get("end_value"),
                transition_begin=int(spec.get("transition_begin", 0)),
            )

        if kind == "cosine_decay":
            return optax.cosine_decay_schedule(
                init_value=float(spec.get("init_value", 1e-3)),
                decay_steps=int(spec.get("decay_steps", 1000)),
                alpha=float(spec.get("alpha", 0.0)),
            )

        if kind == "piecewise_constant":
            boundaries_and_scales = spec.get("boundaries_and_scales")
            if boundaries_and_scales is None:
                raise ValueError("piecewise_constant schedule requires 'boundaries_and_scales'")
            return optax.piecewise_constant_schedule(
                init_value=float(spec.get("init_value", 1e-3)),
                boundaries_and_scales=dict(boundaries_and_scales),
            )

        raise ValueError(f"Unknown learning rate schedule kind: {kind}")

    def _build_training_config(self, data: dict):
        learning_rate = self._build_learning_rate_schedule(data.get("learning_rate", {}))

        allowed  = self._field_names(TrainConfig)
        unknown = set(data.keys()) - allowed
        if unknown:
            raise AttributeError(f"Unknown attributes in integration config: {unknown!r}")

        config_kwargs = {k: v for k, v in data.items() if k in self._field_names(TrainConfig)}
        config_kwargs["learning_rate"] = learning_rate
        return TrainConfig(**config_kwargs)


