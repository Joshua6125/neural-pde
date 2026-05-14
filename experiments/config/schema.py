"""
Configuration schema for experiments.

Uses dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from typing import Any

from src.loss_functions import AlgorithmConfig
from src.models import AnyModelConfig
from src.integration import AnyIntegrationConfig, MonteCarloConfig
from src.train import TrainConfig

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    domain: str  # "wave_equation", "heat_equation"
    problem_params: dict[str, Any] = field(default_factory=dict)
    methods: list[dict] = field(default_factory=list)
    models: list[dict] = field(default_factory=list)
    training: TrainConfig = field(default_factory=TrainConfig)
    integration: AnyIntegrationConfig = field(default_factory=MonteCarloConfig)
    test_data_params: dict[str, Any] = field(default_factory=dict)  # n_time, n_space, etc.
    source_spec: dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        """Validate configuration consistency."""
        if not self.domain:
            raise ValueError("domain must be specified")
        if not self.methods:
            raise ValueError("At least one method must be specified")
        if not self.models:
            raise ValueError("At least one model must be specified")


@dataclass
class ExperimentCombination:
    """A single method-model pair to train."""
    label: str
    model_config: AnyModelConfig | None = None
    algorithm_config: AlgorithmConfig | None = None
