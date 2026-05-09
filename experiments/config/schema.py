"""
Configuration schema for experiments.

Uses dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class IntegrationConfig:
    """Integration strategy configuration."""
    strategy: str = "monte_carlo"  # "monte_carlo", "quadrature"
    n_interior: int = 1600
    n_boundary: int = 100
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    epochs: int = 100
    learning_rate: float = 1e-3
    seed: int = 42
    batch_size: Optional[int] = None
    optimiser: str = "adamw"


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str  # "mlp", "kan"
    hidden_dim: int = 32
    num_layers: int = 3
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodConfig:
    """Loss function / method configuration."""
    name: str  # "pinn", "sls", "gpinn"
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    domain: str  # "wave_equation", "heat_equation"
    problem_params: dict[str, Any] = field(default_factory=dict)
    methods: list[MethodConfig] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    test_data_params: dict[str, Any] = field(default_factory=dict)  # n_time, n_space, etc.

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
    method: MethodConfig
    model: ModelConfig
    label: str
