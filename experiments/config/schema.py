"""
Configuration schema for experiments.

Uses dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

from src.loss_functions import AlgorithmConfig
from src.models import AnyModelConfig
from src.integration import AnyIntegrationConfig, MonteCarloConfig
from src.train import TrainConfig


@dataclass(frozen=True)
class MethodSpec:
    """Declarative method configuration used by the experiment layer."""

    name: str
    output_heads: dict[str, int] = field(default_factory=lambda: {"u": 1})
    ic_weight: float = 1.0
    bc_weight: float = 1.0
    residual_grad_weight: float = 0.0
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MethodSpec":
        name = str(data.get("name", "")).strip().lower()
        output_heads = {str(key): int(value) for key, value in dict(data.get("output_heads", {"u": 1})).items()}
        bc_weight = data.get("bc_weight", data.get("bc_weights", 1.0))
        known_keys = {"name", "output_heads", "ic_weight", "bc_weight", "bc_weights", "residual_grad_weight"}
        extra_params = {key: value for key, value in data.items() if key not in known_keys}

        return cls(
            name=name,
            output_heads=output_heads,
            ic_weight=float(data.get("ic_weight", 1.0)),
            bc_weight=float(bc_weight),
            residual_grad_weight=float(data.get("residual_grad_weight", 0.0)),
            extra_params=extra_params,
        )


@dataclass(frozen=True)
class ModelSpec:
    """Declarative model configuration used by the experiment layer."""

    name: str
    hidden_dim: int = 32
    num_layers: int = 3
    input_dim: int = 2
    output_heads: dict[str, int] = field(default_factory=lambda: {"u": 1})
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelSpec":
        name = str(data.get("name", "")).strip().lower()
        output_heads = {str(key): int(value) for key, value in dict(data.get("output_heads", {"u": 1})).items()}
        known_keys = {"name", "hidden_dim", "num_layers", "input_dim", "output_heads"}
        extra_params = {key: value for key, value in data.items() if key not in known_keys}

        return cls(
            name=name,
            hidden_dim=int(data.get("hidden_dim", 32)),
            num_layers=int(data.get("num_layers", 3)),
            input_dim=int(data.get("input_dim", 2)),
            output_heads=output_heads,
            extra_params=extra_params,
        )

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    domain: str  # "wave_equation", "heat_equation"
    problem_params: dict[str, Any] = field(default_factory=dict)
    methods: list[MethodSpec] = field(default_factory=list)
    models: list[ModelSpec] = field(default_factory=list)
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

    def method_names(self) -> list[str]:
        return [method.name for method in self.methods]

    def model_names(self) -> list[str]:
        return [model.name for model in self.models]

@dataclass
class ExperimentCombination:
    """A single method-model pair to train."""
    label: str
    model_config: AnyModelConfig | None = None
    algorithm_config: AlgorithmConfig | None = None
