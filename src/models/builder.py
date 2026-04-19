from dataclasses import dataclass, field
from typing import Literal, TypeAlias, Mapping

from .neuralnet import NeuralNet
from .kan import KANModel


AnyBuiltModel: TypeAlias = NeuralNet | KANModel


@dataclass(frozen=True)
class BaseModelConfig:
    kind: str
    output_heads: Mapping[str, int] = field(default_factory=lambda:{"output": 1})

    def validate(self) -> None:
        assert self.output_heads,   "output_heads must not be non-empty"
        assert len(self.output_heads) > 0, "output_heads must be non-empty when provided"
        for name, dim in self.output_heads.items():
            assert name, "output head names must be non-empty"
            assert dim > 0, "each output head dimension must be strictly positive"


@dataclass(frozen=True)
class NeuralNetModelConfig(BaseModelConfig):
    """Configuration for the built-in fully connected model."""

    kind: Literal["neuralnet"] = "neuralnet"
    hidden_dim: int = 64
    num_layers: int = 4

    def validate(self) -> None:
        super().validate()
        assert self.hidden_dim > 0, "hidden_dim must be strictly positive"
        assert self.num_layers > 0, "num_layers must be strictly positive"


@dataclass(frozen=True)
class KANModelConfig(BaseModelConfig):
    kind: Literal['kan'] = "kan"
    hidden_dim: int = 64
    num_layers: int = 4
    grid_size: int = 5
    degree: int = 3
    model_type: str = "efficient"
    seed: int = 42

    def validate(self) -> None:
        super().validate()
        assert self.hidden_dim > 0, "hidden_dim must be strictly positive"
        assert self.num_layers > 0, "num_layers must be strictly positive"
        assert self.grid_size > 0, "grid_size must be strictly positive"
        assert self.degree > 0, "degree must be strictly positive"


AnyModelConfig: TypeAlias = NeuralNetModelConfig | KANModelConfig


def build_model(cfg: AnyModelConfig) -> AnyBuiltModel:
    """Build model from declarative model config."""
    if isinstance(cfg, NeuralNetModelConfig):
        cfg.validate()
        return NeuralNet(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            output_heads=cfg.output_heads,
        )

    if isinstance(cfg, KANModelConfig):
        cfg.validate()
        return KANModel(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            output_heads=cfg.output_heads,
            grid_size=cfg.grid_size,
            degree=cfg.degree,
            model_type=cfg.model_type,
            seed=cfg.seed
        )

    raise ValueError("Unknown model config type.")