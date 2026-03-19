from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .neuralnet import NeuralNet


@dataclass(frozen=True)
class NeuralNetModelConfig:
    """Configuration for the built-in fully connected model."""

    kind: Literal["neuralnet"] = "neuralnet"
    hidden_dim: int = 64
    num_layers: int = 4
    output_dim: int = 1

    def validate(self) -> None:
        assert self.hidden_dim > 0, "hidden_dim must be strictly positive"
        assert self.num_layers > 0, "num_layers must be strictly positive"
        assert self.output_dim > 0, "output_dim must be strictly positive"


@dataclass(frozen=True)
class PINNModelConfig:
    """Model specification for PINN training."""

    kind: Literal["pinn"] = "pinn"
    u_model: NeuralNetModelConfig = field(default_factory=NeuralNetModelConfig)


@dataclass(frozen=True)
class LSModelConfig:
    """Model specification for LS training."""

    kind: Literal["ls"] = "ls"
    v_model: NeuralNetModelConfig = field(
        default_factory=lambda: NeuralNetModelConfig(output_dim=1)
    )
    sigma_model: NeuralNetModelConfig = field(
        default_factory=lambda: NeuralNetModelConfig(output_dim=2)
    )


@dataclass(frozen=True)
class PINNModelBundle:
    """Built model objects required by PINN training."""

    u_model: NeuralNet


@dataclass(frozen=True)
class LSModelBundle:
    """Built model objects required by LS training."""

    v_model: NeuralNet
    sigma_model: NeuralNet


AnyModelConfig: TypeAlias = PINNModelConfig | LSModelConfig
AnyModelBundle: TypeAlias = PINNModelBundle | LSModelBundle


def _build_neuralnet(cfg: NeuralNetModelConfig) -> NeuralNet:
    cfg.validate()
    return NeuralNet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )


def build_model_bundle(model_cfg: AnyModelConfig) -> AnyModelBundle:
    """Build internal model bundle from declarative model config."""
    if isinstance(model_cfg, PINNModelConfig):
        return PINNModelBundle(u_model=_build_neuralnet(model_cfg.u_model))

    if isinstance(model_cfg, LSModelConfig):
        return LSModelBundle(
            v_model=_build_neuralnet(model_cfg.v_model),
            sigma_model=_build_neuralnet(model_cfg.sigma_model),
        )

    raise ValueError("Unknown model config type.")