from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias, Mapping, Protocol, Any, cast
from typing_extensions import runtime_checkable

from .neuralnet import NeuralNet
from .kan import KANModel

import jax
import jax.numpy as jnp
import flax.linen as nn


# AnyBuiltModel: TypeAlias = NeuralNet | KANModel


@runtime_checkable
class BuiltModelProtocol(Protocol):
    def init(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any: ...
    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]: ...


class BuiltModelAdapter:
    def __init__(self, module: nn.Module):
        self._module = module

    def init(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        if isinstance(self._module, KANModel) and self._module.input_dim is None:
            inferred_dim = int(sample_input.shape[-1])
            self._module = replace(self._module, input_dim=inferred_dim)
        return self._module.init(rng_key, sample_input)

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        out = self._module.apply(params, x)
        if not isinstance(out, dict):
            raise TypeError("Model.apply must return a dict[str, ndarray] for training.")
        return cast(dict[str, jnp.ndarray], out)


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
    """Configuration for KAN model."""

    kind: Literal["kan"] = "kan"
    hidden_dim: int = 64
    num_layers: int = 4
    input_dim: int | None = None
    grid_size: int = 5
    degree: int = 3
    model_type: str = "efficient"
    seed: int = 42

    def validate(self) -> None:
        super().validate()
        assert self.hidden_dim > 0, "hidden_dim must be strictly positive"
        assert self.num_layers > 0, "num_layers must be strictly positive"
        if self.input_dim is not None:
            assert self.input_dim > 0, "input_dim must be strictly positive"
        assert self.grid_size > 0, "grid_size must be strictly positive"
        assert self.degree > 0, "degree must be strictly positive"


AnyModelConfig: TypeAlias = NeuralNetModelConfig | KANModelConfig


def build_model(cfg: AnyModelConfig) -> BuiltModelAdapter:
    """Build model from declarative model config."""
    if isinstance(cfg, NeuralNetModelConfig):
        cfg.validate()
        return BuiltModelAdapter(
            NeuralNet(
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                output_heads=cfg.output_heads,
            )
        )

    if isinstance(cfg, KANModelConfig):
        cfg.validate()
        return BuiltModelAdapter(
            KANModel(
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                output_heads=cfg.output_heads,
                input_dim=cfg.input_dim,
                grid_size=cfg.grid_size,
                degree=cfg.degree,
                model_type=cfg.model_type,
                seed=cfg.seed
            )
        )

    raise ValueError("Unknown model config type.")