from dataclasses import dataclass, field
from typing import Literal, TypeAlias, Mapping, Protocol, Any, Callable, cast
from typing_extensions import runtime_checkable

from .mlp import MLP
from .kan import KANModel
from .xnode import XNODE

import jax
import jax.numpy as jnp
import flax.linen as nn


# AnyBuiltModel: TypeAlias = MLP | KANModel | XNODE


@runtime_checkable
class BuiltModelProtocol(Protocol):
    def init(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any: ...
    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]: ...


class BuiltModelAdapter:
    def __init__(
        self,
        module: nn.Module,
        initial_condition_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ):
        self._module = module
        self._initial_condition_fn = initial_condition_fn

    @staticmethod
    def _normalize_sample_input(sample_input: jnp.ndarray) -> jnp.ndarray:
        sample = jnp.asarray(sample_input)
        if sample.ndim > 1 and sample.shape[0] == 1:
            sample = jnp.squeeze(sample, axis=0)
        return sample

    @staticmethod
    def _normalize_initial_condition(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.atleast_1d(jnp.asarray(value))

    def init(self, rng_key: jax.Array, sample_input: jnp.ndarray) -> Any:
        sample = self._normalize_sample_input(sample_input)
        if self._initial_condition_fn is None:
            return self._module.init(rng_key, sample)

        initial_condition = self._normalize_initial_condition(self._initial_condition_fn(sample))
        return self._module.init(rng_key, sample, initial_condition)

    def apply(self, params: Any, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        sample = self._normalize_sample_input(x)
        if self._initial_condition_fn is None:
            out = self._module.apply(params, sample)
        else:
            initial_condition = self._normalize_initial_condition(self._initial_condition_fn(sample))
            out = self._module.apply(params, sample, initial_condition)
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
class MLPModelConfig(BaseModelConfig):
    """Configuration for the built-in fully connected model."""

    kind: Literal["mlp"] = "mlp"
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
    input_dim: int = 1
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
        assert self.input_dim > 0, "input_dim must be strictly positive"


@dataclass(frozen=True)
class XNODEConfig(BaseModelConfig):
    """Configuration for XNODE model."""

    kind: Literal["xnode"] = "xnode"
    hidden_dim: int = 64
    num_layers: int = 4
    t_max: float = 1.0

    def validate(self) -> None:
        super().validate()
        assert self.hidden_dim > 0, "hidden_dim must be strictly positive"
        assert self.num_layers > 0, "num_layers must be strictly positive"
        assert self.t_max > 0, "t_max must be strictly positive"


AnyModelConfig: TypeAlias = MLPModelConfig | KANModelConfig | XNODEConfig


def build_model(
    cfg: AnyModelConfig,
    initial_condition_fn: Callable[[jnp.ndarray], jnp.ndarray] | float | jnp.ndarray | None = None,
) -> BuiltModelAdapter:
    """Build model from declarative model config."""
    if initial_condition_fn is not None and not callable(initial_condition_fn):
        constant = jnp.asarray(initial_condition_fn)

        def _constant_initial_condition(_: jnp.ndarray, constant: jnp.ndarray = constant) -> jnp.ndarray:
            return constant

        initial_condition_fn = _constant_initial_condition

    if isinstance(cfg, MLPModelConfig):
        cfg.validate()
        return BuiltModelAdapter(
            MLP(
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

    if isinstance(cfg, XNODEConfig):
        cfg.validate()
        if initial_condition_fn is None:
            raise ValueError("XNODE models require an initial_condition_fn.")
        return BuiltModelAdapter(
            XNODE(
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                output_heads=cfg.output_heads,
                t_max=cfg.t_max,
            ),
            initial_condition_fn=cast(Callable[[jnp.ndarray], jnp.ndarray], initial_condition_fn),
        )

    raise ValueError("Unknown model config type.")