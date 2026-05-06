from typing import Mapping

import flax.linen as nn
import jax.numpy as jnp
from flax import nnx
from jaxkan.models.KAN import KAN


class KANModel(nn.Module):
    """Linen-compatible wrapper around jaxKAN's NNX KAN model.

    The public behaviour matches ``MLP``:
    - ``model.init(key, x)`` returns Linen variables
    - ``model.apply(variables, x)`` returns ``dict[str, jnp.ndarray]``
    """

    hidden_dim: int
    num_layers: int
    output_heads: Mapping[str, int]
    input_dim: int
    grid_size: int = 5
    degree: int = 3
    model_type: str = "efficient"  # aliases: "efficient" | "cheby" | "original"
    seed: int = 42

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be strictly positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be strictly positive")
        if len(self.output_heads) == 0:
            raise ValueError("output_heads must be non-empty")
        for name, dim in self.output_heads.items():
            if not name:
                raise ValueError("output head names must be non-empty")
            if dim <= 0:
                raise ValueError("each output head dimension must be strictly positive")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be strictly positive")

    def _layer_dims(self) -> list[int]:
        total_out_dim = sum(self.output_heads.values())
        return [self.input_dim] + [self.hidden_dim] * self.num_layers + [total_out_dim]

    def setup(self):
        self.validate()
        layer_type, required_parameters = self._kan_hparams()
        self.kan = nnx.bridge.to_linen(
            KAN,
            self._layer_dims(),
            layer_type=layer_type,
            required_parameters=required_parameters,
            seed=self.seed,
            skip_rng=True,
            name="kan_backbone",
        )

    @nn.compact
    def __call__(self, x) -> dict[str, jnp.ndarray]:
        self.validate()

        was_unbatched = x.ndim == 1
        x_in = x[None, :] if was_unbatched else x

        y = self.kan(x_in)

        outputs = self._split_output_heads(y)

        if was_unbatched:
            return {name: value[0] for name, value in outputs.items()}
        return outputs


    def _split_output_heads(self, y: jnp.ndarray) -> dict[str, jnp.ndarray]:
        outputs: dict[str, jnp.ndarray] = {}
        start = 0
        for name, dim in self.output_heads.items():
            outputs[name] = y[..., start : start + dim]
            start += dim
        return outputs

    def _kan_hparams(self) -> tuple[str, dict[str, object]]:
        model_type = self.model_type.lower()

        if model_type == "original":
            return "base", {"k": self.degree, "G": self.grid_size}

        if model_type == "cheby":
            return "chebyshev", {"D": self.degree, "flavor": "default"}

        if model_type == "efficient":
            return "chebyshev", {"D": self.degree, "flavor": "exact"}

        if model_type in {"base", "spline"}:
            return model_type, {"k": self.degree, "G": self.grid_size}

        if model_type == "chebyshev":
            return "chebyshev", {"D": self.degree, "flavor": "default"}

        raise ValueError(
            "Unknown model_type. Supported values: "
            "efficient, cheby, original, base, spline, chebyshev"
        )
