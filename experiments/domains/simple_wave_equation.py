"""
Wave equation domain plugin.

Implements the manufactured-solution setup used by the legacy experiment:
    u(t, x) = sin(πt/T) * sin(π(x + L)/(2L))

with analytical derivatives and a closed-form forcing term.
"""

from __future__ import annotations

import os
from typing import Any, Tuple
from dataclasses import asdict, fields, is_dataclass
from typing import cast

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.loss_functions import AlgorithmConfig, PINNConfig, SLSConfig, gPINNConfig
from src.models import AnyModelConfig, KANModelConfig, MLPModelConfig

from .base import DomainPlugin


class SimpleWaveEquationDomain(DomainPlugin):
    """1D manufactured wave-equation domain."""

    def __init__(self, L: float = 1.0, T: float = 1.0, c: float = 1.0):
        self.L = float(L)
        self.T = float(T)
        self.c = float(c)
        self._batched_analytical_solution = jax.vmap(self.analytical_solution, in_axes=(0, 0))

    @property
    def name(self) -> str:
        return "Simple Wave Equation (1D)"

    @property
    def description(self) -> str:
        return f"Manufactured wave solution on [0,{self.T}]x[-{self.L},{self.L}]"

    def analytical_solution(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the manufactured analytical solution."""
        t_array = jnp.asarray(t)
        x_array = jnp.asarray(x)

        t_term = jnp.sin(jnp.pi * t_array / self.T)
        x_term = jnp.sin(jnp.pi * (x_array + self.L) / (2 * self.L))
        result = t_term * x_term

        return jnp.where(jnp.isclose(t_array / self.T, 1.0, atol=1e-10), 0.0, result)

    def analytical_solution_t(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Time derivative of the manufactured analytical solution."""
        return (
            jnp.pi / self.T
        ) * jnp.cos(jnp.pi * t / self.T) * jnp.sin(jnp.pi * (x + self.L) / (2 * self.L))

    def analytical_solution_x(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Spatial derivative of the manufactured analytical solution."""
        return (
            jnp.sin(jnp.pi * t / self.T)
            * jnp.cos(jnp.pi * (x + self.L) / (2 * self.L))
            * (jnp.pi / (2 * self.L))
        )

    def source_function(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Closed-form source term for the manufactured PDE."""
        u = self.analytical_solution(t, x)
        coeff = -jnp.pi**2 / self.T**2 + self.c**2 * jnp.pi**2 / (4 * self.L**2)
        return coeff * u

    def zero_vector_source(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Zero vector source used by the SLS formulation."""
        _ = (t, x)
        return jnp.zeros((1,))

    def get_sample_input(self) -> jnp.ndarray:
        """Return a representative sample input for model initialisation."""
        return jnp.asarray([[0.5, 0.0]], dtype=jnp.float32)

    def _canonical_name(self, name: str) -> str:
        return name.strip().lower()

    def _mapping(self, data: Any) -> dict[str, Any]:
        if is_dataclass(data):
            return asdict(cast(Any, data))
        return dict(data)

    def _field_names(self, cls):
        return {f.name for f in fields(cls)}

    def _build_model_config(self, model_data: Any, method_data: Any) -> AnyModelConfig:
        model_data = self._mapping(model_data)
        method_data = self._mapping(method_data)
        model_type = self._canonical_name(model_data.get("name", "mlp"))

        common_keys = {"name", "hidden_dim", "num_layers", "input_dim", "extra_params"}

        if model_type == "mlp":
            ConfigCls = MLPModelConfig
        elif model_type == "kan":
            ConfigCls = KANModelConfig
        else:
            raise ValueError(f"Requested model type {model_type} must be mlp or kan. ")

        allowed = self._field_names(ConfigCls) | common_keys
        unknown = set(model_data.keys()) - allowed
        if unknown:
            raise AttributeError(f"Unknown attributes in integration config: {unknown!r}")

        config_kwargs = {k: v for k, v in model_data.items() if k in self._field_names(ConfigCls)}
        config_kwargs["output_heads"] = method_data.get("output_heads", {})
        return ConfigCls(**config_kwargs)

    def build_source_configs(self, model_data: Any, method_data: Any) -> tuple[AnyModelConfig, AlgorithmConfig]:
        model_data = self._mapping(model_data)
        method_data = self._mapping(method_data)
        method = self._canonical_name(method_data.get("name", ""))
        model_cfg = self._build_model_config(model_data, method_data)

        if method == "pinn":
            algorithm_cfg = PINNConfig(
                model=model_cfg,
                f=lambda v: self.source_function(jnp.array(v[0]), jnp.array(v[1])),
                u0=lambda v: self.analytical_solution(jnp.array(v[0]), jnp.array(v[1])),
                ut0=lambda v: self.analytical_solution_t(jnp.array(v[0]), jnp.array(v[1])),
                c=self.c,
                ic_weight=float(method_data.get("ic_weight", 1.0)),
                bc_weight=float(method_data.get("bc_weight", method_data.get("bc_weights", 100.0))),
            )
        elif method == "sls":
            algorithm_cfg = SLSConfig(
                model=model_cfg,
                f=lambda v: self.source_function(jnp.array(v[0]), jnp.array(v[1])),
                g=lambda v: self.zero_vector_source(jnp.array(v[0]), jnp.array(v[1])),
                v0=lambda v: self.analytical_solution_t(jnp.array(v[0]), jnp.array(v[1])),
                sigma0=lambda v: jnp.array(
                    [self.analytical_solution_x(jnp.array(v[0]), jnp.array(v[1]))]
                ),
            )
        elif method == "gpinn":
            algorithm_cfg = gPINNConfig(
                model=model_cfg,
                f=lambda v: self.source_function(jnp.array(v[0]), jnp.array(v[1])),
                u0=lambda v: self.analytical_solution(jnp.array(v[0]), jnp.array(v[1])),
                ut0=lambda v: self.analytical_solution_t(jnp.array(v[0]), jnp.array(v[1])),
                c=self.c,
                ic_weight=float(method_data.get("ic_weight", 1.0)),
                bc_weight=float(method_data.get("bc_weight", method_data.get("bc_weights", 100.0))),
                residual_grad_weight=float(method_data.get("residual_grad_weight", 1e-2)),
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        return model_cfg, algorithm_cfg

    def get_test_data(
        self,
        n_time: int = 20,
        n_space: int = 80,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test points and reference solutions."""
        np.random.seed(seed)

        t_points = np.linspace(0, self.T, n_time)
        x_points = np.linspace(-self.L, self.L, n_space)

        test_t, test_x = np.meshgrid(t_points, x_points, indexing="ij")
        test_points = np.stack([test_t.reshape(-1), test_x.reshape(-1)], axis=1)

        print(f"Computing analytical solutions for {len(test_points)} test points...")
        reference_solutions = np.asarray(
            self._batched_analytical_solution(
                jnp.asarray(test_points[:, 0]),
                jnp.asarray(test_points[:, 1]),
            )
        )
        print("Reference solutions computed.\n")

        return test_points, reference_solutions

    def plot_domain_specific(
        self,
        data: dict[str, Any],
        output_path: str,
    ) -> None:
        """Create wave-equation specific plots."""
        plot_type = data.get("plot_type", "convergence")

        if plot_type == "convergence":
            self._plot_convergence(data, output_path)
        elif plot_type == "error_map":
            self._plot_error_map(data, output_path)
        elif plot_type == "solution_snapshot":
            self._plot_solution_snapshot(data, output_path)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

    def _plot_convergence(self, data: dict[str, Any], output_path: str) -> None:
        metrics_by_method = data.get("metrics_by_method", {})

        plt.figure(figsize=(10, 6))

        for method_name, metrics in metrics_by_method.items():
            losses = [m.total_loss for m in metrics]
            epochs = [m.step for m in metrics]
            plt.plot(epochs, losses, linewidth=2, label=f"{method_name} Loss")

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Convergence", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.yscale("log")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Convergence plot saved to {output_path}")

    def _plot_error_map(self, data: dict[str, Any], output_path: str) -> None:
        test_points = data.get("test_points")
        predictions_by_method = data.get("predictions_by_method", {})
        u_exact = data.get("reference_solutions")

        if test_points is None or u_exact is None:
            print("  Skipping error map: missing test_points or reference_solutions")
            return

        unique_t = np.unique(test_points[:, 0])
        unique_x = np.unique(test_points[:, 1])

        if len(unique_t) * len(unique_x) == len(u_exact):
            n_methods = len(predictions_by_method)
            fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), sharey=True)

            if n_methods == 1:
                axes = np.array([axes])

            for i, (method_name, prediction) in enumerate(predictions_by_method.items()):
                error = np.abs(prediction - u_exact)
                error_grid = error.reshape(len(unique_t), len(unique_x))

                im = axes[i].imshow(
                    error_grid,
                    extent=(
                        float(unique_x.min()),
                        float(unique_x.max()),
                        float(unique_t.min()),
                        float(unique_t.max()),
                    ),
                    origin="lower",
                    aspect="auto",
                    cmap="hot",
                )
                cbar = plt.colorbar(im, ax=axes[i])
                cbar.set_label("Absolute Error |u_pred - u_exact|", fontsize=11)

                axes[i].set_xlabel("x", fontsize=11)
                axes[i].set_title(f"{method_name} Error", fontsize=13)

            axes[0].set_ylabel("t", fontsize=11)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Error map saved to {output_path}")

    def _plot_solution_snapshot(self, data: dict[str, Any], output_path: str) -> None:
        """Plot solution snapshot at a specific time."""
        _ = (data, output_path)
