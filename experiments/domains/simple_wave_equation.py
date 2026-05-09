"""
Wave equation domain plugin.

Implements a 1D wave equation with initial conditions and boundaries:
    ∂²u/∂t² - c²∂²u/∂x² = 0

on domain [0, T] x [-L, L] with:
- Initial conditions: u(0, x) = sin(πx/L), u_t(0, x) = 0
- Boundary conditions: u(t, ±L) = 0
- Analytical solution: u(t,x) = sin(πx/L)cos(πct/L)
"""

from typing import Tuple, Any
import numpy as np
import jax.numpy as jnp
from jax import vmap
from functools import partial
import matplotlib.pyplot as plt
import os

from .base import DomainPlugin


class SimpleWaveEquationDomain(DomainPlugin):
    """1D wave equation with analytical solution."""

    def __init__(self, L: float = 1.0, T: float = 1.0, c: float = 1.0):
        """Initialize wave equation parameters.

        Args:
            L: Half-width of spatial domain [-L, L]
            T: Final time
            c: Wave speed
        """
        self.L = L
        self.T = T
        self.c = c

    @property
    def name(self) -> str:
        return "Simple Wave Equation (1D)"

    @property
    def description(self) -> str:
        return f"∂²u/∂t² = {self.c}²∂²u/∂x² on [0,{self.T}]x[-{self.L},{self.L}]"

    def analytical_solution(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute u(t,x) = sin(πx/L)cos(πct/L).

        Args:
            t: Time value(s)
            x: Space value(s)

        Returns:
            u(t, x) values
        """
        return np.sin(np.pi * x / self.L) * np.cos(np.pi * self.c * t / self.L)

    def get_test_data(
        self,
        n_time: int = 20,
        n_space: int = 80,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test points and reference solutions.

        Args:
            n_time: Number of time points
            n_space: Number of spatial points
            seed: Random seed

        Returns:
            (test_points, reference_solutions) where:
            - test_points: shape (n_time * n_space, 2) with columns [t, x]
            - reference_solutions: shape (n_time * n_space,)
        """
        np.random.seed(seed)

        t_points = np.linspace(0, self.T, n_time)
        x_points = np.linspace(-self.L, self.L, n_space)

        # Create Cartesian product: all (t, x) pairs
        test_points_list = []
        for t in t_points:
            for x in x_points:
                test_points_list.append([t, x])

        test_points = np.array(test_points_list)

        print(f"Computing analytical solutions for {len(test_points)} test points...")
        reference_solutions = []

        for i, (t, x) in enumerate(test_points):
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{len(test_points)}")

            u_ref = float(self.analytical_solution(
                np.array(t), np.array(x)
            ))
            reference_solutions.append(u_ref)

        print("Reference solutions computed.\n")

        return test_points, np.array(reference_solutions)

    def plot_domain_specific(
        self,
        data: dict[str, Any],
        output_path: str,
    ) -> None:
        """Create wave equation specific plots.

        Supported plot types (key 'plot_type'):
        - 'convergence': Training loss curves
        - 'error_map': Absolute error heatmaps by method
        - 'solution_snapshot': Solution u(t,x) visualisation
        """
        plot_type = data.get("plot_type", "convergence")

        if plot_type == "convergence":
            self._plot_convergence(data, output_path)
        elif plot_type == "error_map":
            self._plot_error_map(data, output_path)
        elif plot_type == "solution_snapshot":
            self._plot_solution_snapshot(data, output_path)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

    def _plot_convergence(
        self,
        data: dict[str, Any],
        output_path: str,
    ) -> None:
        """Plot training loss convergence curves."""
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

    def _plot_error_map(
        self,
        data: dict[str, Any],
        output_path: str,
    ) -> None:
        """Plot absolute error heatmaps."""
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
            fig, axes = plt.subplots(
                1, n_methods, figsize=(6 * n_methods, 5), sharey=True
            )

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

    def _plot_solution_snapshot(
        self,
        data: dict[str, Any],
        output_path: str,
    ) -> None:
        """Plot solution snapshot at a specific time."""
        # Placeholder for future use
        pass
