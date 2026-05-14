"""
Abstract base class for domain plugins.

Defines the interface that all domain-specific implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
import jax.numpy as jnp

from src.loss_functions import AlgorithmConfig
from src.models import AnyModelConfig


class DomainPlugin(ABC):
    """Abstract base for domain-specific plugins.

    Each problem domain implements:
    1. Analytical solutions for ground truth
    2. Test data generation (points + reference values)
    3. Domain-specific visualisation

    Plugins are instantiated once per experiment and passed configuration.
    """

    @abstractmethod
    def analytical_solution(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Compute analytical solution at (t, x) points.

        Args:
            t: Time values, shape (n,)
            x: Space values, shape (n,) or (n, d) for d-dimensional spaces

        Returns:
            Solution values at those points, shape (n,)
        """
        pass

    @abstractmethod
    def get_test_data(
        self,
        n_time: int = 20,
        n_space: int = 80,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate or load test data for evaluation.

        Args:
            n_time: Number of time points
            n_space: Number of spatial points per dimension
            seed: Random seed for reproducibility

        Returns:
            (test_points, reference_solutions) tuple where:
            - test_points: shape (n_samples, n_dims), typically (n, 2) for (t, x)
            - reference_solutions: shape (n_samples,)
        """
        pass

    @abstractmethod
    def plot_domain_specific(
        self,
        data: dict[str, Any],
        output_path: str,
    ) -> None:
        """Create domain-specific visualisations.

        Args:
            data: Dictionary with visualisation data (domain-dependent)
                Common keys: 'test_points', 'predictions', 'reference'
            output_path: Directory to save plots
        """
        pass

    @abstractmethod
    def build_source_configs(
        self,
        model_data: dict,
        method_data: dict
    ) -> tuple[AnyModelConfig, AlgorithmConfig]:
        """Build source-native model and algorithm configs for a combination."""
        pass

    @abstractmethod
    def get_sample_input(self) -> jnp.ndarray:
        """Return a representative sample input for model initialisation."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the domain."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of the PDE/problem."""
        pass
