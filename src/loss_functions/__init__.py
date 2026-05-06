"""Algorithms module: unified training algorithms combining method + loss.

Each algorithm is self-contained and combines:
- Configuration (model + PDE parameters)
- Loss function (residual computation)
- Algorithm class (parameterization + loss binding)
"""

from typing import TypeAlias

from .base import AlgorithmConfig, Loss
from .pinn import PINN, PINNConfig, PINNLoss
from .sls import SLS, SLSConfig, SLSLoss

from ..models import BuiltModelProtocol


AlgorithmConfigType: TypeAlias = PINNConfig | SLSConfig
LossAlgorithm: TypeAlias = PINN | SLS


def build_algorithm(config: AlgorithmConfig, model: BuiltModelProtocol) -> LossAlgorithm:
    """Build an algorithm from configuration and model.

    This factory function instantiates the appropriate algorithm class
    based on the configuration type.

    Parameters
    ----------
    config : AlgorithmConfig
        Algorithm configuration (PINNConfig, SLSConfig, etc.)
    model : BuiltModelProtocol
        Already-built neural network model from build_model()

    Returns
    -------
    LossAlgorithm
        Configured algorithm (implements TrainingMethod) ready for training

    Raises
    ------
    ValueError
        If config type is not recognised
    """
    if isinstance(config, PINNConfig):
        return PINN(model=model, config=config)
    elif isinstance(config, SLSConfig):
        return SLS(model=model, config=config)
    else:
        raise ValueError(f"Unknown algorithm config type: {type(config).__name__}")


__all__ = [
    # Base classes
    "AlgorithmConfig",
    "Loss",
    # PINN
    "PINN",
    "PINNConfig",
    "PINNLoss",
    # SLS
    "SLS",
    "SLSConfig",
    "SLSLoss",
    # Factory
    "build_algorithm",
    # Type alias
    "AlgorithmConfigType",
    "LossAlgorithm",
]
