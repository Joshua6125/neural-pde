"""Algorithms module: unified training algorithms combining method + loss.

Each algorithm is self-contained and combines:
- Configuration (model + PDE parameters)
- Loss function (residual computation)
- Algorithm class (parameterization + loss binding)
"""

from typing import TypeAlias

from .base import AlgorithmConfig, Loss
from .pinn import PINN, PINNConfig, PINNLoss
from .ls import LS, LSConfig, LSLoss

from ..models import AnyBuiltModel

# Type alias for all available algorithm configurations
AlgorithmConfigType: TypeAlias = PINNConfig | LSConfig


def build_algorithm(config: AlgorithmConfig, model: AnyBuiltModel) -> PINN | LS:
    """Build an algorithm from configuration and model.

    This factory function instantiates the appropriate algorithm class
    based on the configuration type.

    Parameters
    ----------
    config : AlgorithmConfig
        Algorithm configuration (PINNConfig, LSConfig, etc.)
    model : AnyBuiltModel
        Already-built neural network model from build_model()

    Returns
    -------
    PINN or LS
        Configured algorithm (implements TrainingMethod) ready for training

    Raises
    ------
    ValueError
        If config type is not recognised
    """
    if isinstance(config, PINNConfig):
        return PINN(model=model, config=config)
    elif isinstance(config, LSConfig):
        return LS(model=model, config=config)
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
    # LS
    "LS",
    "LSConfig",
    "LSLoss",
    # Factory
    "build_algorithm",
    # Type alias
    "AlgorithmConfigType",
]
