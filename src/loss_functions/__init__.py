"""Algorithms module: unified training algorithms combining method + loss.

Each algorithm is self-contained and combines:
- Configuration (model + PDE parameters)
- Loss function (residual computation)
- Algorithm class (parameterization + loss binding)
"""

from typing import TypeAlias

from .base import AlgorithmConfig, Loss
from .fosls import FOSLS, FOSLSConfig, FOSLSLoss
from .pinn import PINN, PINNConfig, PINNLoss
from .gpinn import gPINN, gPINNConfig, gPINNLoss
from .vpinn import vPINN, vPINNConfig, vPINNLoss

from ..models import BuiltModelProtocol


AlgorithmConfigType: TypeAlias = PINNConfig | gPINNConfig | FOSLSConfig | vPINNConfig
LossAlgorithm: TypeAlias = PINN | gPINN | FOSLS | vPINN


def build_algorithm(config: AlgorithmConfig, model: BuiltModelProtocol) -> LossAlgorithm:
    """Build an algorithm from configuration and model.

    This factory function instantiates the appropriate algorithm class
    based on the configuration type.

    Parameters
    ----------
    config : AlgorithmConfig
        Algorithm configuration (PINNConfig, gPINNConfig)
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
    elif isinstance(config, gPINNConfig):
        return gPINN(model=model, config=config)
    elif isinstance(config, FOSLSConfig):
        return FOSLS(model=model, config=config)
    elif isinstance(config, vPINNConfig):
        return vPINN(model=model, config=config)
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
    # gPINN
    "gPINN",
    "gPINNConfig",
    "gPINNLoss",
    # FOSLS
    "FOSLS",
    "FOSLSConfig",
    "FOSLSLoss",
    # vPINN
    "vPINN",
    "vPINNConfig",
    "vPINNLoss",
    # Factory
    "build_algorithm",
    # Type alias
    "AlgorithmConfigType",
    "LossAlgorithm",
]
