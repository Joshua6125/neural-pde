from .loss_base import LossBase
from .loss_PINN import LossPINN
from .loss_LS import LossLS
from ..config import Config


def get_loss_function(config: Config) -> LossBase:
    """Factory function for choosing loss function.

    Parameters
    ----------
    config : Config
        Configuration object specifying loss function and parameters.

    Returns
    -------
    LossBase
        Loss function instance (LossPINN or LossLS).
    """
    method = config.loss_function.lower()

    if method == 'pinn':
        return LossPINN(config)
    elif method == 'ls':
        return LossLS(config)
    else:
        raise ValueError(
            f"Unknown integration method: '{config.loss_function}'. "
            f"Must be 'PINN' or 'LS'."
        )

__all__ = ["LossBase", "LossPINN", "LossLS"]
