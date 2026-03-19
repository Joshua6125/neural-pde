from .loss_base import AnyLossConfig, PINNLossConfig, LSLossConfig, LossBase
from .loss_PINN import LossPINN
from .loss_LS import LossLS


def get_loss_function(
        loss_cfg: AnyLossConfig,
        u_model=None,
        v_model=None,
        sigma_model=None
    ) -> LossBase:
    """Function for choosing loss function.

    Parameters
    ----------
    loss_cfg : LSLossConfig | PINNLossConfig
        Loss configuration object specifying parameters for the chosen method.

    Returns
    -------
    LossBase
        Loss function instance (LossPINN or LossLS).
    """

    if isinstance(loss_cfg, PINNLossConfig):
        if u_model is None:
            raise ValueError("Must provide model in PINN loss function declaration.")

        return LossPINN(
            u_model=u_model,
            c=loss_cfg.c,
            f=loss_cfg.f,
            u0=loss_cfg.u0,
            ut0=loss_cfg.ut0,
            ic_weight=loss_cfg.ic_weight,
            bc_weight=loss_cfg.bc_weight,
        )
    elif isinstance(loss_cfg, LSLossConfig):
        if v_model is None or sigma_model is None:
            raise ValueError("Must provide models in LS loss function declaration.")

        return LossLS(
            v_model=v_model,
            sigma_model=sigma_model,
            f=loss_cfg.f,
            g=loss_cfg.g,
            v0=loss_cfg.v0,
            sigma0=loss_cfg.sigma0,
        )
    else:
        raise ValueError("Unknown loss function config.")

__all__ = [
    "LossBase",
    "LSLossConfig",
    "PINNLossConfig",
    "LossPINN",
    "LossLS",
    "get_loss_function",
]
