from ...loss_functions import LSLossConfig, PINNLossConfig
from ...models import LSModelBundle, PINNModelBundle
from .LS import LSMethod
from .PINN import PINNMethod
from .base import TrainingMethod


def get_training_method(
        loss_cfg: LSLossConfig | PINNLossConfig,
        model_bundle: PINNModelBundle | LSModelBundle,
    ) -> TrainingMethod:
    """Factory function for choosing training method plugin."""

    if isinstance(loss_cfg, PINNLossConfig):
        if not isinstance(model_bundle, PINNModelBundle):
            raise ValueError("PINN loss config requires PINN model config/bundle.")
        return PINNMethod(u_model=model_bundle.u_model, loss_cfg=loss_cfg)

    if isinstance(loss_cfg, LSLossConfig):
        if not isinstance(model_bundle, LSModelBundle):
            raise ValueError("LS loss config requires LS model config/bundle.")
        return LSMethod(
            v_model=model_bundle.v_model,
            sigma_model=model_bundle.sigma_model,
            loss_cfg=loss_cfg,
        )

    raise ValueError("Unknown loss configuration type.")


__all__ = ["TrainingMethod", "PINNMethod", "LSMethod", "get_training_method"]
