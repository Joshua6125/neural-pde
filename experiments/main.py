"""
Main entry point for running experiments

Full run:
python main.py --config-name experiment1

Only train models:
python main.py --config-name experiment1 make_plots=False

To generate plots of existing trained models and training data:
python main.py --config-name experiment1 hydra.run.dir=outputs/xxxx-xx-xx/xx-xx-xx generate_data=False make_plots=True

To reuse a saved run config as the base for a new run:
python main.py --config-name experiment1 +previous_output_dir=outputs/xxxx-xx-xx/xx-xx-xx generate_data=False make_plots=True

Run with modified existing parameters
python main.py --config-name experiment1 training.learning_rate.init_value=1e-4

Or add additional parameters
python main.py --config-name experiment1 +additional_parameter=value
"""

import sys
from pathlib import Path
from typing import cast

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import importlib


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def _load_previous_output_config(previous_output_dir: str) -> DictConfig:
    config_path = Path(previous_output_dir) / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No saved config found at {config_path}")

    return cast(DictConfig, OmegaConf.load(config_path))


def _task_overrides_config(overrides: list[str]) -> DictConfig:
    filtered_overrides = [
        override
        for override in overrides
        if not override.startswith("previous_output_dir=")
        and not override.startswith("+previous_output_dir=")
    ]

    if not filtered_overrides:
        return OmegaConf.create()

    return OmegaConf.from_dotlist(filtered_overrides)


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

    previous_output_dir = cfg.get("previous_output_dir")
    if previous_output_dir:
        cfg = cast(
            DictConfig,
            OmegaConf.merge(
                _load_previous_output_config(previous_output_dir),
                _task_overrides_config(list(HydraConfig.get().overrides.task)),
            ),
        )

    if "script_name" not in cfg:
        raise ValueError("The configuration must specify 'script_name' (e.g. script_name: my_exp_script).")

    # Dynamically import the experiment script from experiments/scripts/
    script_module = importlib.import_module(f"experiments.scripts.{cfg.script_name}")

    generate_data = cfg.get("generate_data", True)
    make_plots = cfg.get("make_plots", True)

    # Call the main run function of the experiment
    if hasattr(script_module, "run"):
        script_module.run(cfg, output_dir, generate_data=generate_data, make_plots=make_plots)
    else:
        raise AttributeError(f"Script 'experiments.scripts.{cfg.script_name}' must define a 'run(cfg, output_dir)' function.")


if __name__ == "__main__":
    main()
