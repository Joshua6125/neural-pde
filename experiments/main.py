"""
Main entry point for running experiments

Full run:
python main.py --config-name experiment1

Only train models:
python main.py --config-name experiment1 make_plots=False

To generate plots of existing trained models and training data:
python main.py --config-name experiment1 hydra.run.dir=outputs/xxxx-xx-xx/xx-xx-xx generate_data=False make_plots=True

Run with modified existing parameters
python main.py --config-name experiment1 training.learning_rate.init_value=1e-4

Or add additional parameters
python main.py --config-name experiment1 +additional_parameter=value
"""

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import importlib


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

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
