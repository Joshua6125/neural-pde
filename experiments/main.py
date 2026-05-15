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
    # Hydra automatically creates a timestamped output directory for the run
    output_dir = HydraConfig.get().runtime.output_dir

    print(f"=== Starting Experiment ===")
    print(f"Output directory: {output_dir}")
    print(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")

    if "script_name" not in cfg:
        raise ValueError("The configuration must specify 'script_name' (e.g. script_name: my_exp_script).")

    # Dynamically import the experiment script from experiments/scripts/
    script_module = importlib.import_module(f"experiments.scripts.{cfg.script_name}")

    # Call the main run function of the experiment
    if hasattr(script_module, "run"):
        script_module.run(cfg, output_dir)
    else:
        raise AttributeError(f"Script 'experiments.scripts.{cfg.script_name}' must define a 'run(cfg, output_dir)' function.")

if __name__ == "__main__":
    main()
