"""
The purpose of this experiment is to compar pairs of loss functions and neural architectures.

In particular, all loss functions are compared to one another with MLP.
"""


from omegaconf import DictConfig


def run(
    cfg: DictConfig,
    output_dir: str,
    generate_data: bool=True,
    make_plots: bool=True
):
    """
    Entry point for an experiment.

    Parameters
    ----------
    cfg : DictConfig
        The overridden experiment configuration.
    output_dir : str
        The path where all results/artifacts should be stored.
    prev_data : str | None
        Path to previous run of experiments. This skips execution.
    make_plots : bool
        Should plots be generated.
    """



    # Save a dummy artifact
    # artifact_path = os.path.join(output_dir, "results.txt")
    # with open(artifact_path, "w") as f:
    #     f.write("Experiment completed successfully.")

    # print(f"Saved artifact to {artifact_path}")
