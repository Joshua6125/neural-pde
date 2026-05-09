"""
Visualisation for comparing multiple experiment runs.

Usage:
    viz = ComparisonVisualiser()

    # Compare convergence across runs
    viz.plot_convergence_comparison(run_ids, output_path="plots/convergence_comparison.png")

    # Compare errors across runs
    viz.plot_error_comparison(run_ids, output_path="plots/errors_comparison.png")
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from results_loader import ResultsLoader


class ComparisonVisualiser:
    """Create comparison visualisations across multiple runs."""

    def __init__(self, results_root: str = "results"):
        """Initialize visualiser.

        Args:
            results_root: Root directory for results
        """
        self.loader = ResultsLoader(results_root)

    def plot_convergence_comparison(
        self,
        run_ids: List[str],
        output_path: str = "results/convergence_comparison.png",
    ) -> None:
        """Plot training convergence curves from multiple runs overlaid.

        Args:
            run_ids: List of run IDs to compare
            output_path: Output file path
        """
        plt.figure(figsize=(12, 6))

        for run_id in run_ids:
            try:
                run = self.loader.load_run(run_id)
                metadata = run["metadata"]

                # This is a placeholder; actual convergence data would come from
                # metrics stored in artifacts
                config = metadata.get("config", {})
                label = f"{metadata.get('domain', 'unknown')} ({run_id[:8]})"

                # TODO: Load metrics from artifacts and plot
                # metrics = load_metrics(run["artifacts_dir"])
                # plt.plot(epochs, losses, label=label)

            except Exception as e:
                print(f"  Warning: Could not load convergence for {run_id}: {e}")

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Convergence: Multi-Run Comparison", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.yscale("log")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Convergence comparison saved to {output_path}")

    def plot_error_comparison(
        self,
        run_ids: List[str],
        output_path: str = "results/error_comparison.png",
    ) -> None:
        """Create side-by-side error heatmaps for multiple runs.

        Args:
            run_ids: List of run IDs to compare
            output_path: Output file path
        """
        n_runs = len(run_ids)
        fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5))

        if n_runs == 1:
            axes = [axes]

        for idx, run_id in enumerate(run_ids):
            try:
                run = self.loader.load_run(run_id)
                metadata = run["metadata"]

                # This is a placeholder; actual error data would come from
                # predictions stored in artifacts
                title = f"{metadata.get('domain', 'unknown')}\n{run_id[:16]}"
                axes[idx].set_title(title, fontsize=12)
                axes[idx].text(0.5, 0.5, "No error data", ha="center", va="center")

                # TODO: Load error arrays and display as heatmap
                # errors = load_errors(run["artifacts_dir"])
                # im = axes[idx].imshow(errors, cmap="hot")
                # plt.colorbar(im, ax=axes[idx])

            except Exception as e:
                print(f"  Warning: Could not load errors for {run_id}: {e}")
                axes[idx].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Error comparison saved to {output_path}")

    def plot_method_ranking(
        self,
        run_ids: List[str],
        metric: str = "l2_error",
        output_path: str = "results/method_ranking.png",
    ) -> None:
        """Create ranking visualisation for methods across runs.

        Args:
            run_ids: List of run IDs
            metric: Metric to rank by
            output_path: Output file path
        """
        # Placeholder: actual implementation would extract metrics and create bar chart
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Method ranking by {metric}\n(placeholder)",
                ha="center", va="center", fontsize=14)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Method ranking saved to {output_path}")
