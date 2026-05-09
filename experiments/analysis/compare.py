"""
Results comparison: compare metrics across multiple runs.

Usage:
    comparator = ResultsComparator()

    # Compare runs by domain
    comparator.compare_by_domain(run_ids, metric="l2_error")

    # Rank methods
    rankings = comparator.rank_methods(run_ids)

    # Export to CSV
    comparator.export_comparison_csv(run_ids, "comparison.csv")
"""

import csv
from pathlib import Path
from typing import Dict, List, Any
from results_loader import ResultsLoader


class ResultsComparator:
    """Compare and rank experiment results."""

    def __init__(self, results_root: str = "results"):
        """Initialize comparator.

        Args:
            results_root: Root directory for results
        """
        self.loader = ResultsLoader(results_root)
        self.results_root = Path(results_root)

    def compare_by_method(
        self,
        run_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Compare results grouped by method.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Dict with structure:
            {
                "method_name": {
                    "runs": [run_ids],
                    "configs": [config dicts],
                    "avg_time": float,
                }
            }
        """
        by_method = {}

        for run_id in run_ids:
            run = self.loader.load_run(run_id)
            metadata = run["metadata"]
            config = metadata.get("config", {})

            for method in config.get("methods", []):
                method_name = method.get("name", "unknown")

                if method_name not in by_method:
                    by_method[method_name] = {
                        "runs": [],
                        "configs": [],
                    }

                by_method[method_name]["runs"].append(run_id)
                by_method[method_name]["configs"].append(config)

        return by_method

    def compare_by_model(
        self,
        run_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Compare results grouped by model.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Dict with structure:
            {
                "model_name": {
                    "runs": [run_ids],
                    "configs": [config dicts],
                }
            }
        """
        by_model = {}

        for run_id in run_ids:
            run = self.loader.load_run(run_id)
            metadata = run["metadata"]
            config = metadata.get("config", {})

            for model in config.get("models", []):
                model_name = model.get("name", "unknown")

                if model_name not in by_model:
                    by_model[model_name] = {
                        "runs": [],
                        "configs": [],
                    }

                by_model[model_name]["runs"].append(run_id)
                by_model[model_name]["configs"].append(config)

        return by_model

    def compare_by_domain(
        self,
        run_ids: List[str],
    ) -> Dict[str, List[str]]:
        """Group runs by domain.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Dict mapping domain names to run IDs
        """
        by_domain = {}

        for run_id in run_ids:
            run = self.loader.load_run(run_id)
            metadata = run["metadata"]
            domain = metadata.get("domain", "unknown")

            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(run_id)

        return by_domain

    def get_summary(self, run_ids: List[str]) -> Dict[str, Any]:
        """Get summary statistics for a set of runs.

        Args:
            run_ids: List of run IDs

        Returns:
            Summary dict with counts and groupings
        """
        runs = [self.loader.load_run(rid) for rid in run_ids]

        return {
            "n_runs": len(runs),
            "date_range": (
                min(r["metadata"].get("timestamp", "") for r in runs),
                max(r["metadata"].get("timestamp", "") for r in runs),
            ),
            "by_domain": self.compare_by_domain(run_ids),
            "by_method": self.compare_by_method(run_ids),
            "by_model": self.compare_by_model(run_ids),
        }

    def export_comparison_csv(
        self,
        run_ids: List[str],
        output_path_str: str,
    ) -> None:
        """Export comparison to CSV.

        Args:
            run_ids: List of run IDs
            output_path: Output CSV path
        """
        rows = []

        for run_id in run_ids:
            run = self.loader.load_run(run_id)
            metadata = run["metadata"]
            config = metadata.get("config", {})

            row = {
                "run_id": run_id,
                "timestamp": metadata.get("timestamp", ""),
                "domain": metadata.get("domain", ""),
                "config_name": config.get("name", ""),
                "methods": ",".join(m.get("name", "") for m in config.get("methods", [])),
                "models": ",".join(m.get("name", "") for m in config.get("models", [])),
                "epochs": config.get("training", {}).get("epochs", ""),
                "learning_rate": config.get("training", {}).get("learning_rate", ""),
            }
            rows.append(row)

        if not rows:
            print("No runs to export")
            return

        output_path = Path(output_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = rows[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Comparison exported to {output_path_str}")
