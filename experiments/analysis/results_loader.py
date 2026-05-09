"""
Results loader: load and aggregate results across multiple experiment runs.

Usage:
    loader = ResultsLoader()

    # Load single run
    run = loader.load_run("2026-05-09_14-32-45")

    # Load multiple runs
    runs = loader.load_runs(
        after="2026-05-09",
        domain="wave_equation",
    )

    # Get aggregated metrics
    metrics = loader.get_metrics(run_ids)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any


class ResultsLoader:
    """Load and aggregate experiment results."""

    def __init__(self, results_root: str = "results"):
        """Initialize loader.

        Args:
            results_root: Root directory for results
        """
        self.results_root = Path(results_root)
        self.index_path = self.results_root / "runs_index.json"

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load a single run's metadata and results.

        Args:
            run_id: Run ID string

        Returns:
            Dict with keys: metadata, test_data, metrics, artifacts
        """
        run_dir = self.results_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Load metadata
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load test data if available
        test_data = {}
        test_points_path = run_dir / "artifacts" / "test_points.pkl"
        if test_points_path.exists():
            with open(test_points_path, "rb") as f:
                test_data["test_points"] = pickle.load(f)

        ref_sols_path = run_dir / "artifacts" / "reference_solutions.pkl"
        if ref_sols_path.exists():
            with open(ref_sols_path, "rb") as f:
                test_data["reference_solutions"] = pickle.load(f)

        # List artifacts
        artifacts = {}
        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.exists():
            for f in artifacts_dir.iterdir():
                if f.is_file():
                    artifacts[f.name] = str(f)

        return {
            "run_id": run_id,
            "metadata": metadata,
            "test_data": test_data,
            "artifacts_dir": str(run_dir / "artifacts"),
            "plots_dir": str(run_dir / "plots"),
            "artifacts": artifacts,
        }

    def load_runs(
        self,
        run_ids: Optional[List[str]] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load multiple runs with optional filtering.

        Args:
            run_ids: Explicit list of run IDs to load
            after: Filter runs after this date (YYYY-MM-DD or timestamp)
            before: Filter runs before this date
            domain: Filter by domain name

        Returns:
            List of run dicts
        """
        if run_ids:
            return [self.load_run(rid) for rid in run_ids]

        # Read index to filter
        if not self.index_path.exists():
            return []

        with open(self.index_path, "r") as f:
            index = json.load(f)

        filtered_ids = []

        for run_id, info in index.items():
            timestamp = info.get("timestamp", "")
            run_domain = info.get("domain", "")

            if domain and run_domain != domain:
                continue

            if after and timestamp < after:
                continue

            if before and timestamp > before:
                continue

            filtered_ids.append(run_id)

        return [self.load_run(rid) for rid in sorted(filtered_ids)]

    def get_metrics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and aggregate metrics from runs.

        Args:
            runs: List of run dicts from load_run(s)

        Returns:
            Aggregated metrics dict
        """
        summary = {
            "n_runs": len(runs),
            "domains": set(),
            "methods": set(),
            "models": set(),
            "timestamps": [],
        }

        for run in runs:
            metadata = run["metadata"]
            summary["domains"].add(metadata.get("domain", "unknown"))
            summary["timestamps"].append(metadata.get("timestamp", ""))

            config = metadata.get("config", {})
            for method in config.get("methods", []):
                summary["methods"].add(method.get("name", "unknown"))
            for model in config.get("models", []):
                summary["models"].add(model.get("name", "unknown"))

        # Convert sets to lists for JSON serialization
        summary["domains"] = list(summary["domains"])
        summary["methods"] = list(summary["methods"])
        summary["models"] = list(summary["models"])

        return summary

    def compare_runs(
        self,
        run_ids: List[str],
        metric_name: str = "l2_error",
    ) -> Dict[str, Dict[str, float]]:
        """Compare a metric across runs.

        Args:
            run_ids: List of run IDs to compare
            metric_name: Metric to extract (domain-specific)

        Returns:
            Dict mapping method/model combinations to metric values
        """
        results = {}

        for run_id in run_ids:
            run = self.load_run(run_id)
            metadata = run["metadata"]
            config = metadata.get("config", {})

            # This is a placeholder; actual metric extraction depends on
            # how metrics are stored in artifacts
            results[run_id] = {
                "domain": metadata.get("domain"),
                "timestamp": metadata.get("timestamp"),
                "methods": [m.get("name") for m in config.get("methods", [])],
                "models": [m.get("name") for m in config.get("models", [])],
            }

        return results

    def list_all_runs(self) -> List[str]:
        """List all run IDs in chronological order."""
        if not self.index_path.exists():
            return []

        with open(self.index_path, "r") as f:
            index = json.load(f)

        return sorted(index.keys(), key=lambda rid: index[rid].get("timestamp", ""))
