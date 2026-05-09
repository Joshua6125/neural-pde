"""
Results manager: handles versioned result storage and metadata tracking.

Each experiment run gets a timestamped directory with:
- metadata.json: Config snapshot, git commit, domain, methods
- plots/: PNG visualisations
- artifacts/: Raw numpy arrays, pickles, model checkpoints
- runs_index.json: Global index of all runs for quick lookup
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess


class RunID:
    """Timestamped run identifier."""

    @staticmethod
    def generate() -> str:
        """Generate a new run ID: YYYY-MM-DD_HH-mm-ss.

        Returns:
            Run ID string
        """
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def parse(run_id: str) -> Optional[datetime]:
        """Parse run ID back to datetime.

        Args:
            run_id: Run ID string

        Returns:
            datetime object or None if invalid format
        """
        try:
            return datetime.strptime(run_id, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            return None


class ResultsManager:
    """Manage versioned result directories and metadata."""

    def __init__(self, results_root: str = "results"):
        """Initialize results manager.

        Args:
            results_root: Root directory for all results
        """
        self.results_root = Path(results_root)
        self.index_path = self.results_root / "runs_index.json"
        self._ensure_root_exists()

    def _ensure_root_exists(self) -> None:
        """Create results root and index if needed."""
        self.results_root.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({})

    def create_run(
        self,
        config: Dict[str, Any],
        domain: str,
        run_id: Optional[str] = None,
    ) -> "RunManager":
        """Create a new versioned run directory.

        Args:
            config: Experiment configuration dict
            domain: Domain name (for metadata)
            run_id: Optional explicit run ID; generates timestamp if not provided

        Returns:
            RunManager for this run
        """
        if run_id is None:
            run_id = RunID.generate()

        run_dir = self.results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)

        # Write metadata
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "config": config,
            "git_commit": self._get_git_commit(),
        }

        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update global index
        self._add_to_index(run_id, metadata)

        return RunManager(run_dir, metadata)

    def load_run(self, run_id: str) -> "RunManager":
        """Load metadata for an existing run.

        Args:
            run_id: Run ID to load

        Returns:
            RunManager for this run

        Raises:
            FileNotFoundError: If run doesn't exist
        """
        run_dir = self.results_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return RunManager(run_dir, metadata)

    def list_runs(self) -> list[str]:
        """List all run IDs in chronological order.

        Returns:
            List of run ID strings
        """
        index = self._read_index()
        # Sort by timestamp
        return sorted(index.keys(), key=lambda rid: index[rid].get("timestamp", ""))

    def _add_to_index(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """Add run to global index."""
        index = self._read_index()
        index[run_id] = {
            "timestamp": metadata["timestamp"],
            "domain": metadata["domain"],
            "config_name": metadata.get("config", {}).get("name", "unnamed"),
        }
        self._write_index(index)

    def _read_index(self) -> Dict[str, Any]:
        """Read runs index."""
        if not self.index_path.exists():
            return {}
        with open(self.index_path, "r") as f:
            return json.load(f)

    def _write_index(self, index: Dict[str, Any]) -> None:
        """Write runs index."""
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Get current git commit hash, if in a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None


class RunManager:
    """Manage a single experiment run's outputs."""

    def __init__(self, run_dir: Path, metadata: Dict[str, Any]):
        """Initialize run manager.

        Args:
            run_dir: Run directory path
            metadata: Metadata dictionary
        """
        self.run_dir = run_dir
        self.metadata = metadata
        self.run_id = metadata["run_id"]

    @property
    def plots_dir(self) -> Path:
        """Directory for saving plots."""
        return self.run_dir / "plots"

    @property
    def artifacts_dir(self) -> Path:
        """Directory for saving raw data/models."""
        return self.run_dir / "artifacts"

    def save_plot(self, filename: str, filepath: Path) -> None:
        """Record a plot file saved to disk.

        Args:
            filename: Short name (e.g., "convergence.png")
            filepath: Actual file path
        """
        # Plots are typically saved by domain visualisation code
        # This is mainly for metadata tracking
        pass

    def save_artifact(self, filename: str, data: Any, format: str = "pickle") -> Path:
        """Save raw data/model artifact.

        Args:
            filename: Name of artifact
            data: Data to save
            format: Format ("pickle", "numpy", "msgpack")

        Returns:
            Path where artifact was saved
        """
        import pickle
        import numpy as np

        output_path = self.artifacts_dir / filename

        if format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
        elif format == "numpy":
            np.save(output_path, data)
        elif format == "msgpack":
            import msgpack
            with open(output_path, "wb") as f:
                msgpack.packb(data, default=str, f=f)
        else:
            raise ValueError(f"Unknown format: {format}")

        return output_path

    def get_metadata(self) -> Dict[str, Any]:
        """Get run metadata."""
        return self.metadata.copy()
