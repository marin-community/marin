"""
VLM Evaluation Checkpoint Manager.

Provides checkpoint saving and resume functionality for VLM evaluation,
supporting both direct benchmark evaluation and lm-eval harness paths.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import fsspec

logger = logging.getLogger(__name__)


@dataclass
class VLMEvalCheckpoint:
    """Checkpoint state for VLM evaluation."""

    version: str = "1.0"
    task_name: str = ""
    checkpoint_type: str = "vlm_eval"

    # Config snapshot
    config: dict[str, Any] = field(default_factory=dict)

    # State
    completed_indices: list[int] = field(default_factory=list)
    total_examples: int = 0
    last_checkpoint_time: str = ""

    # Predictions keyed by string index
    predictions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Partial metrics
    partial_metrics: dict[str, Any] = field(default_factory=dict)

    def get_completed_set(self) -> set[int]:
        """Get set of completed indices for fast lookup."""
        return set(self.completed_indices)

    def add_result(self, index: int, result: dict[str, Any]):
        """Add a single result to the checkpoint."""
        if index not in self.get_completed_set():
            self.completed_indices.append(index)
        self.predictions[str(index)] = result
        # Update partial metrics
        if result.get("correct"):
            self.partial_metrics["correct"] = self.partial_metrics.get("correct", 0) + 1
        self.partial_metrics["total"] = self.partial_metrics.get("total", 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "task_name": self.task_name,
            "checkpoint_type": self.checkpoint_type,
            "config": self.config,
            "state": {
                "completed_indices": self.completed_indices,
                "total_examples": self.total_examples,
                "last_checkpoint_time": self.last_checkpoint_time,
            },
            "predictions": self.predictions,
            "partial_metrics": self.partial_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLMEvalCheckpoint":
        """Create checkpoint from dictionary."""
        state = data.get("state", {})
        return cls(
            version=data.get("version", "1.0"),
            task_name=data.get("task_name", ""),
            checkpoint_type=data.get("checkpoint_type", "vlm_eval"),
            config=data.get("config", {}),
            completed_indices=state.get("completed_indices", []),
            total_examples=state.get("total_examples", 0),
            last_checkpoint_time=state.get("last_checkpoint_time", ""),
            predictions=data.get("predictions", {}),
            partial_metrics=data.get("partial_metrics", {}),
        )


class VLMCheckpointManager:
    """Manages checkpoint saving and loading for VLM evaluation."""

    def __init__(
        self,
        task_name: str,
        checkpoint_dir: str,
        checkpoint_interval: int = 100,
        config_snapshot: dict[str, Any] | None = None,
        run_id: str | None = None,
    ):
        self.task_name = task_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.config_snapshot = config_snapshot or {}

        # Current checkpoint state
        self.checkpoint = VLMEvalCheckpoint(
            task_name=task_name,
            config=self.config_snapshot,
        )

        # Counter for checkpoint interval
        self._results_since_last_save = 0

        # Run ID for this evaluation
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def checkpoint_path(self) -> str:
        """Get the checkpoint file path."""
        return os.path.join(self.checkpoint_dir, f"{self.task_name}_checkpoint_{self._run_id}.json")

    def should_save(self) -> bool:
        """Check if we should save a checkpoint now."""
        if self.checkpoint_interval <= 0:
            return False
        return self._results_since_last_save >= self.checkpoint_interval

    def add_result(self, index: int, result: dict[str, Any]):
        """Add a result and optionally save checkpoint."""
        self.checkpoint.add_result(index, result)
        self._results_since_last_save += 1

        if self.should_save():
            self.save()
            self._results_since_last_save = 0

    def save(self):
        """Save checkpoint to disk."""
        self.checkpoint.last_checkpoint_time = datetime.now().isoformat()

        try:
            fs, plain_path = fsspec.core.url_to_fs(self.checkpoint_dir)
            fs.makedirs(plain_path, exist_ok=True)

            checkpoint_file = os.path.join(plain_path, f"{self.task_name}_checkpoint_{self._run_id}.json")

            with fs.open(checkpoint_file, "w") as f:
                json.dump(self.checkpoint.to_dict(), f, indent=2, default=str)

            logger.info(
                f"Saved checkpoint: {len(self.checkpoint.completed_indices)} examples "
                f"({self.checkpoint.partial_metrics.get('correct', 0)}/"
                f"{self.checkpoint.partial_metrics.get('total', 0)} correct) "
                f"to {checkpoint_file}"
            )
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def get_remaining_indices(self, total: int) -> list[int]:
        """Get list of indices that still need to be evaluated."""
        completed = self.checkpoint.get_completed_set()
        return [i for i in range(total) if i not in completed]

    def is_completed(self, index: int) -> bool:
        """Check if an example has already been evaluated."""
        return index in self.checkpoint.get_completed_set()

    def finalize(self) -> dict[str, Any]:
        """Finalize evaluation and return merged results."""
        # Save final checkpoint
        self.save()

        # Return aggregated results
        return {
            "total": len(self.checkpoint.completed_indices),
            "correct": self.checkpoint.partial_metrics.get("correct", 0),
            "predictions": self.checkpoint.predictions,
        }

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> "VLMCheckpointManager | None":
        """Load a checkpoint from file."""
        try:
            fs, plain_path = fsspec.core.url_to_fs(checkpoint_path)

            if not fs.exists(plain_path):
                logger.info(f"No checkpoint found at {checkpoint_path}")
                return None

            with fs.open(plain_path, "r") as f:
                data = json.load(f)

            checkpoint = VLMEvalCheckpoint.from_dict(data)

            # Extract run_id from checkpoint path
            basename = os.path.basename(checkpoint_path)
            # Format: {task_name}_checkpoint_{run_id}.json
            run_id = basename.replace(f"{checkpoint.task_name}_checkpoint_", "").replace(".json", "")

            # Create manager from loaded checkpoint
            manager = cls(
                task_name=checkpoint.task_name,
                checkpoint_dir=os.path.dirname(checkpoint_path),
                config_snapshot=checkpoint.config,
                run_id=run_id,
            )
            manager.checkpoint = checkpoint

            logger.info(f"Loaded checkpoint: {len(checkpoint.completed_indices)} examples completed from {checkpoint_path}")

            return manager
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return None

    @classmethod
    def find_latest_checkpoint(cls, checkpoint_dir: str, task_name: str) -> str | None:
        """Find the latest checkpoint for a task."""
        try:
            fs, plain_path = fsspec.core.url_to_fs(checkpoint_dir)

            if not fs.exists(plain_path):
                return None

            pattern = f"{plain_path}/{task_name}_checkpoint_*.json"
            checkpoints = fs.glob(pattern)

            if not checkpoints:
                return None

            # Sort by timestamp in filename (newest first)
            checkpoints.sort(reverse=True)

            # Re-add protocol if needed
            latest = checkpoints[0]
            if checkpoint_dir.startswith("gs://") and not latest.startswith("gs://"):
                latest = f"gs://{latest}"

            return latest
        except Exception as e:
            logger.warning(f"Failed to find latest checkpoint in {checkpoint_dir}: {e}")
            return None
