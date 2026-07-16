"""Pluggable cluster backends for submitting + monitoring eval jobs.

An ``EvalBackend`` abstracts the cluster submission layer (Iris TPU/GPU,
SkyPilot, SLURM, local). The package ships an :class:`IrisBackend` adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class EvalBackend(Protocol):
    """Interface for cluster job submission + monitoring."""

    def submit(
        self,
        *,
        command: List[str],
        job_name: str,
        env_vars: Dict[str, str],
        accelerator: str,
        replicas: int = 1,
        cpu: float = 8.0,
        memory: str = "256GB",
        disk: str = "100GB",
        task_image: Optional[str] = None,
        priority: str = "interactive",
        max_retries: int = 0,
        timeout: int = 0,
        secrets_env: Optional[str] = None,
        dry_run: bool = False,
        no_wait: bool = False,
    ) -> Any:
        """Submit a job to the cluster. Returns a job handle or exit code."""
        ...

    def query(self, job_id: str) -> Any:
        """Query the status of a submitted job."""
        ...

    def logs(self, job_id: str, *, follow: bool = True) -> Any:
        """Stream or fetch logs for a submitted job."""
        ...


__all__ = ["EvalBackend"]
