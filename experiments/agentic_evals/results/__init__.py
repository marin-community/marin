"""Pluggable result sinks for eval output.

A ``ResultSink`` receives the Harbor job directory + metadata after a run
completes (called by ``EvalRunner.post_harbor_hook``). The package ships:
- :class:`NoOpResultSink` (default — does nothing)
- :class:`LocalResultSink` (writes result.json locally)
- :class:`HFResultSink` (uploads traces to HuggingFace, opt-in via ``[hf]``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ResultSink(Protocol):
    """Interface for post-run result handling."""

    def publish(
        self,
        *,
        job_dir: Path,
        job_name: str,
        model_name: Optional[str],
        benchmark_name: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Process the completed Harbor job directory.

        Args:
            job_dir: Path to the Harbor job directory (contains trial subdirs).
            job_name: The Harbor job name.
            model_name: The model that was evaluated.
            benchmark_name: The benchmark/dataset identifier.
            metadata: Optional extra metadata dict.

        Returns:
            Optional URL or identifier for the published result.
        """
        ...


class NoOpResultSink:
    """Default sink that does nothing."""

    def publish(
        self,
        *,
        job_dir: Path,
        job_name: str,
        model_name: Optional[str],
        benchmark_name: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        return None


__all__ = ["ResultSink", "NoOpResultSink"]
