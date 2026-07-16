"""LocalResultSink — writes a result.json summary to an output directory.

A lightweight sink that collects Harbor job statistics (trial counts, reward
summaries, exception breakdown) and persists them as JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .infra_errors import compute_infra_error_stats


class LocalResultSink:
    """Writes ``result.json`` with job stats to an output directory."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def publish(
        self,
        *,
        job_dir: Path,
        job_name: str,
        model_name: Optional[str],
        benchmark_name: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        job_dir = Path(job_dir)
        result_path = self.output_dir / f"{job_name}_result.json"

        summary: Dict[str, Any] = {
            "job_name": job_name,
            "model_name": model_name,
            "benchmark_name": benchmark_name,
            "job_dir": str(job_dir),
        }

        # Try to read Harbor's result.json + stats
        harbor_result_path = job_dir / "result.json"
        if harbor_result_path.exists():
            try:
                harbor_result = json.loads(harbor_result_path.read_text())
                stats = harbor_result.get("stats", {})
                summary["stats"] = stats
                n_infra, infra_breakdown = compute_infra_error_stats(stats)
                summary["n_infra_errors"] = n_infra
                summary["infra_error_breakdown"] = infra_breakdown
            except Exception as e:
                summary["stats_error"] = str(e)

        # Count trial directories
        trial_dirs = [d for d in job_dir.iterdir() if d.is_dir() and d.name != "logs"]
        summary["n_trial_dirs"] = len(trial_dirs)

        if metadata:
            summary["metadata"] = metadata

        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(summary, indent=2))
        print(f"[local-sink] Wrote result summary to {result_path}")
        return str(result_path)


__all__ = ["LocalResultSink"]
