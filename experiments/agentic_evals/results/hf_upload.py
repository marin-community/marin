"""HFResultSink — uploads Harbor eval traces to HuggingFace.

Uses ``harbor.utils.traces_utils.export_traces`` + ``huggingface_hub`` to upload
the traces dataset. Decoupled from Supabase entirely (unlike the OT-Agent
original which coupled DB + HF upload).

The upload logic is adapted from OT-Agent ``hpc/launch_utils.py``
(``upload_traces_to_hf``, ``derive_benchmark_repo``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


def sanitize_repo_for_job(repo_id: str) -> str:
    """Return a filesystem-safe repo identifier."""
    safe = re.sub(r"[^A-Za-z0-9._\-]+", "-", repo_id.strip())
    safe = re.sub(r"-+", "-", safe)
    return safe.strip("-_") or "job"


def sanitize_hf_repo_id(repo_id: str) -> str:
    """Ensure repo ID matches HF naming rules (org/name, lowercase)."""
    repo_id = repo_id.strip()
    if "/" not in repo_id:
        repo_id = f"laion/{repo_id}"
    return repo_id


def derive_benchmark_repo(
    harbor_dataset: Optional[str] = None,
    dataset_path: Optional[str] = None,
    explicit_repo: Optional[str] = None,
) -> str:
    """Derive a benchmark repository identifier from dataset info."""
    raw: Optional[str] = None
    if explicit_repo:
        raw = explicit_repo
    elif harbor_dataset:
        raw = harbor_dataset
    elif dataset_path:
        raw = Path(dataset_path).name

    if not raw:
        return "unknown-benchmark"
    return sanitize_repo_for_job(raw)


class HFResultSink:
    """Uploads eval traces to HuggingFace via harbor's export_traces utility."""

    def __init__(
        self,
        hf_repo_id: str,
        *,
        hf_token: Optional[str] = None,
        hf_private: bool = False,
        hf_episodes: str = "last",
    ):
        self.hf_repo_id = sanitize_hf_repo_id(hf_repo_id)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.hf_private = hf_private
        self.hf_episodes = hf_episodes

    def publish(
        self,
        *,
        job_dir: Path,
        job_name: str,
        model_name: Optional[str],
        benchmark_name: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Upload traces from ``job_dir`` to HuggingFace. Returns the HF URL."""
        if not self.hf_token:
            print("[hf-sink] No HF token provided; skipping upload.")
            return None

        job_path = Path(job_dir)
        if not job_path.exists():
            print(f"[hf-sink] Job directory {job_path} does not exist; skipping.")
            return None

        try:
            from harbor.utils.traces_utils import export_traces
        except ImportError as exc:
            print(f"[hf-sink] harbor.utils.traces_utils not importable ({exc}); skipping.")
            return None

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            print(f"[hf-sink] huggingface_hub not installed ({exc}); skipping.")
            return None

        print(f"[hf-sink] Exporting traces from {job_path} -> {self.hf_repo_id}")

        # export_traces builds the ShareGPT-format trace dataset from the
        # Harbor job directory and uploads it to HuggingFace.
        hf_url = export_traces(
            job_dir=str(job_path),
            hf_repo_id=self.hf_repo_id,
            token=self.hf_token,
            private=self.hf_private,
            episodes=self.hf_episodes,
        )
        print(f"[hf-sink] Upload complete: {hf_url}")
        return hf_url


__all__ = ["HFResultSink", "derive_benchmark_repo", "sanitize_hf_repo_id"]
