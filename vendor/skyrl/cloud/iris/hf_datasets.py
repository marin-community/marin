"""HuggingFace dataset helpers for the Iris RL launcher.

Detects the on-disk task-dataset format (raw task dirs versus parquet with
``task_binary``) and downloads dataset snapshots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import huggingface_hub
from huggingface_hub import snapshot_download

from marinskyrl.hugging_face_retry import call_with_hugging_face_retry, load_hugging_face_with_retry
from marinskyrl.resource_locator import HFDatasetSelector, parse_hf_dataset_selector


def is_raw_tasks_directory(snapshot_dir) -> bool:
    """Check if a directory contains raw task folders (not parquet with task_binary).

    Raw task directories have subdirectories with ``instruction.md`` files, rather
    than parquet files with ``task_binary`` columns that need extraction. Used to
    auto-detect the downloaded HuggingFace dataset format.
    """
    snapshot_dir = Path(snapshot_dir)

    parquet_files = list(snapshot_dir.rglob("*.parquet"))
    if parquet_files:
        try:
            import pyarrow.parquet as pq

            for pf in parquet_files[:1]:
                table = pq.read_table(pf)
                if "task_binary" in table.column_names:
                    return False  # needs extraction
        except Exception:
            pass

    instruction_files = list(snapshot_dir.rglob("instruction.md"))
    if instruction_files:
        return True

    return False


def resolve_hf_dataset_selector(value: str) -> HFDatasetSelector:
    """Resolve a dataset selector's revision to an immutable Hub commit."""
    selector = parse_hf_dataset_selector(value)
    if selector is None:
        raise ValueError(f"Invalid Hugging Face dataset selector: {value!r}")
    info = call_with_hugging_face_retry(
        lambda: huggingface_hub.HfApi().dataset_info(selector.repo_id, revision=selector.revision),
        operation=f"resolve Hugging Face dataset {selector.repo_id}@{selector.revision or 'main'}",
    )
    return HFDatasetSelector(selector.repo_id, info.sha, selector.subdir)


def download_hf_dataset(selector_value: str, revision: Optional[str] = None) -> str:
    """Download a dataset selector and return its selected local directory."""
    selector = parse_hf_dataset_selector(selector_value)
    if selector is None:
        raise ValueError(f"Invalid Hugging Face dataset selector: {selector_value!r}")
    if revision is not None and selector.revision is not None and revision != selector.revision:
        raise ValueError("revision conflicts with the revision embedded in the dataset selector")
    cache_path = os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub"))
    snapshot = load_hugging_face_with_retry(
        lambda: snapshot_download(
            repo_id=selector.repo_id,
            cache_dir=str(cache_path),
            revision=revision or selector.revision,
            repo_type="dataset",
            allow_patterns=[f"{selector.subdir}/**"] if selector.subdir else None,
        ),
        resource_id=selector.repo_id,
        resource_kind="dataset snapshot",
    )
    selected = Path(snapshot) / selector.subdir if selector.subdir else Path(snapshot)
    if not selected.is_dir():
        raise FileNotFoundError(f"Dataset selector {selector_value!r} did not resolve a subdirectory")
    return str(selected)
