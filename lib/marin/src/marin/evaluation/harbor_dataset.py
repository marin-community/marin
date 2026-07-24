# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve Harbor datasets that are stored as local or Hugging Face task directories."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def materialize_harbor_dataset(dataset: str, revision: str, workdir: Path) -> Path | None:
    """Return a local Harbor task directory, or ``None`` for a registry-backed dataset.

    ``hf://org/repo`` and ``hf:org/repo`` identify Hugging Face dataset repositories whose root
    contains Harbor task directories. ``revision`` is the Hugging Face revision; the legacy ``hf``
    sentinel selects the repository's default revision. An existing local directory is returned
    unchanged. Every other value remains a Harbor registry name.
    """
    dataset_path = Path(dataset).expanduser()
    is_hugging_face = (
        dataset.startswith("hf://")
        or dataset.startswith("hf:")
        or (revision == "hf" and "/" in dataset and not dataset_path.exists())
    )
    if is_hugging_face:
        if dataset.startswith("hf://"):
            repo_id = dataset.removeprefix("hf://")
        elif dataset.startswith("hf:"):
            repo_id = dataset.removeprefix("hf:")
        else:
            repo_id = dataset
        local_dir = workdir / "hf_dataset"
        local_dir.mkdir(parents=True, exist_ok=True)
        root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=None if revision == "hf" else revision,
                local_dir=str(local_dir),
                cache_dir=str(workdir / "hf_cache"),
                token=os.environ.get("HF_TOKEN", False),
            )
        )
        gitattributes = root / ".gitattributes"
        if gitattributes.exists():
            gitattributes.unlink()
        return root

    if not dataset_path.exists():
        return None
    if not dataset_path.is_dir():
        raise ValueError(f"Harbor dataset path must be a directory, got: {dataset_path}")
    return dataset_path
