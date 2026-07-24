# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve Harbor datasets that are stored as local or Hugging Face task directories."""

from pathlib import Path

from huggingface_hub import snapshot_download

_HF_URL_PREFIX = "hf://"


def materialize_harbor_dataset(
    dataset: str,
    revision: str,
    workdir: Path,
    *,
    hf_token: str | None,
) -> Path | None:
    """Return a local Harbor task directory, or ``None`` for a registry-backed dataset.

    ``hf://org/repo`` identifies a Hugging Face dataset repository whose root contains Harbor task
    directories. An existing local directory is returned unchanged. Every other value remains a
    Harbor registry name.
    """
    dataset_path = Path(dataset).expanduser()
    if dataset.startswith(_HF_URL_PREFIX):
        repo_id = dataset.removeprefix(_HF_URL_PREFIX)
        local_dir = workdir / "hf_dataset"
        local_dir.mkdir(parents=True, exist_ok=True)
        root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=str(local_dir),
                cache_dir=str(workdir / "hf_cache"),
                token=hf_token or False,
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
