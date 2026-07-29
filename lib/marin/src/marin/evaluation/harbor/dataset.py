# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve Marin-owned Harbor dataset sources."""

from pathlib import Path

from huggingface_hub import snapshot_download

from marin.evaluation.harbor.driver_config import HarborDatasetKind, ValidatedHarborConfig


def validate_harbor_dataset_source(config: ValidatedHarborConfig) -> None:
    """Fail before submission when a relative local source is not a directory."""
    if config.dataset_kind != HarborDatasetKind.LOCAL:
        return
    path = config.config_dir / config.dataset_selector
    if not path.exists():
        raise ValueError(f"Harbor local dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Harbor local dataset path must be a directory: {path}")


def materialize_harbor_dataset(
    config: ValidatedHarborConfig,
    workdir: Path,
    *,
    hf_token: str | None,
) -> Path | None:
    """Return a local Harbor task directory, or ``None`` for a registry-backed dataset.

    Hugging Face sources are downloaded on the evaluation worker. Relative local
    sources resolve against the policy file. Harbor registry selectors remain in
    the opaque policy and need no materialized path.
    """
    if config.dataset_kind == HarborDatasetKind.HUGGING_FACE:
        local_dir = workdir / "hf_dataset"
        local_dir.mkdir(parents=True, exist_ok=True)
        root = Path(
            snapshot_download(
                repo_id=config.dataset_selector,
                repo_type="dataset",
                revision=config.dataset_revision,
                local_dir=str(local_dir),
                cache_dir=str(workdir / "hf_cache"),
                token=hf_token or False,
            )
        )
        gitattributes = root / ".gitattributes"
        if gitattributes.exists():
            gitattributes.unlink()
        return root

    if config.dataset_kind == HarborDatasetKind.HARBOR_REGISTRY:
        return None

    dataset_path = config.config_dir / config.dataset_selector
    if not dataset_path.is_dir():
        raise ValueError(f"Harbor local dataset path must be a directory: {dataset_path}")
    return dataset_path
