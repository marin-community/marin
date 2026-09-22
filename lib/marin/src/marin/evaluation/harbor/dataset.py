# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve Marin workspace paths used by Harbor datasets."""

from pathlib import Path

from rigging.config_discovery import find_project_root

from marin.evaluation.harbor.driver_config import HarborDatasetKind, ValidatedHarborConfig


def _local_dataset_path(config: ValidatedHarborConfig) -> Path:
    workspace_root = find_project_root()
    if workspace_root is None:
        raise ValueError("Harbor local datasets require a Marin workspace")
    if config.workspace_dataset_path is None:
        raise ValueError("Harbor local dataset metadata has no workspace-relative path")
    return workspace_root / config.workspace_dataset_path


def validate_harbor_dataset_source(config: ValidatedHarborConfig) -> None:
    """Fail before submission when a relative local source is not a directory."""
    if config.dataset_kind != HarborDatasetKind.LOCAL:
        return
    path = _local_dataset_path(config)
    if not path.exists():
        raise ValueError(f"Harbor local dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Harbor local dataset path must be a directory: {path}")


def local_harbor_dataset_path(config: ValidatedHarborConfig) -> Path | None:
    """Return a workspace task path for local datasets; Harbor resolves other sources."""
    if config.dataset_kind != HarborDatasetKind.LOCAL:
        return None

    dataset_path = _local_dataset_path(config)
    if not dataset_path.is_dir():
        raise ValueError(f"Harbor local dataset path must be a directory: {dataset_path}")
    return dataset_path
