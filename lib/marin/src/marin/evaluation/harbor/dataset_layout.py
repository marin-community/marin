# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared checks for materialized Harbor task directories."""

from pathlib import Path


def dataset_task_count(path: Path) -> int:
    """Return the positive number of task directories under ``path``."""
    count = sum(entry.is_dir() and (entry / "task.toml").is_file() for entry in path.iterdir())
    if count <= 0:
        raise ValueError("Harbor dataset must contain at least one task")
    return count
