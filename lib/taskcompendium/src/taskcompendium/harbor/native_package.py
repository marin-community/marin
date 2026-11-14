# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize original native package files without executing or rewriting them."""

import os
from pathlib import Path

from taskcompendium.models import NoGrader, TaskSpec
from taskcompendium.runtime.staging import resource_files


def write_native_task(task: TaskSpec, directory: Path) -> None:
    """Write a fresh original Harbor package, retaining file bytes, modes and timestamps."""
    if not isinstance(task.grader, NoGrader) or "harbor" not in task.tags:
        raise ValueError("Native package materialization requires an original Harbor task")
    directory.mkdir(parents=True, exist_ok=False)
    for file in resource_files((*task.resources.verifier, *task.resources.oracle), ""):
        target = directory / file.path.removeprefix("/")
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Native source resource leaves its package: {file.path}")
        if target.exists() and target.read_bytes() != file.data:
            raise ValueError(f"Native source roles disagree on resource bytes: {file.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.data)
        target.chmod(file.mode)
        if file.mtime_ns is not None:
            os.utime(target, ns=(file.mtime_ns, file.mtime_ns))
