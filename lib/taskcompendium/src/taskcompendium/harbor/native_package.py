# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize original native package files without executing or rewriting them."""

import os
import tomllib
from pathlib import Path

from taskcompendium.models import NoGrader, TaskSpec
from taskcompendium.runtime.docker_build import materialize_build_directories
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.staging import resource_files


def validate_native_package(task: TaskSpec) -> None:
    """Require the complete original Harbor task-directory layout inside the TaskSpec."""
    files = {resource.path: resource for resource in task.resources.verifier}
    required = {"task.toml", "instruction.md", "tests/test.sh"}
    if not isinstance(task.grader, NoGrader) or "harbor" not in task.tags or not required <= files.keys():
        raise ValueError(f"Incomplete native Harbor package: {task.id}")
    environment = tomllib.loads(resource_bytes(files["task.toml"]).decode()).get("environment", {})
    if not environment.get("docker_image") and "environment/Dockerfile" not in files:
        raise ValueError(f"Harbor package declares no environment: {task.id}")


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
    build = task.environment_requirements.docker_build
    if build is not None:
        materialize_build_directories(build, directory / "environment")
