# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate original native Harbor packages without executing or rewriting them."""

import tomllib

from taskcompendium.models import NoGrader, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

TASK_CONFIG = "task.toml"


def validate_native_package(task: TaskSpec) -> None:
    """Require the Harbor package files and a declared environment among the verifier resources."""
    files = {resource.path: resource for resource in task.resources.verifier}
    required = {TASK_CONFIG, "instruction.md", "tests/test.sh"}
    if not isinstance(task.grader, NoGrader) or "harbor" not in task.tags or not required <= files.keys():
        raise ValueError(f"Incomplete native Harbor package: {task.id}")
    environment = tomllib.loads(resource_bytes(files[TASK_CONFIG]).decode()).get("environment", {})
    if not environment.get("docker_image") and "environment/Dockerfile" not in files:
        raise ValueError(f"Harbor package declares no environment: {task.id}")
