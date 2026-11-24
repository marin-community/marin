# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build inline task resources."""

import base64
import hashlib
from typing import Any

from taskcompendium.models import (
    EnvironmentRequirements,
    InlineFile,
    ScriptGrader,
    TaskResource,
    TaskSpec,
    VerifyitGrader,
)


def inline_resource(path: str, data: bytes) -> TaskResource:
    """Build an inline resource at a relative workspace path."""
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(data).decode("ascii")))


def resource_bytes(resource: TaskResource) -> bytes:
    """Decode a validated inline resource."""
    return base64.b64decode(resource.source.content_base64, validate=True)


def build_environments(task: TaskSpec) -> list[tuple[str, EnvironmentRequirements]]:
    """The task's environments that can carry unresolved build inputs, by the role their files take."""
    environments = [("actor_build", task.environment_requirements)]
    if isinstance(task.grader, ScriptGrader | VerifyitGrader) and task.grader.environment is not None:
        environments.append(("grader_build", task.grader.environment))
    return environments


def resource_inventory(task: TaskSpec) -> list[tuple[str, TaskResource]]:
    """Every file the task carries, by role: the four resource groups, then the actor and grader build inputs."""
    resources = [
        (role, resource)
        for role, group in (
            ("all", task.resources.all),
            ("worker", task.resources.worker),
            ("oracle", task.resources.oracle),
            ("verifier", task.resources.verifier),
        )
        for resource in group
    ]
    for role, environment in build_environments(task):
        if environment.docker_build is not None:
            resources.extend((role, resource) for resource in environment.docker_build.files)
    return resources


def resource_signature(role: str, resource: TaskResource, data: bytes) -> dict[str, Any]:
    """A file's role, path, metadata and bytes (``data``, already decoded), as the task review signs them."""
    return {
        "role": role,
        **resource.model_dump(mode="json", exclude={"source"}),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }
