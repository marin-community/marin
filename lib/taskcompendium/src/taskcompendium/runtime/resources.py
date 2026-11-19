# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inline resources used by the optional episode runtime."""

import base64
import os
from pathlib import Path

from shellbox.image import DockerfileSource

from taskcompendium.models import DockerBuildContext, InlineFile, TaskResource


def inline_resource(path: str, data: bytes) -> TaskResource:
    """Build an inline resource at a relative workspace path."""
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(data).decode("ascii")))


def resource_bytes(resource: TaskResource) -> bytes:
    """Decode a validated inline resource."""
    return base64.b64decode(resource.source.content_base64, validate=True)


def materialize_build_context(recipe: DockerBuildContext, directory: Path) -> DockerfileSource:
    """Write a declared build context in a new directory owned by the caller."""
    directory.mkdir(parents=True, exist_ok=False)
    for resource in recipe.files:
        target = directory / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))
        target.chmod(int(resource.mode or "644", 8))
        if resource.mtime_ns is not None:
            os.utime(target, ns=(resource.mtime_ns, resource.mtime_ns))
    return DockerfileSource(directory, directory / "Dockerfile")
