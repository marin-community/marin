# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build inline task resources."""

import base64
import os
from pathlib import Path

from taskcompendium.models import DOCKERFILE, DockerBuildContext, InlineFile, TaskResource


def inline_resource(path: str, data: bytes) -> TaskResource:
    """Build an inline resource at a relative workspace path."""
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(data).decode("ascii")))


def resource_bytes(resource: TaskResource) -> bytes:
    """Decode a validated inline resource."""
    return base64.b64decode(resource.source.content_base64, validate=True)


def write_build_context(context: DockerBuildContext, directory: Path) -> Path:
    """Write a build context's files under ``directory`` with their modes and timestamps; return its Dockerfile."""
    for resource in context.files:
        target = directory / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))
        target.chmod(0o644 if resource.mode is None else int(resource.mode, 8))
        if resource.mtime_ns is not None:
            os.utime(target, ns=(resource.mtime_ns, resource.mtime_ns))
    return directory / DOCKERFILE
