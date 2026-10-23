# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inline resources used by the optional episode runtime."""

import base64

from taskcompendium.models import InlineFile, TaskResource


def inline_resource(path: str, data: bytes) -> TaskResource:
    """Build an inline resource at a relative workspace path."""
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(data).decode("ascii")))


def resource_bytes(resource: TaskResource) -> bytes:
    """Decode a validated inline resource."""
    return base64.b64decode(resource.source.content_base64, validate=True)
