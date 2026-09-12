# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize only explicitly visible resources, checking immutable references."""

import hashlib
from pathlib import Path
from typing import BinaryIO, cast

import fsspec

from taskcompendium.models import Embedded, Resource, ResourceRole, TaskSpecification

MAX_RESOURCE_BYTES = 64 * 1024 * 1024


def resource_bytes(resource: Resource, max_bytes: int = MAX_RESOURCE_BYTES) -> bytes:
    if isinstance(resource.content, Embedded):
        data = resource.content.data
    else:
        with fsspec.open(resource.content.uri, "rb") as stream:
            data = cast(BinaryIO, stream).read(max_bytes + 1)
        if hashlib.sha256(data).hexdigest() != resource.content.sha256:
            raise ValueError(f"Resource digest mismatch: {resource.path}")
    if len(data) > max_bytes:
        raise ValueError(f"Resource exceeds materialization limit: {resource.path}")
    return data


def contained_path(root: Path, relative: str) -> Path:
    """Resolve a path and reject symlink escapes from an owned workspace."""
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes workspace: {relative}")
    return candidate


def materialize(specification: TaskSpecification, role: ResourceRole, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for resource in specification.resources:
        if role not in resource.roles:
            continue
        target = contained_path(root, resource.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))
        target.chmod(0o700 if resource.executable else 0o600)
