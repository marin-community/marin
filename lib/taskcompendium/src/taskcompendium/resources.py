# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private task resources and their safe materialization."""

import base64
import binascii
import hashlib
import os
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, model_validator

MAX_RESOURCE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 64 * 1024 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class ResourceVisibility(StrEnum):
    """Who may inspect a task resource during a trial."""

    AGENT = "agent"
    VERIFIER = "verifier"
    ORACLE = "oracle"


class ResourceReference(BaseModel):
    """An opaque locator resolved by a trusted caller and pinned by digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    locator: str
    sha256: str

    @model_validator(mode="after")
    def validate_reference(self) -> "ResourceReference":
        if not self.locator or not SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("Resource references require a locator and lowercase SHA256 digest")
        return self


class TaskResource(BaseModel):
    """One file in the private task specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    visibility: ResourceVisibility
    executable: bool = False
    content: str | None = None
    content_base64: str | None = None
    reference: ResourceReference | None = None

    @model_validator(mode="after")
    def validate_resource(self) -> "TaskResource":
        validate_resource_path(self.path)
        if sum(value is not None for value in (self.content, self.content_base64, self.reference)) != 1:
            raise ValueError("A resource requires exactly one inline content form or reference")
        if self.content_base64 is not None:
            decode_base64_content(self.content_base64)
        return self


@dataclass(frozen=True)
class ResolvedResource:
    """A validated file payload ready to install in one environment."""

    path: PurePosixPath
    visibility: ResourceVisibility
    executable: bool
    content: bytes


ResourceResolver = Callable[[ResourceReference], bytes]


def decode_base64_content(value: str) -> bytes:
    """Decode the canonical JSON representation of inline binary content."""
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Invalid base64 resource content") from error
    if base64.b64encode(payload).decode("ascii") != value:
        raise ValueError("Base64 resource content must be canonical")
    return payload


def validate_resource_path(path: str) -> PurePosixPath:
    """Require a portable normalized relative path with no traversal."""
    if not path or "\\" in path or "\x00" in path or unicodedata.normalize("NFC", path) != path:
        raise ValueError(f"Invalid resource path: {path!r}")
    if path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"Resource path must be normalized and relative: {path!r}")
    if ":" in path.split("/")[0]:
        raise ValueError(f"Resource path must be normalized and relative: {path!r}")
    return PurePosixPath(path)


def validate_resource_paths(resources: Iterable[TaskResource]) -> None:
    """Reject file collisions, including case and ancestor collisions."""
    files: set[str] = set()
    for resource in resources:
        parts = validate_resource_path(resource.path).parts
        folded = tuple(part.casefold() for part in parts)
        key = "/".join(folded)
        if key in files or any("/".join(folded[:index]) in files for index in range(1, len(folded))):
            raise ValueError(f"Resource path collision: {resource.path}")
        if any(existing.startswith(f"{key}/") for existing in files):
            raise ValueError(f"Resource path collision: {resource.path}")
        files.add(key)


def validate_resources(
    resources: Iterable[TaskResource],
    *,
    trusted_resolver: ResourceResolver | None = None,
) -> tuple[ResolvedResource, ...]:
    """Resolve and check all bytes before a trial or materialization begins.

    The resolver is supplied by the caller. This module never fetches a URL.
    """
    resources = tuple(resources)
    validate_resource_paths(resources)
    resolved: list[ResolvedResource] = []
    total_bytes = 0
    for resource in resources:
        if resource.content is not None:
            payload = resource.content.encode("utf-8")
        elif resource.content_base64 is not None:
            payload = decode_base64_content(resource.content_base64)
        else:
            if trusted_resolver is None or resource.reference is None:
                raise ValueError(f"Resource {resource.path!r} requires a trusted resolver")
            payload = trusted_resolver(resource.reference)
            if not isinstance(payload, bytes):
                raise TypeError("A trusted resource resolver must return bytes")
            if hashlib.sha256(payload).hexdigest() != resource.reference.sha256:
                raise ValueError(f"Resource digest mismatch: {resource.path}")
        total_bytes += len(payload)
        if len(payload) > MAX_RESOURCE_BYTES or total_bytes > MAX_TOTAL_RESOURCE_BYTES:
            raise ValueError("Task resources exceed size limits")
        resolved.append(
            ResolvedResource(validate_resource_path(resource.path), resource.visibility, resource.executable, payload)
        )
    return tuple(resolved)


def materialize_resources(
    resources: Iterable[TaskResource],
    destination: Path,
    *,
    visibility: ResourceVisibility | frozenset[ResourceVisibility],
    trusted_resolver: ResourceResolver | None = None,
) -> None:
    """Install selected files after validating every task resource.

    The destination is a caller-owned root. Existing files and symlinks below
    that root are rejected rather than followed or overwritten.
    """
    resolved = validate_resources(resources, trusted_resolver=trusted_resolver)
    allowed = frozenset({visibility}) if isinstance(visibility, ResourceVisibility) else visibility
    selected = tuple(resource for resource in resolved if resource.visibility in allowed)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError(f"Unsafe resource destination: {destination}")
    for resource in selected:
        current = destination
        for part in resource.path.parts[:-1]:
            current = current / part
            if current.is_symlink() or (current.exists() and not current.is_dir()):
                raise ValueError(f"Unsafe resource parent: {current}")
        target = destination.joinpath(*resource.path.parts)
        if target.is_symlink() or target.exists():
            raise ValueError(f"Resource target already exists: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for resource in selected:
        target = destination.joinpath(*resource.path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(resource.content)
        target.chmod(0o755 if resource.executable else 0o644)
