# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private configuration and resource staging for isolated script verification."""

import base64
import binascii
import hashlib
import math
import os
import re
import unicodedata
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from taskcompendium.models import VerifierKind, VerifierSpec

SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
PINNED_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
PROTOCOL_VERSION = 1
MAX_RESOURCE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 64 * 1024 * 1024

type ResourceResolver = Callable[[str], bytes]


def _relative_resource_path(value: str) -> str:
    """Accept one unambiguous path beneath the verifier's private /tests mount."""
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or ":" in value.split("/")[0]
        or unicodedata.normalize("NFC", value) != value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("Private resource path must be relative to /tests without traversal")
    return value


def _decode_embedded(value: str) -> bytes:
    try:
        content = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Invalid base64 private resource") from error
    if base64.b64encode(content).decode("ascii") != value:
        raise ValueError("Private resource base64 must be canonical")
    return content


class PrivateResource(BaseModel):
    """A private file pinned by content digest and one source of bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str
    sha256: str
    executable: bool = False
    embedded_base64: str | None = Field(default=None, repr=False)
    uri: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _relative_resource_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not SHA256_HEX.fullmatch(value):
            raise ValueError("Private resource requires a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> "PrivateResource":
        if (self.embedded_base64 is None) == (self.uri is None):
            raise ValueError("Private resource requires exactly one of embedded_base64 or uri")
        if self.uri is not None and not self.uri.strip():
            raise ValueError("Private resource URI must be nonempty")
        if self.embedded_base64 is not None:
            content = _decode_embedded(self.embedded_base64)
            if hashlib.sha256(content).hexdigest() != self.sha256:
                raise ValueError("Private resource digest does not match embedded bytes")
        return self


class NetworkPolicy(StrEnum):
    """Network access granted to the isolated verifier runtime."""

    DISABLED = "disabled"
    ENABLED = "enabled"


def embedded_resource(path: str, content: bytes, *, executable: bool = False) -> PrivateResource:
    """Pin embedded private file content by its SHA256 digest."""
    return PrivateResource(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        embedded_base64=base64.b64encode(content).decode("ascii"),
        executable=executable,
    )


class ScriptVerifier(BaseModel):
    """Source-independent, pinned inputs for one isolated script grader."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    entrypoint: str
    args: tuple[str, ...] = ()
    timeout_seconds: float
    runtime_image: str
    resources: tuple[PrivateResource, ...]
    network_policy: NetworkPolicy = NetworkPolicy.DISABLED
    protocol_version: int = PROTOCOL_VERSION

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        return _relative_resource_path(value)

    @field_validator("runtime_image")
    @classmethod
    def validate_runtime_image(cls, value: str) -> str:
        if not PINNED_IMAGE.fullmatch(value):
            raise ValueError("Verifier runtime image must be pinned by SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "ScriptVerifier":
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Script timeout must be finite and positive")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported script result protocol: {self.protocol_version}")
        _validate_resource_paths(self.resources)
        entrypoint = next((resource for resource in self.resources if resource.path == self.entrypoint), None)
        if entrypoint is None or not entrypoint.executable:
            raise ValueError("Script entrypoint must be an executable private resource")
        return self


def script_verifier(config: ScriptVerifier) -> VerifierSpec:
    """Serialize a script contract for a private TaskSpec verifier."""
    return VerifierSpec(kind=VerifierKind.SCRIPT, parameters_json=config.model_dump_json())


def _validate_resource_paths(resources: Sequence[PrivateResource]) -> None:
    paths: set[str] = set()
    for resource in resources:
        folded = resource.path.casefold()
        if folded in paths or any(folded.startswith(f"{path}/") or path.startswith(f"{folded}/") for path in paths):
            raise ValueError(f"Private resource path collision: {resource.path}")
        paths.add(folded)


def materialize_private_resources(
    resources: Sequence[PrivateResource], destination: Path, resolve_uri: ResourceResolver | None = None
) -> dict[str, Path]:
    """Verify and write private files beneath a verifier-only directory.

    URI retrieval belongs to the caller, so this function does not fetch from
    the network or run the supplied executable in the host process.
    """
    if destination.is_symlink():
        raise ValueError("Private resource destination cannot be a symlink")
    content_by_path: dict[str, bytes] = {}
    executable_paths: set[str] = set()
    _validate_resource_paths(resources)
    total_bytes = 0
    for resource in resources:
        path = _relative_resource_path(resource.path)
        if resource.embedded_base64 is not None:
            content = _decode_embedded(resource.embedded_base64)
        else:
            if resolve_uri is None or resource.uri is None:
                raise ValueError(f"No URI resolver for private resource: {path}")
            content = resolve_uri(resource.uri)
            if not isinstance(content, bytes):
                raise TypeError("Private resource resolver must return bytes")
        if hashlib.sha256(content).hexdigest() != resource.sha256:
            raise ValueError(f"Private resource digest mismatch: {path}")
        total_bytes += len(content)
        if len(content) > MAX_RESOURCE_BYTES or total_bytes > MAX_TOTAL_RESOURCE_BYTES:
            raise ValueError("Private resources exceed size limits")
        content_by_path[path] = content
        if resource.executable:
            executable_paths.add(path)

    for path in content_by_path:
        target = destination / path
        current = destination
        for part in Path(path).parts[:-1]:
            current = current / part
            if current.is_symlink() or (current.exists() and not current.is_dir()):
                raise ValueError(f"Unsafe private resource parent: {current}")
        if target.is_symlink() or target.exists():
            raise ValueError(f"Private resource target already exists: {target}")

    destination.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for path, content in content_by_path.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
        target.chmod(0o700 if path in executable_paths else 0o600)
        staged[path] = target
    return staged
