# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable curator-packet schema for hydrated open-ended task inputs.

This module deliberately depends only on Pydantic and the Python standard
library so source-specific extractors can use it without installing Taskforge.
Taskforge-specific loading lives in ``open_ended_packet``.
"""

import hashlib
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKET_FORMAT = "open-ended-task-packet/1"
MANIFEST_PATH = "manifest.json"


class OpenEndedReleaseStatus(StrEnum):
    """Whether a curator-reviewed bundle may leave local development."""

    UNREVIEWED_DO_NOT_PUBLISH = "unreviewed_do_not_publish"
    INTERNAL_ONLY = "internal_only"
    APPROVED_FOR_RELEASE = "approved_for_release"


class PacketVisibility(StrEnum):
    """How the packet loader treats a declared file."""

    METADATA = "metadata"
    SOLVER = "solver"
    AUTHOR_PRIVATE = "author_private"


class PacketSource(BaseModel):
    """Portable form of TaskCompendium's pinned source provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    row: str = Field(min_length=1)
    importer_revision: str = Field(min_length=1)


class PacketFile(BaseModel):
    """One content-addressed packet file and its eventual resource path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    packet_path: str = Field(min_length=1)
    resource_path: str | None
    visibility: PacketVisibility
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_paths(self) -> Self:
        _validate_relative_path(self.packet_path, "packet_path")
        if self.visibility is PacketVisibility.METADATA:
            if self.resource_path is not None:
                raise ValueError("metadata files have no resource_path")
        elif self.resource_path is None:
            raise ValueError(f"{self.visibility} files require resource_path")
        else:
            _validate_relative_path(self.resource_path, "resource_path")
        return self


class PacketManifest(BaseModel):
    """Complete identity and file inventory of one curator packet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["open-ended-task-packet/1"] = PACKET_FORMAT
    packet_id: str = Field(min_length=1)
    bundle_id: str = Field(min_length=1)
    split_group: str = Field(min_length=1)
    brief_path: str = Field(min_length=1)
    source: PacketSource
    tags: tuple[str, ...] = ()
    release_status: OpenEndedReleaseStatus
    files: tuple[PacketFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        packet_paths = [item.packet_path for item in self.files]
        if len(set(packet_paths)) != len(packet_paths):
            raise ValueError("packet file paths must be unique")
        resource_paths = [item.resource_path for item in self.files if item.resource_path is not None]
        if len(set(resource_paths)) != len(resource_paths):
            raise ValueError("packet resource paths must be unique")
        if self.brief_path not in packet_paths:
            raise ValueError("brief_path must name a declared packet file")
        brief = next(item for item in self.files if item.packet_path == self.brief_path)
        if brief.visibility is not PacketVisibility.METADATA:
            raise ValueError("brief_path must be metadata")
        if not any(item.visibility is PacketVisibility.SOLVER for item in self.files):
            raise ValueError("a packet needs a solver resource")
        if not any(item.visibility is PacketVisibility.AUTHOR_PRIVATE for item in self.files):
            raise ValueError("a packet needs private author evidence")
        return self


class PacketFileDeclaration(BaseModel):
    """A file declaration before its size and digest are measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    packet_path: str = Field(min_length=1)
    resource_path: str | None
    visibility: PacketVisibility

    @model_validator(mode="after")
    def valid_paths(self) -> Self:
        PacketFile(
            packet_path=self.packet_path,
            resource_path=self.resource_path,
            visibility=self.visibility,
            size_bytes=0,
            sha256="0" * 64,
        )
        return self


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_files(root: Path, declarations: tuple[PacketFileDeclaration, ...]) -> tuple[PacketFile, ...]:
    """Measure declared files and reject missing, undeclared, or symbolic-link content."""
    declared = {item.packet_path: item for item in declarations}
    if len(declared) != len(declarations):
        raise ValueError("packet file declarations must have unique packet paths")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != MANIFEST_PATH
    }
    missing = sorted(set(declared) - actual)
    undeclared = sorted(actual - set(declared))
    if missing or undeclared:
        raise ValueError(f"packet files differ from declarations: missing={missing}, undeclared={undeclared}")
    files = []
    for packet_path, declaration in declared.items():
        path = root / packet_path
        if path.is_symlink():
            raise ValueError(f"packet file is a symbolic link: {packet_path}")
        files.append(
            PacketFile(
                packet_path=packet_path,
                resource_path=declaration.resource_path,
                visibility=declaration.visibility,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return tuple(sorted(files, key=lambda item: item.packet_path))


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{field} must be a normalized relative POSIX path: {value!r}")
