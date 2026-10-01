# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a TaskTrove Clean task archive without extracting it.

Each task is a gzip-compressed tarball containing ``task.toml`` with a
``[metadata]`` source identity, ``instruction.md``, and ``tests/verifier.toml``;
other members can supply environment or test files. The reader retains regular
members as bytes and checks their paths, count, and total size before import.
"""

import hashlib
import io
import json
import tarfile
import tomllib
from dataclasses import dataclass

from taskcompendium.importers.tasktrove.models import TaskArchive

TASK_MANIFEST = "task.toml"
METADATA_TABLE = "metadata"
INSTRUCTION_FILE = "instruction.md"
VERIFIER_FILE = "tests/verifier.toml"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 1_024


@dataclass(frozen=True)
class TaskTroveMetadata:
    """Typed metadata shared by the supported archive importers."""

    source: str
    family: str
    converter: str
    mode: str
    tags: tuple[str, ...]


def import_metadata(archive: TaskArchive) -> TaskTroveMetadata:
    """Read and validate the fields shared by TaskTrove Clean importers."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a table")
        source, family, converter, mode = (metadata[key] for key in ("tasktrove_source", "family", "converter", "mode"))
        tags = metadata.get("tags", [])
        if not all(isinstance(value, str) for value in (source, family, converter, mode)):
            raise ValueError("source, family, converter, and mode must be strings")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("tags must be an ordered list of strings")
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove importer metadata: {error}") from error
    return TaskTroveMetadata(source, family, converter, mode, tuple(tags))


def task_id(archive: TaskArchive) -> str:
    """Return a stable task identifier from the pinned source identity."""
    identity = json.dumps((archive.source.dataset, archive.source.revision, archive.source.row), separators=(",", ":"))
    return f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}"


def read_archive(
    data: bytes,
    upstream_subset: str,
    archive_path: str,
    release_uri: str,
    release_revision: str,
) -> TaskArchive:
    """Read an archive, checking its subset and path against its manifest.

    The caller supplies release provenance; the archive cannot verify it.
    """
    if not all((upstream_subset, archive_path, release_uri, release_revision)) or len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Task archive has an invalid identity or exceeds the input limit")
    files: dict[str, bytes] = {}
    size = 0
    members = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive:
            members += 1
            if members > MAX_ARCHIVE_MEMBERS:
                raise ValueError("Task archive exceeds member limit")
            name = member.name.removeprefix("./")
            if member.isdir():
                continue
            if not member.isfile() or not name or name.startswith("/") or ".." in name.split("/") or name in files:
                raise ValueError(f"Unsupported archive member: {member.name}")
            size += member.size
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError("Task archive exceeds expanded size limit")
            stream = archive.extractfile(member)
            assert stream is not None
            files[name] = stream.read()
    try:
        metadata = tomllib.loads(files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if not isinstance(metadata, dict):
            raise ValueError("Task archive metadata is not a table")
        if metadata.get("tasktrove_source") != upstream_subset or metadata.get("tasktrove_path") != archive_path:
            raise ValueError("Task archive does not match its declared source identity")
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove archive metadata: {error}") from error
    return TaskArchive(upstream_subset, archive_path, release_uri, release_revision, files)
