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
import tarfile
import tomllib
import zlib

from taskcompendium.importers.tasktrove.models import TaskArchive

TASK_MANIFEST = "task.toml"
METADATA_TABLE = "metadata"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 1_024


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
    try:
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
    except (tarfile.TarError, EOFError, OSError, zlib.error) as error:
        raise ValueError(f"Invalid TaskTrove archive container: {error}") from error
    try:
        metadata = tomllib.loads(files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if not isinstance(metadata, dict):
            raise ValueError("Task archive metadata is not a table")
        if metadata.get("tasktrove_source") != upstream_subset or metadata.get("tasktrove_path") != archive_path:
            raise ValueError("Task archive does not match its declared source identity")
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove archive metadata: {error}") from error
    return TaskArchive(
        upstream_subset,
        archive_path,
        release_uri,
        release_revision,
        hashlib.sha256(data).hexdigest(),
        files,
    )
