# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the pinned TaskTrove Clean archive format without extracting it."""

import io
import tarfile
import tomllib
from dataclasses import dataclass

from taskcompendium.models import Source

IMPORTER_REVISION = "taskcompendium-tasktrove-v0.1"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 1_024


@dataclass(frozen=True)
class TaskArchive:
    """A bounded, release-pinned TaskTrove archive."""

    tasktrove_source: str
    tasktrove_path: str
    release_uri: str
    release_revision: str
    files: dict[str, bytes]

    @property
    def source(self) -> Source:
        return Source(
            dataset=self.release_uri,
            revision=self.release_revision,
            row=f"{self.tasktrove_source}:{self.tasktrove_path}",
            importer_revision=IMPORTER_REVISION,
        )


def read_archive(
    data: bytes,
    tasktrove_source: str,
    tasktrove_path: str,
    release_uri: str,
    release_revision: str,
) -> TaskArchive:
    """Read regular archive members without extracting them to the host filesystem."""
    if not all((tasktrove_source, tasktrove_path, release_uri, release_revision)) or len(data) > MAX_ARCHIVE_BYTES:
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
        metadata = tomllib.loads(files["task.toml"].decode())["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Task archive metadata is not a table")
        if metadata.get("tasktrove_source") != tasktrove_source or metadata.get("tasktrove_path") != tasktrove_path:
            raise ValueError("Task archive does not match its declared source identity")
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove archive metadata: {error}") from error
    return TaskArchive(tasktrove_source, tasktrove_path, release_uri, release_revision, files)
