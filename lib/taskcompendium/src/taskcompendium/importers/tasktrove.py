# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the pinned TaskTrove Clean archive format without extracting it."""

import io
import tarfile
from dataclasses import dataclass

from taskcompendium.models import Source

RELEASE = "2026.09.10.9"
RELEASE_ROOT = f"s3://marin-us-east-02a/marin/tasktrove/clean/{RELEASE}"
IMPORTER_REVISION = "taskcompendium-tasktrove-v0.1"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class TaskArchive:
    """A bounded, release-pinned TaskTrove archive."""

    row: str
    files: dict[str, bytes]

    @property
    def source(self) -> Source:
        return Source(RELEASE_ROOT, RELEASE, self.row, IMPORTER_REVISION)


def read_archive(data: bytes, row: str) -> TaskArchive:
    """Read regular archive members without extracting them to the host filesystem."""
    if not row or len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Task archive has an invalid row or exceeds the input limit")
    files: dict[str, bytes] = {}
    size = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive:
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
    return TaskArchive(row, files)
