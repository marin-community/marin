# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archives for task resources, submissions and downloaded grading artifacts."""

import io
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from taskcompendium.models import TaskResource
from taskcompendium.runtime.resources import resource_bytes


@dataclass(frozen=True)
class StagedFile:
    path: str
    data: bytes
    mode: int = 0o644
    mtime_ns: int | None = None


def resource_files(resources: Sequence[TaskResource], root: str) -> list[StagedFile]:
    return [
        StagedFile(
            f"{root}/{resource.path}",
            resource_bytes(resource),
            0o644 if resource.mode is None else int(resource.mode, 8),
            resource.mtime_ns,
        )
        for resource in resources
    ]


def write_archive(archive_path: Path, files: list[StagedFile], artifacts: list[tuple[Path, str]]) -> None:
    """Write files and artifacts in install order, retaining modes and timestamps."""
    with tarfile.open(archive_path, "w") as archive:
        for file in files:
            member = tarfile.TarInfo(file.path.removeprefix("/"))
            member.size = len(file.data)
            member.mode = file.mode
            if file.mtime_ns is not None:
                seconds, nanos = divmod(file.mtime_ns, 1_000_000_000)
                member.mtime = seconds
                member.pax_headers = {"mtime": f"{seconds}.{nanos:09d}"}
            archive.addfile(member, io.BytesIO(file.data))
        for local, target in artifacts:
            for path in [local, *sorted(local.rglob("*"))] if local.is_dir() else [local]:
                name = target if path == local else f"{target}/{path.relative_to(local).as_posix()}"
                info = archive.gettarinfo(str(path), name.removeprefix("/"))
                info.mtime, info.uid, info.gid, info.uname, info.gname = int(info.mtime), 0, 0, "", ""
                if info.isfile():
                    with path.open("rb") as data:
                        archive.addfile(info, data)
                else:
                    archive.addfile(info)
