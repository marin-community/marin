# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free archive snapshots for comparing isolated converter processes."""

import hashlib
import io
import json
import tarfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any


@dataclass(frozen=True)
class FileSnapshot:
    type: str
    mode: int
    size: int
    sha256: str | None
    link: str
    parsed_toml: dict[str, Any] | None
    toml_error: str | None


@dataclass(frozen=True)
class TaskSnapshot:
    source: str
    path: str
    outcome: str
    detail: str
    files: dict[str, FileSnapshot]
    metadata: dict[str, Any] = field(default_factory=dict)
    legacy_static_rejection: str | None = None
    legacy_static_detail: str | None = None


def archive_snapshot(blob: bytes | None, namespace: str) -> dict[str, FileSnapshot]:
    """Hash every member's bytes and retain its path, type, permissions, and link target.

    Archive compression, order, ownership and timestamps are outside this comparison.
    Paths and file modes are retained exactly, including explicit directory entries.
    """
    if blob is None:
        return {}
    files = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        for member in archive:
            path = f"{namespace}/{member.name}"
            if path in files:
                raise ValueError(f"Duplicate archive member: {path}")
            content = None
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                content = stream.read()
            files[path] = _file_snapshot(
                member.name, content, member.type.hex(), member.mode, member.size, member.linkname
            )
    return files


def _file_snapshot(name: str, content: bytes | None, kind: str, mode: int, size: int, link: str) -> FileSnapshot:
    parsed, error = None, None
    if content is not None and name.endswith(".toml"):
        try:
            parsed = json.loads(json.dumps(tomllib.loads(content.decode()), default=json_temporal))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exception:
            error = str(exception)
    return FileSnapshot(
        kind, mode, size, hashlib.sha256(content).hexdigest() if content is not None else None, link, parsed, error
    )


def file_map_snapshot(files: Mapping[str, bytes], modes: Mapping[str, int], namespace: str) -> dict[str, FileSnapshot]:
    """Snapshot regular files before archive serialization, with explicit resolved modes."""
    return {
        f"{namespace}/{name}": _file_snapshot(name, content, tarfile.REGTYPE.hex(), modes[name], len(content), "")
        for name, content in files.items()
    }


def task_snapshot(
    source: str,
    path: str,
    outcome: str,
    *,
    task_binary: bytes | None = None,
    solution_binary: bytes | None = None,
    detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> TaskSnapshot:
    return TaskSnapshot(
        source,
        path,
        outcome,
        detail,
        {**archive_snapshot(task_binary, "task"), **archive_snapshot(solution_binary, "oracle")},
        metadata or {},
    )


def json_temporal(value: Any) -> dict[str, str]:
    """Keep TOML dates distinguishable from strings in the report protocol."""
    if isinstance(value, (datetime, date, time)):
        return {"toml_type": type(value).__name__, "iso8601": value.isoformat()}
    raise TypeError(f"Cannot encode {type(value).__name__}")


def snapshot_from_dict(value: dict[str, Any]) -> TaskSnapshot:
    return TaskSnapshot(
        value["source"],
        value["path"],
        value["outcome"],
        value["detail"],
        {path: FileSnapshot(**file) for path, file in value["files"].items()},
        value["metadata"],
        value.get("legacy_static_rejection"),
        value.get("legacy_static_detail"),
    )
