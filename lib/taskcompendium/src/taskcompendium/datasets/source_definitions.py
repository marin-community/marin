# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode TaskTrove archive records for reusable normalizers."""

import base64
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.models import ResourceGroups, TaskResource
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.runtime.resources import inline_resource


def unpack_task_binary(row: dict[str, Any], _staged_root: StoragePath) -> dict[str, Any]:
    """Expose Harbor task files in the form consumed by TaskTrove converters."""
    blob = row["task_binary"]
    if not isinstance(blob, bytes):
        raise ValueError("Task binary must contain archived bytes")
    files: dict[str, bytes] = {}
    file_metadata = {}
    archive_links = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.issym() or member.islnk():
                archive_links[name] = {"target": member.linkname, "kind": "symlink" if member.issym() else "hardlink"}
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    files[name] = handle.read()
                    file_metadata[name] = {
                        "mode": format(member.mode & 0o7777, "04o"),
                        "mtime_ns": int(Decimal(member.pax_headers.get("mtime", str(member.mtime))) * 1_000_000_000),
                    }
    prepared = {
        "path": row["path"],
        "instruction": files["instruction.md"].decode(),
        "files": {name: base64.b64encode(data).decode() for name, data in files.items()},
        "archive_sha256": hashlib.sha256(blob).hexdigest(),
        "file_metadata": file_metadata,
        "archive_links": archive_links,
    }
    if "tests/verifier_data.json" in files:
        prepared["verifier_data"] = json.loads(files["tests/verifier_data.json"])
    return prepared


def tasktrove_files(config: str) -> SourceFiles:
    """Select and decode one TaskTrove component's archived tasks."""
    return SourceFiles((f"{config}/tasks.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary)


def archive_resources(row: Mapping[str, Any]) -> ResourceGroups:
    """Retain archive bytes and metadata in their original public or private role."""
    files = row["files"]
    metadata = row["file_metadata"]

    def resource(path: str, encoded: str) -> TaskResource:
        original = metadata[path]
        return TaskResource(
            path=path.removeprefix("tests/") if path.startswith("tests/") else path,
            source=inline_resource(path, base64.b64decode(encoded, validate=True)).source,
            mode=original["mode"],
            mtime_ns=original["mtime_ns"],
        )

    resources = {path: resource(path, encoded) for path, encoded in files.items()}
    provenance = inline_resource(
        "taskcompendium/archive-provenance.json",
        json.dumps({"archive_sha256": row["archive_sha256"], "source_path": row["path"]}).encode(),
    )
    return ResourceGroups(
        worker=tuple(item for path, item in resources.items() if path.startswith("setup_files/")),
        verifier=(
            provenance,
            *(item for path, item in resources.items() if path.startswith("tests/")),
        ),
        oracle=tuple(item for path, item in resources.items() if not path.startswith(("setup_files/", "tests/"))),
    )
