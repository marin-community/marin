# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode TaskTrove archive rows and keep their files in the right resource roles.

A TaskTrove row is a Harbor task archive: ``instruction.md``, ``task.toml``, ``environment/``,
``tests/`` and optionally ``solution/``, with paths relative to the archive root.
"""

import base64
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from taskcompendium.models import ResourceGroups, TaskResource
from taskcompendium.pipeline.inputs import StagedInputs
from taskcompendium.runtime.resources import inline_resource

TASKS_FILE = "tasks.parquet"

INSTRUCTION = "instruction.md"
TASK_TOML = "task.toml"
DOCKERFILE = "environment/Dockerfile"
TEST_SH = "tests/test.sh"
SOLUTION_DIR = "solution/"
SOLVE_SH = "solution/solve.sh"
TESTS_MOUNT = "/tests"
"""Where Harbor mounts a task's ``tests/`` directory, and only at grading time."""


@dataclass
class TaskFiles:
    """Decoded contents of one task archive, keyed by path relative to the archive root."""

    files: dict[str, bytes] = field(default_factory=dict)

    def text(self, path: str) -> str:
        return self.files[path].decode("utf-8", errors="replace")

    def get_text(self, path: str) -> str | None:
        blob = self.files.get(path)
        return None if blob is None else blob.decode("utf-8", errors="replace")

    @property
    def has_solution(self) -> bool:
        return any(p.startswith(SOLUTION_DIR) for p in self.files)

    def under(self, prefix: str) -> dict[str, bytes]:
        return {p: b for p, b in self.files.items() if p.startswith(prefix)}

    def write_to(self, root: Path) -> None:
        """Materialize every file under ``root``, creating directories as needed."""
        for path, data in self.files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def unpack_task_binary(row: dict[str, Any], _inputs: StagedInputs) -> dict[str, Any]:
    """Expose an archived Harbor task's files, base64-encoded, with its instruction and verifier data."""
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


def archive_file(data: Mapping[str, Any], path: str) -> bytes | None:
    """One file from an unpacked archive row, or ``None`` when the archive lacks it."""
    files = data.get("files")
    value = files.get(path) if isinstance(files, dict) else None
    return base64.b64decode(value, validate=True) if isinstance(value, str) else None


def archive_resources(data: Mapping[str, Any]) -> ResourceGroups:
    """Keep setup files with the worker, tests with the grader and everything else with the oracle."""
    files = data["files"]
    metadata = data["file_metadata"]

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
        json.dumps({"archive_sha256": data["archive_sha256"], "source_path": data["path"]}).encode(),
    )
    return ResourceGroups(
        worker=tuple(item for path, item in resources.items() if path.startswith("setup_files/")),
        verifier=(
            provenance,
            *(item for path, item in resources.items() if path.startswith("tests/")),
        ),
        oracle=tuple(item for path, item in resources.items() if not path.startswith(("setup_files/", "tests/"))),
    )


def archive_files(data: Mapping[str, Any]) -> TaskFiles:
    """The files of a row decoded by :func:`unpack_task_binary`."""
    return TaskFiles({path: base64.b64decode(encoded, validate=True) for path, encoded in data["files"].items()})
