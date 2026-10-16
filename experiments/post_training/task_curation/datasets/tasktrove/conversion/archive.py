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
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from taskcompendium.convert.answers import unsupported
from taskcompendium.models import (
    EnvironmentRequirements,
    FileReward,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskResource,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection
from taskcompendium.runtime.resources import inline_resource

TASKS_FILE = "tasks.parquet"
TASKTROVE_REPO = "open-thoughts/TaskTrove"

INSTRUCTION = "instruction.md"
TASK_TOML = "task.toml"
DOCKERFILE = "environment/Dockerfile"
TEST_SH = "tests/test.sh"
VERIFIER_DATA = "tests/verifier_data.json"
SOLUTION_DIR = "solution/"
SOLVE_SH = "solution/solve.sh"
TESTS_MOUNT = "/tests"
"""Where Harbor mounts a task's ``tests/`` directory, and only at grading time."""
UV_IMAGE = "ghcr.io/astral-sh/uv:0.8"
TEST_SH_REWARD = FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),))
"""The numeric reward file a Harbor ``tests/test.sh`` writes."""


@dataclass
class TaskFiles:
    """Decoded contents of one task archive, keyed by path relative to the archive root."""

    files: dict[str, bytes] = field(default_factory=dict)

    def text(self, path: str) -> str:
        return self.files[path].decode("utf-8", errors="replace")

    def get_text(self, path: str) -> str | None:
        blob = self.files.get(path)
        return None if blob is None else blob.decode("utf-8", errors="replace")

    def under(self, prefix: str) -> dict[str, bytes]:
        return {p: b for p, b in self.files.items() if p.startswith(prefix)}


def unpack_task_binary(row: dict[str, Any], _context: ConversionContext) -> dict[str, Any]:
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
    if VERIFIER_DATA in files:
        prepared["verifier_data"] = json.loads(files[VERIFIER_DATA])
    return prepared


def archive_file(data: Mapping[str, Any], path: str) -> bytes | None:
    """One file from an unpacked archive row, or ``None`` when the archive lacks it."""
    files = data.get("files")
    value = files.get(path) if isinstance(files, dict) else None
    return base64.b64decode(value, validate=True) if isinstance(value, str) else None


def archive_resource(data: Mapping[str, Any], path: str) -> TaskResource:
    """Materialize one archived file while preserving its declared permissions."""
    original = data["file_metadata"][path]
    return TaskResource(
        path=path.removeprefix("tests/") if path.startswith("tests/") else path,
        source=inline_resource(path, base64.b64decode(data["files"][path], validate=True)).source,
        mode=original["mode"],
        mtime_ns=original["mtime_ns"],
    )


def archive_resources(data: Mapping[str, Any]) -> ResourceGroups:
    """Keep setup files with the worker, tests with the grader and everything else with the oracle."""
    resources = {path: archive_resource(data, path) for path in data["files"]}
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


def archive_script_grader(
    data: Mapping[str, Any],
    *,
    required: tuple[str, ...],
    environment: EnvironmentRequirements,
    answer_path: str | None,
    env: Mapping[str, str] | None = None,
) -> ScriptGrader | ImportRejection:
    """Grade with the archive's own ``tests/test.sh``, under the verifier timeout in its ``task.toml``.

    ``required`` names the other archive files the test script reads. An archive with links, which
    cannot be mounted as regular files, or without these files is unsupported.
    """
    if data.get("archive_links"):
        return unsupported("unsupported_archive_links", "The archive has links that cannot be mounted as regular files")
    missing = [path for path in (TASK_TOML, TEST_SH, *required) if archive_file(data, path) is None]
    if missing:
        return unsupported("missing_archive_grader", f"The archive lacks {', '.join(missing)}")
    task_toml = archive_file(data, TASK_TOML)
    assert task_toml is not None
    return ScriptGrader(
        argv=("bash", f"{TESTS_MOUNT}/test.sh"),
        cwd="/",
        env=dict(env or {}),
        environment=environment,
        answer_path=answer_path,
        reward=TEST_SH_REWARD,
        timeout=float(tomllib.loads(task_toml.decode())["verifier"]["timeout_sec"]),
    )
