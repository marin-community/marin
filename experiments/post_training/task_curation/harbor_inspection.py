# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect immutable curation artifacts through Harbor's locked production runtime."""

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

from taskcompendium.harbor.native_package import write_native_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import NoGrader, TaskSpec

HARBOR_PROJECT = Path(__file__).resolve().parents[3] / "vendor/harbor"


def native_task_checksums(tasks: Iterable[TaskSpec]) -> dict[str, str]:
    """Read original native package checksums with Harbor, without executing source commands."""
    with TemporaryDirectory(prefix="native-package-checksums-") as scratch:
        paths = {}
        for task in tasks:
            if not isinstance(task.grader, NoGrader) or "harbor" not in task.tags:
                continue
            path = Path(scratch) / canonical_sha256(task.id)
            write_native_task(task, path)
            paths[task.id] = str(path)
        if not paths:
            return {}
        inspected = harbor_inspection("tasks", list(paths.values()))
        if set(inspected) != set(paths.values()):
            raise ValueError("Original native packages failed Harbor validation")
        return {task_id: inspected[path]["task_checksum"] for task_id, path in paths.items()}


def harbor_inspection(kind: str, paths: list[str]) -> dict:
    """Validate in Harbor's locked runtime without importing it into the curation environment."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(HARBOR_PROJECT),
            "--frozen",
            "--extra",
            "taskcompendium",
            "python",
            "-m",
            "harbor.cli.curation_inspect",
            kind,
        ],
        input=json.dumps(paths),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)
