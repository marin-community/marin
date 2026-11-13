# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ship a grade script with a row's hidden data and the scorer files it imports.

The grader runs ``python3 /tests/grade.py`` from ``/`` in the grader image. The script puts
``/tests`` on its import path, reads ``/tests/config.json``, the reply at the answer path or the
conversation at ``/tests/conversation.json``, scores with the shipped scorer modules, and prints
the reward as its last line of output. It exits nonzero when it cannot score, so a missing module
or dependency is an infrastructure error, never a zero.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from taskcompendium.grader import GraderPackage
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, StdoutReward, TaskResource
from taskcompendium.runtime.resources import inline_resource

GRADE_ARGV = ("python3", "/tests/grade.py")


def shipped_files(root: Path, *paths: str) -> tuple[TaskResource, ...]:
    """The files at ``paths`` below ``root``, installed at the same relative paths below ``/tests``.

    Keeping a vendored module's package path lets the script import it as upstream does.
    """
    return tuple(inline_resource(path, (root / path).read_bytes()) for path in paths)


def grade_script(script: Path, *imports: TaskResource) -> tuple[TaskResource, ...]:
    """``script`` installed as ``/tests/grade.py``, with the files it imports."""
    return (inline_resource("grade.py", script.read_bytes()), *imports)


def script_package(
    files: tuple[TaskResource, ...],
    config: Mapping[str, Any],
    *,
    environment: EnvironmentRequirements,
    timeout: float,
    answer_path: str | None,
    env: Mapping[str, str] | None = None,
) -> GraderPackage:
    """A script grader for ``files`` from :func:`grade_script`, with ``config`` as ``/tests/config.json``.

    ``answer_path`` is where the runtime writes the final reply; ``None`` grades the agent's captured files.
    """
    grader = ScriptGrader(
        argv=GRADE_ARGV,
        cwd="/",
        env=dict(env or {}),
        environment=environment,
        answer_path=answer_path,
        reward=StdoutReward(),
        timeout=timeout,
    )
    config_file = inline_resource("config.json", json.dumps(dict(config), allow_nan=False, sort_keys=True).encode())
    return GraderPackage(grader, (*files, config_file))
