# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conversation tasks that a SkyRL grade script scores with a vendored scorer.

A task ships the script as ``/tests/grade.py``, one scorer module from ``scorers/`` under its file
name, and the row's hidden data as ``/tests/config.json``. The script runs in the grader image,
imports the scorer from ``/tests``, reads the reply from ``/app/answer.txt`` and prints the reward
as its last line of output.
"""

import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from taskcompendium.convert.conversation import conversation_task
from taskcompendium.grader import GraderPackage
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, StdoutReward, TaskSpec, TextMessage
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource

SCORERS = Path(__file__).with_name("scorers")
GRADE_ARGV = ("python3", "/tests/grade.py")


@cache
def _file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def scorer_task(
    row: RawRow,
    *,
    events: Sequence[TextMessage],
    script: Path,
    scorer: str,
    config: Mapping[str, Any],
    environment: EnvironmentRequirements,
    timeout: float,
    env: Mapping[str, str],
    evidence: Mapping[str, Any],
) -> TaskSpec:
    """A task whose final reply ``script`` grades in ``environment`` by importing ``scorers/<scorer>``."""
    grader = ScriptGrader(
        argv=GRADE_ARGV,
        cwd="/",
        env=dict(env),
        environment=environment,
        reward=StdoutReward(),
        timeout=timeout,
    )
    resources = (
        inline_resource("grade.py", _file_bytes(script)),
        inline_resource(scorer, _file_bytes(SCORERS / scorer)),
        inline_resource("config.json", json.dumps(dict(config), allow_nan=False, sort_keys=True).encode()),
    )
    return conversation_task(row, events=events, package=GraderPackage(grader, resources), evidence=evidence)
