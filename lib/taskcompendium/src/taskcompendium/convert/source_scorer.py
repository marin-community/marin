# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade with a scorer function installed in the grader image, unchanged.

The grader runs ``verifyit/execution/source_callable.py`` in the grader image. It reads
``invocation.json`` (the scorer's ``module:function``, which inputs to pass, and how to read the
reward) and ``config.json`` (the row's grading data under ``contract``), calls the scorer on the
extracted answer, and writes ``{"reward": ...}`` to the score file.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verifyit.execution import source_callable

from taskcompendium.grader import GraderPackage
from taskcompendium.models import EnvironmentRequirements, FileReward, RewardFile, RewardFileFormat, ScriptGrader
from taskcompendium.runtime.resources import inline_resource

SOURCE_CALLABLE = Path(source_callable.__file__)
ANSWER_PATH = "/app/answer.txt"
SCORE_PATH = "/logs/verifier/score.json"


def source_scorer_package(
    *,
    invocation: Mapping[str, Any],
    config: Mapping[str, Any],
    environment: EnvironmentRequirements,
    timeout: float,
    state_path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> GraderPackage:
    """A script grader that calls the image-installed scorer named by ``invocation``.

    ``state_path`` names a file the scorer reads for the captured terminal message, when the
    scorer needs tool calls rather than answer text.
    """
    argv = (
        "python3",
        "/tests/source_callable.py",
        "/tests/invocation.json",
        "/tests/config.json",
        ANSWER_PATH,
        SCORE_PATH,
    )
    return GraderPackage(
        ScriptGrader(
            argv=(*argv, state_path) if state_path is not None else argv,
            cwd="/",
            env=dict(env or {}),
            environment=environment,
            answer_path=ANSWER_PATH,
            reward=FileReward(files=(RewardFile(path=SCORE_PATH, format=RewardFileFormat.JSON),)),
            timeout=timeout,
        ),
        (
            inline_resource("source_callable.py", SOURCE_CALLABLE.read_bytes()),
            inline_resource("invocation.json", json.dumps(dict(invocation), allow_nan=False, sort_keys=True).encode()),
            inline_resource("config.json", json.dumps(dict(config), allow_nan=False, sort_keys=True).encode()),
        ),
    )
