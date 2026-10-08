# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade with a scorer installed in the grader image, unchanged.

The grader runs a Python script from ``/tests`` in the grader image. The script reads the row's
grading data from ``config.json`` and the extracted answer from ``ANSWER_PATH``, calls the scorer, and
writes ``{"reward": ...}`` to ``SCORE_PATH``. :func:`source_scorer_package` ships
``verifyit/execution/source_callable.py``, which names the scorer's ``module:function``, its inputs and
how to read its reward in ``invocation.json``; :func:`grade_script_package` ships a source's own script.
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
SCORE_REWARD = FileReward(files=(RewardFile(path=SCORE_PATH, format=RewardFileFormat.JSON),))


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
            reward=SCORE_REWARD,
            timeout=timeout,
        ),
        (
            inline_resource("source_callable.py", SOURCE_CALLABLE.read_bytes()),
            inline_resource("invocation.json", json.dumps(dict(invocation), allow_nan=False, sort_keys=True).encode()),
            inline_resource("config.json", json.dumps(dict(config), allow_nan=False, sort_keys=True).encode()),
        ),
    )


def grade_script_package(
    script: str,
    source: bytes,
    *,
    config: Mapping[str, Any],
    environment: EnvironmentRequirements,
    timeout: float,
    leading_args: tuple[str, ...] = (),
    env: Mapping[str, str] | None = None,
) -> GraderPackage:
    """A script grader that runs ``source`` as ``python3 /tests/<script> *leading_args config ANSWER SCORE``.

    ``leading_args`` precede the config path; a script that loads its scorer from a fixed location
    takes that location there.
    """
    return GraderPackage(
        ScriptGrader(
            argv=("python3", f"/tests/{script}", *leading_args, "/tests/config.json", ANSWER_PATH, SCORE_PATH),
            cwd="/",
            env=dict(env or {}),
            environment=environment,
            answer_path=ANSWER_PATH,
            reward=SCORE_REWARD,
            timeout=timeout,
        ),
        (
            inline_resource(script, source),
            inline_resource("config.json", json.dumps(dict(config), allow_nan=False, sort_keys=True).encode()),
        ),
    )
