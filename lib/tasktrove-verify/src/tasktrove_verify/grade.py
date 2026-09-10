# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch a spec to its mode and turn a task defect into an ``invalid_task`` reward."""

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tasktrove_verify.reward import InvalidTask, Reward, invalid_task
from tasktrove_verify.spec import Mode, Spec, mode_of

# Each mode module takes its own spec type; the dispatch key guarantees the match.
Grader = Callable[[Any, Path, Path], Reward]

# Mode modules are imported on first use: several depend on an extra (math-verify, jsonschema,
# reasoning-gym, openai) that only the images needing that mode install. A missing extra
# surfaces as an ImportError from the grader, which the CLI records as infra_error.
MODE_MODULES: dict[Mode, str] = {
    Mode.MCQ: "mcq",
    Mode.MATH: "math_answer",
    Mode.NUMERIC: "numeric",
    Mode.EXACT: "exact",
    Mode.JSON_SCHEMA: "json_schema",
    Mode.IFEVAL: "ifeval",
    Mode.REASONING_GYM: "reasoning_gym_answer",
    Mode.STDIO: "stdio",
    Mode.PYTEST: "pytest_report",
    Mode.JUNIT: "junit",
    Mode.GOTEST: "gotest",
    Mode.JUDGE: "judge",
    Mode.SCRIPT: "script",
}
GRADERS: dict[Mode, Grader] = {}


def grader_for(mode: Mode) -> Grader:
    if mode not in GRADERS:
        GRADERS[mode] = importlib.import_module(f"tasktrove_verify.modes.{MODE_MODULES[mode]}").grade
    return GRADERS[mode]


def grade(spec: Spec, tests_dir: Path, workspace: Path) -> Reward:
    """Grade one task. ``tests_dir`` holds verifier.toml and its data; ``workspace`` is the agent's tree.

    Modes that read an output file use ``spec.output``; execution modes use ``spec.workspace``.
    ``workspace`` here is the fallback for specs that leave those at their defaults but run
    somewhere else, such as a local gate in a temporary directory.
    """
    try:
        return grader_for(mode_of(spec))(spec, tests_dir, workspace)
    except InvalidTask as error:
        return invalid_task(str(error))
