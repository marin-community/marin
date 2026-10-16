# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode next-action: an OpenHands expert turn, graded by :mod:`verifyit.adapters.openhands_next_action`.

Only the rule tier runs, so the reward is deterministic and pairs the rules leave open score 0
with reason ``undecided``. The adapter imports the judge client, so this mode needs the ``judge``
extra even though it never calls a judge.
"""

import json
from pathlib import Path
from typing import Any

from verifyit.adapters.openhands_next_action import Operation, TurnState, call_action, grade_next_action
from verifyit.grade import InvalidTask, Reward, read_output, scored
from verifyit.modes.grade_pivot import output_message
from verifyit.spec import NextActionSpec


def _expected_call(spec: NextActionSpec) -> dict[str, Any]:
    try:
        call = json.loads(spec.expected_call)
    except json.JSONDecodeError as error:
        raise InvalidTask("expected_call must be a JSON object") from error
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), str):
        raise InvalidTask("expected_call needs a name and JSON-string arguments")
    return call


def _state(spec: NextActionSpec) -> TurnState:
    return TurnState(spec.repo_root, spec.cwd)


def validate_next_action(spec: NextActionSpec) -> None:
    if not spec.repo_root.startswith("/") or not spec.cwd.startswith("/"):
        raise InvalidTask("repo_root and cwd must be absolute paths")
    expert = call_action(_expected_call(spec), _state(spec))
    if expert.operation in (Operation.THINK, Operation.TRACKER, Operation.UNPARSEABLE):
        raise InvalidTask(f"expert turn is not a gradable pivot: {expert.operation}")


def grade_next_action_candidate(spec: NextActionSpec, message: dict[str, Any]) -> Reward:
    return grade_next_action(_expected_call(spec), message, _state(spec))


def grade(spec: NextActionSpec, _tests_dir: Path, workspace: Path) -> Reward:
    validate_next_action(spec)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    message = output_message(text)
    if message is None:
        return scored(0.0, reason="invalid_message")
    return grade_next_action_candidate(spec, message)
