# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The rollout records that the analysis reads: the failures that it takes, and the text that a model call sees."""

import json
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from rolloutengine.models import GradedRollout, RolloutOperation
from taskcompendium.grading_result import Outcome

MAX_TASK_CHARS = 300
FAILURE_OPERATION = "operation"
"""The key of ``RolloutFailure.diagnostics`` that names the interrupted operation, as rolloutengine records it."""
POLICY_INTERRUPTIONS = frozenset({RolloutOperation.ATTEMPT, RolloutOperation.ADVANCE})
"""The interruptions that the policy can cause: a slow or looping policy runs past the attempt deadline, and a bad
tool call can stop the harness after a model turn."""


def policy_failed(graded: GradedRollout, pass_threshold: float) -> bool:
    """A grade below ``pass_threshold``, no valid submission, or an interruption in ``POLICY_INTERRUPTIONS``.

    A task, grader, or infrastructure failure, and an interruption by the model server or the machine, shows no gap
    of the policy, so it does not become a training target.
    """
    failure = graded.rollout.failure
    if failure is not None:
        return failure.diagnostics.get(FAILURE_OPERATION) in POLICY_INTERRUPTIONS
    grade = graded.grade
    if grade.status == Outcome.SUBMISSION_FAILURE:
        return True
    return grade.status == Outcome.GRADED and grade.reward is not None and grade.reward < pass_threshold


def rollout_text(graded: GradedRollout) -> str:
    """Render the rollout summary, each conversation's model options, and its numbered messages."""
    lines = [f"{json.dumps(_summary(graded))}\n"]
    message_index = 0
    for conversation_index, conversation in enumerate(graded.rollout.conversations):
        header = {"conversation": conversation_index, "options": conversation.options}
        lines.append(f"{json.dumps(header)}\n")
        for message in conversation.messages:
            lines.append(f"[{message_index}] {_render(message)}\n")
            message_index += 1
    return "".join(lines)


def _summary(graded: GradedRollout) -> dict[str, Any]:
    """The grade, how the rollout stopped or what interrupted it, and its turns and tool calls."""
    rollout, grade = graded.rollout, graded.grade
    turns = [step.turn.message for conversation in rollout.conversations for step in conversation.steps]
    calls = [_call_text(call) for message in turns for call in message.get("tool_calls") or ()]
    text = " ".join(
        str(message.get("content") or "")
        for conversation in rollout.conversations
        for message in conversation.messages
        if message.get("role") == "user"
    )
    return {
        "run_id": rollout.run_id,
        "task_id": rollout.task_id,
        "outcome": grade.status.value,
        "reward": grade.reward,
        "stop_reason": rollout.stop_reason,
        "interrupted_during": rollout.failure.diagnostics.get(FAILURE_OPERATION) if rollout.failure else None,
        "turns": len(turns),
        "tool_calls": len(calls),
        "repeated_calls": sum(previous == call for previous, call in pairwise(calls)),
        "task": text[:MAX_TASK_CHARS],
    }


def _call_text(call: Any) -> str:
    """Render valid calls as ``name(arguments)`` and malformed calls as JSON."""
    function = call.get("function") if isinstance(call, Mapping) else None
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return f"{function['name']}({function.get('arguments', '')})"
    return f"malformed {json.dumps(call, default=str)}"


def _render(message: Mapping[str, Any]) -> str:
    calls = [_call_text(call) for call in message.get("tool_calls") or ()]
    text = " ".join([str(message.get("content") or ""), *(f"<tool call> {call}" for call in calls)]).strip()
    return f"{message.get('role')}: {text}"
