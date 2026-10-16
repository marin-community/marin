# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Relay routing between a student (v0) and a teacher (GLM 5.3), decided per request and per trial.

Rules (owner-approved for #207): the student answers until a trigger.

- Format error: the newest observation is the harness's parse-error message. The teacher writes that
  one turn (a repair); control then returns to the student.
- Context budget: the prompt reaches ``budget_tokens`` (student tokenizer). The teacher takes over to
  the end of the episode.
- Completion claim: the student's reply claims the task is done. The reply is withheld; the teacher
  answers the same request (it confirms completion or continues) and keeps control to the end.

Trigger texts come from the pinned harnesses: Terminus-2's ``Previous response had parsing errors:``
and ``"task_complete": true``; mini-swe-agent 2.1.0's ``Tool call error:`` (``mini.yaml``
``format_error_template``) and ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum

TERMINUS_PARSE_ERROR = "Previous response had parsing errors:"
MINI_SWE_FORMAT_ERROR = "Tool call error:"
MINI_SWE_SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
TERMINUS_COMPLETE = re.compile(r'"task_complete"\s*:\s*true')


class Author(StrEnum):
    STUDENT = "v0"
    TEACHER = "glm"


class Trigger(StrEnum):
    FORMAT_ERROR = "format_error"
    CONTEXT_BUDGET = "context_budget"
    COMPLETION_CLAIM = "completion_claim"


@dataclass
class TrialState:
    """Relay state of one trial; ``takeover`` holds the trigger that handed control to the teacher."""

    takeover: Trigger | None = None


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def latest_observation(messages: list[dict]) -> str:
    """The text of the messages after the last assistant turn (empty before the first one)."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return "\n".join(_text(m.get("content")) for m in messages[index + 1 :])
    return ""


def is_format_error(messages: list[dict]) -> bool:
    observation = latest_observation(messages)
    return TERMINUS_PARSE_ERROR in observation or observation.lstrip().startswith(MINI_SWE_FORMAT_ERROR)


def is_completion_claim(reply: dict) -> bool:
    if TERMINUS_COMPLETE.search(_text(reply.get("content"))):
        return True
    for call in reply.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") == "mark_task_complete":
            return True
        arguments = function.get("arguments")
        if MINI_SWE_SUBMIT in (arguments if isinstance(arguments, str) else json.dumps(arguments)):
            return True
    return False


def route(state: TrialState, messages: list[dict], student_prompt_tokens: int, budget_tokens: int) -> tuple[Author, Trigger | None]:
    """Choose who answers this request, before any reply is generated. Updates ``state`` on a takeover."""
    if state.takeover is not None:
        return Author.TEACHER, None
    if is_format_error(messages):
        return Author.TEACHER, Trigger.FORMAT_ERROR
    if student_prompt_tokens >= budget_tokens:
        state.takeover = Trigger.CONTEXT_BUDGET
        return Author.TEACHER, Trigger.CONTEXT_BUDGET
    return Author.STUDENT, None
