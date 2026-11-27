# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-captured completion status that a script grader reads beside the model's answer.

The status comes from the rollout host, never from the model's text: the transport's stop reason,
whether the output budget truncated the reply, a provider refusal, and any tool-call events in the
graded reply. A grader that declares ``reply_status_path`` receives it as JSON at that path; an
attempt without a recorded stop reason cannot be graded.
"""

import json

from taskcompendium.models import AssistantToolCalls, GradingAttempt, ScriptGrader, TaskSpec
from taskcompendium.runtime.staging import StagedFile

REPLY_ORIGIN = "provider"


def reply_status_files(task: TaskSpec, attempt: GradingAttempt) -> list[StagedFile]:
    """The staged reply status for a grader that declares one, else nothing."""
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or grader.reply_status_path is None:
        return []
    if attempt.stop_reason is None:
        raise ValueError(f"Task {task.id} grades host-captured reply status, but the attempt recorded no stop reason")
    reply = attempt.conversation.events[len(task.context.events) :]
    tool_events = tuple(
        f"reply[{index}].tool_calls" for index, event in enumerate(reply) if isinstance(event, AssistantToolCalls)
    )
    status = {
        "origin": REPLY_ORIGIN,
        "finish_reason": attempt.stop_reason,
        "truncated": attempt.stop_reason == "length",
        "provider_refusal": False,
        "tool_events": list(tool_events),
    }
    return [StagedFile(grader.reply_status_path, json.dumps(status, allow_nan=False).encode())]
