# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builders for test submissions across the public answer conventions."""

import json

from taskcompendium.models import AssistantToolCalls, ConversationToolCall, ConversationTrace, TaskSpec, TextMessage
from taskcompendium.submission import (
    ANSWER_CALL_NAME,
    ANSWER_FIELD,
    AnswerCall,
    GradingAttempt,
    JsonAnswer,
    SubmissionConvention,
)


def answer_attempt(task: TaskSpec, convention: SubmissionConvention, answer: str) -> GradingAttempt:
    """Build the assistant response extracted by a supported text convention."""
    if isinstance(convention, AnswerCall):
        response = AssistantToolCalls(
            calls=(
                ConversationToolCall(
                    call_id="answer",
                    name=ANSWER_CALL_NAME,
                    arguments={ANSWER_FIELD: answer},
                ),
            )
        )
    else:
        content = json.dumps({ANSWER_FIELD: answer}) if isinstance(convention, JsonAnswer) else answer
        response = TextMessage(role="assistant", content=content)
    return GradingAttempt(ConversationTrace(events=(*task.context.events, response)), object())
