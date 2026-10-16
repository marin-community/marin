# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Earlier assistant reasoning is opt-in: unset, tasks serialize as before; set, it renders as reasoning_content."""

import pytest

from taskcompendium.models import AssistantToolCalls, ConversationInput, ConversationToolCall, TextMessage, ToolResult
from taskcompendium.submission import conversation_messages

CALL = ConversationToolCall(call_id="c", name="ls", arguments={})
RESULT = (ToolResult(call_id="c", content="a.py"), TextMessage(role="user", content="Go on."))


def test_unset_reasoning_leaves_serialized_conversations_unchanged():
    context = ConversationInput(
        events=(TextMessage(role="assistant", content="Hi."), AssistantToolCalls(calls=(CALL,)), *RESULT)
    )
    assert "reasoning" not in context.model_dump_json()
    assert all("reasoning_content" not in message for message in conversation_messages(context.events))


def test_set_reasoning_renders_as_reasoning_content_and_round_trips():
    context = ConversationInput(
        events=(
            TextMessage(role="assistant", content="Hi.", reasoning="greet"),
            AssistantToolCalls(calls=(CALL,), reasoning="look"),
            *RESULT,
        )
    )
    assert ConversationInput.model_validate_json(context.model_dump_json()) == context
    assert [message.get("reasoning_content") for message in conversation_messages(context.events)] == [
        "greet",
        "look",
        None,
        None,
    ]


def test_only_assistant_messages_carry_reasoning():
    with pytest.raises(ValueError, match="Only assistant messages"):
        TextMessage(role="user", content="Hi.", reasoning="hidden")
