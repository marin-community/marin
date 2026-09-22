# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.datakit.chat_normalize import ChatChannel, message_text
from marin.datakit.download.ultrachat_persona import row_to_chat_doc
from openai_harmony import Message


@pytest.mark.parametrize("include_closing_user", [False, True])
def test_row_to_chat_doc_restores_prompt_and_ends_on_assistant(include_closing_user: bool) -> None:
    messages = [
        {"role": "user", "content": None},
        {"role": "assistant", "content": "Initial answer"},
        {"role": "user", "content": "Follow-up"},
        {"role": "assistant", "content": "Final answer"},
    ]
    if include_closing_user:
        messages.append({"role": "user", "content": "Thanks"})
    row = {
        "prompt_id": "prompt-1",
        "persona_uuid": "persona-1",
        "messages": messages,
        "turns": len(messages),
        "end_reason": "resolved",
    }

    [document] = row_to_chat_doc(row, {"prompt-1": "Restored opening prompt"})
    restored = [Message.from_dict(message) for message in document["messages"]]

    assert message_text(restored[0]) == "Restored opening prompt"
    assert restored[-1].channel == ChatChannel.FINAL
    assert message_text(restored[-1]) == "Final answer"
    assert document["source_id"] == "prompt-1:persona-1"


def test_row_to_chat_doc_excludes_missing_source_prompt() -> None:
    row = {
        "prompt_id": "missing",
        "persona_uuid": "persona-1",
        "messages": [
            {"role": "user", "content": None},
            {"role": "assistant", "content": "Answer"},
        ],
        "turns": 2,
        "end_reason": "resolved",
    }

    assert row_to_chat_doc(row, {}) == []
