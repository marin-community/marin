# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from openai_harmony import Message, Role

from marin.datakit.chat_normalize import ChatChannel, message_text, validate_chat_messages
from marin.datakit.download.openthoughts_agent import HF_DATASET_ID, row_to_chat_doc


def test_row_to_chat_doc_parses_reasoning_across_interactive_turns() -> None:
    row = {
        "run_id": "run-1",
        "conversations": [
            {"role": "user", "content": "Fix the repository."},
            {
                "role": "assistant",
                "content": '<think>Inspect the files first.</think>{"commands":["ls"],"task_complete":false}',
            },
            {"role": "user", "content": "New terminal output: README.md"},
            {
                "role": "assistant",
                "content": '<think>The requested change is complete.</think>{"commands":[],"task_complete":true}',
            },
        ],
    }

    [document] = row_to_chat_doc(row)
    messages = [Message.from_dict(message) for message in document["messages"]]

    validate_chat_messages(messages)
    assert [message.author.role for message in messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
        Role.ASSISTANT,
    ]
    assert messages[1].channel == ChatChannel.ANALYSIS
    assert message_text(messages[1]) == "Inspect the files first."
    assert messages[-1].channel == ChatChannel.FINAL
    assert document["source"] == HF_DATASET_ID
    assert document["source_id"] == "run-1"
