# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.chat_normalize import ChatChannel, message_text
from marin.datakit.download.synthetic_misconceptions import row_to_chat_doc
from openai_harmony import Message


def test_row_to_chat_doc_removes_trailing_user_turn() -> None:
    row = {
        "id": 7,
        "opener_index": 3,
        "messages": [
            {"role": "user", "content": "I heard a misconception."},
            {"role": "assistant", "content": "Here is the correction."},
            {"role": "user", "content": "I am still unsure."},
            {"role": "assistant", "content": "Here is more evidence."},
            {"role": "user", "content": "Thanks, that clears it up."},
        ],
    }

    [document] = row_to_chat_doc(row)
    messages = [Message.from_dict(message) for message in document["messages"]]

    assert [message.author.role.value for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[-1].channel == ChatChannel.FINAL
    assert message_text(messages[-1]) == "Here is more evidence."
    assert document["source_id"] == "7:3"
