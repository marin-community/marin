# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.openthoughts4_code import HF_DATASET_ID, row_to_chat_doc, row_to_doc


def test_row_to_chat_doc_preserves_code_reasoning_conversation():
    row = {
        "messages": [
            {"role": "user", "content": "Solve this."},
            {"role": "assistant", "content": "Plan it.</think>\n```python\nprint(1)\n```"},
        ],
        "prompt_index": 7,
        "response_index": 2,
        "source_id": "source-1",
    }

    [document] = row_to_chat_doc(row)

    assert document["source"] == HF_DATASET_ID
    assert document["messages"] == [
        {"role": "user", "name": None, "content": [{"type": "text", "text": "Solve this."}]},
        {"role": "assistant", "name": None, "channel": "analysis", "content": [{"type": "text", "text": "Plan it."}]},
        {
            "role": "assistant",
            "name": None,
            "channel": "final",
            "content": [{"type": "text", "text": "```python\nprint(1)\n```"}],
        },
    ]
    assert document["prompt_index"] == 7
    assert document["response_index"] == 2


def test_row_to_chat_doc_leaves_delimiter_free_response_as_final_answer():
    [document] = row_to_chat_doc(
        {
            "messages": [
                {"role": "user", "content": "Solve this."},
                {"role": "assistant", "content": "Here is the answer."},
            ]
        }
    )

    assert document["messages"][1]["content"] == [{"type": "text", "text": "Here is the answer."}]


def test_row_to_doc_renders_the_complete_transcript():
    [document] = row_to_doc(
        {
            "messages": [
                {"role": "user", "content": "Solve this."},
                {"role": "assistant", "content": "Answer."},
            ]
        }
    )

    assert document["text"] == "<user>\nSolve this.\n</user>\n\n<assistant>\nAnswer.\n</assistant>"
