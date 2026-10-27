# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The assistant messages of Snowball replies: the grader reads the answer, not the reasoning."""

import pytest

# The policy runs in the marin-taskforge environment; other suites skip it.
pytest.importorskip("taskforge")
pytest.importorskip("transformers")

import json

from experiments.post_training.rsi.policy import assistant_message

CALL = '<tool_call>\n{"name": "shell", "arguments": {"command": "ls"}}\n</tool_call>'


def test_reasoning_is_not_the_answer_or_the_call():
    message = assistant_message(f"<|start_think|>Run ls.<|end_think|>Listing.{CALL}", frozenset({"shell"}))
    assert message["reasoning_content"] == "Run ls."
    assert message["content"] == "Listing."
    assert [(call["function"]["name"], json.loads(call["function"]["arguments"])) for call in message["tool_calls"]] == [
        ("shell", {"command": "ls"})
    ]

    # A turn that used all its tokens in its reasoning has no answer.
    assert assistant_message("<|start_think|>The answer is 4, or") == {
        "role": "assistant",
        "reasoning_content": "The answer is 4, or",
        "content": "",
    }
