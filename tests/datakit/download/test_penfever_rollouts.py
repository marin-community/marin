# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.datakit.download.penfever_rollouts import PenfeverRollout, row_to_chat_doc


def _dataset(cohort: str) -> PenfeverRollout:
    return PenfeverRollout(
        cohort_name=cohort,
        teacher="teacher",
        task_source="task",
        hf_dataset_id="test/penfever",
        revision="abc123",
    )


def test_terminal_protocol_becomes_reasoning_call_and_observation():
    transform = row_to_chat_doc(_dataset("minimax-m27-131k"))
    [document] = transform(
        {
            "conversations": [
                {"role": "user", "content": "old protocol\n\nTask Description:\nFix the code."},
                {
                    "role": "assistant",
                    "content": (
                        '<think><think>Inspect first.</think></think>\n{"analysis":"duplicate","plan":"duplicate",'
                        '"commands":[{"keystrokes":"ls\\n","duration":0.1}]}'
                    ),
                },
                {"role": "user", "content": "New Terminal Output:\nfile.py"},
                {
                    "role": "assistant",
                    "content": '{"analysis":"Done.","plan":"Stop.","commands":[],"task_complete":true}',
                },
            ],
            "result": "1.0",
        }
    )

    messages = document["messages"]
    assert messages[1]["content"].startswith("Task Description:")
    assert messages[2]["content"] == "<|start_think|>Inspect first.<|end_think|>"
    call = messages[2]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"])["commands"][0]["keystrokes"] == "ls\n"
    assert messages[3] == {
        "role": "tool",
        "content": "New Terminal Output:\nfile.py",
        "name": "terminal",
        "tool_call_id": call["id"],
    }
    assert messages[4]["content"].endswith("Task complete.")
    assert json.loads(document["chat_template_kwargs"])["tools"][0]["name"] == "terminal"


def test_opencode_protocol_becomes_structured_call_and_observation():
    transform = row_to_chat_doc(_dataset("qwen35-122b-131k-opencode"))
    [document] = transform(
        {
            "conversations": [
                {"role": "user", "content": "Fix the code."},
                {
                    "role": "assistant",
                    "content": (
                        "<think>Inspect.</think>\n<tool_call>\n"
                        '{"name":"bash","arguments":{"command":"ls"}}\n</tool_call>'
                    ),
                },
                {"role": "user", "content": "file.py"},
                {"role": "assistant", "content": "<think>Done.</think>The fix is complete."},
            ]
        }
    )

    call = document["messages"][1]["tool_calls"][0]
    assert document["messages"][1]["content"] == "<|start_think|>Inspect.<|end_think|>"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls"}
    assert document["messages"][2]["role"] == "tool"
    assert document["messages"][2]["tool_call_id"] == call["id"]
    assert json.loads(document["chat_template_kwargs"])["tools"][0]["name"] == "bash"


def test_opencode_protocol_recovers_prompt_from_instruction():
    transform = row_to_chat_doc(_dataset("qwen35-122b-131k-opencode"))

    [document] = transform(
        {
            "instruction": "Fix the code.",
            "conversations": [{"role": "user", "content": ""}, {"role": "assistant", "content": "Done."}],
        }
    )

    assert document["messages"][0] == {"role": "user", "content": "Fix the code."}


def test_opencode_protocol_drops_conversation_without_a_recoverable_prompt():
    transform = row_to_chat_doc(_dataset("qwen35-122b-131k-opencode"))

    assert transform({"conversations": [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]}) == []
