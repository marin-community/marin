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
    assert messages[0]["content"][0]["text"].startswith("Task Description:")
    assert messages[1]["channel"] == "analysis"
    assert messages[1]["content"] == [{"type": "text", "text": "Inspect first."}]
    assert messages[2]["recipient"] == "functions.terminal"
    assert json.loads(messages[2]["content"][0]["text"])["commands"][0]["keystrokes"] == "ls\n"
    assert messages[3] == {
        "role": "tool",
        "name": "functions.terminal",
        "channel": "commentary",
        "recipient": "assistant",
        "content": [{"type": "text", "text": "New Terminal Output:\nfile.py"}],
    }
    assert messages[-1]["channel"] == "final"
    assert messages[-1]["content"] == [{"type": "text", "text": "Task complete."}]
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

    messages = document["messages"]
    assert messages[1]["channel"] == "analysis"
    assert messages[1]["content"] == [{"type": "text", "text": "Inspect."}]
    assert messages[2]["recipient"] == "functions.bash"
    assert json.loads(messages[2]["content"][0]["text"]) == {"command": "ls"}
    assert messages[3]["role"] == "tool"
    assert messages[3]["name"] == "functions.bash"
    assert json.loads(document["chat_template_kwargs"])["tools"][0]["name"] == "bash"


def test_opencode_protocol_matches_parallel_calls_to_separate_observations():
    transform = row_to_chat_doc(_dataset("qwen35-122b-131k-opencode"))
    [document] = transform(
        {
            "conversations": [
                {"role": "user", "content": "Inspect both files."},
                {
                    "role": "assistant",
                    "content": (
                        '<tool_call>{"name":"read","arguments":{"path":"a.py","offset":10}}</tool_call>'
                        '<tool_call>{"name":"read","arguments":{"path":"b.py"}}</tool_call>'
                    ),
                },
                {"role": "user", "content": "contents of a.py"},
                {"role": "user", "content": "contents of b.py"},
                {"role": "assistant", "content": "Both files are valid."},
            ]
        }
    )

    messages = document["messages"]
    assert [m["recipient"] for m in messages[1:3]] == ["functions.read", "functions.read"]
    assert [json.loads(m["content"][0]["text"])["path"] for m in messages[1:3]] == ["a.py", "b.py"]
    assert [m["content"][0]["text"] for m in messages[3:5]] == ["contents of a.py", "contents of b.py"]
    [tool] = json.loads(document["chat_template_kwargs"])["tools"]
    assert tool["parameters"]["properties"] == {"offset": {"type": "number"}, "path": {"type": "string"}}
    assert tool["parameters"]["required"] == ["path"]


def test_opencode_protocol_links_bundled_parallel_call_observation_to_each_call():
    transform = row_to_chat_doc(_dataset("qwen35-122b-131k-opencode"))

    [document] = transform(
        {
            "conversations": [
                {"role": "user", "content": "Inspect both files."},
                {
                    "role": "assistant",
                    "content": (
                        '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'
                        '<tool_call>{"name":"read","arguments":{"path":"b.py"}}</tool_call>'
                    ),
                },
                {"role": "user", "content": "combined output"},
                {"role": "assistant", "content": "Both files are valid."},
            ]
        }
    )

    calls = document["messages"][1:3]
    observations = document["messages"][3:5]
    assert [call["recipient"] for call in calls] == ["functions.read", "functions.read"]
    assert [m["name"] for m in observations] == ["functions.read", "functions.read"]
    assert [m["content"][0]["text"] for m in observations] == ["combined output", "combined output"]


def test_opencode_protocol_recovers_prompt_from_instruction():
    transform = row_to_chat_doc(_dataset("qwen35-122b-131k-opencode"))

    [document] = transform(
        {
            "instruction": "Fix the code.",
            "conversations": [{"role": "user", "content": ""}, {"role": "assistant", "content": "Done."}],
        }
    )

    assert document["messages"][0] == {
        "role": "user",
        "name": None,
        "content": [{"type": "text", "text": "Fix the code."}],
    }


def test_opencode_protocol_drops_conversation_without_a_recoverable_prompt():
    transform = row_to_chat_doc(_dataset("qwen35-122b-131k-opencode"))

    assert transform({"conversations": [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]}) == []
