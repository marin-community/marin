# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.datakit.download.opencode import opencode_conversation, opencode_protocol_messages
from marin.datakit.download.penfever_rollouts import PenfeverRollout, row_to_chat_doc
from marin.datakit.download.rollout_transforms import openai_chat_document


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


def test_opencode_protocol_matches_parallel_calls_to_separate_observations():
    tools = [
        {
            "type": "function",
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "offset": {"type": "number"}},
                "required": ["path"],
            },
        }
    ]
    converted = opencode_protocol_messages(
        [
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
        ],
        tools,
    )
    messages, metadata = converted
    document = openai_chat_document(messages, "test", **metadata)

    messages = document["messages"]
    assert [m["recipient"] for m in messages[1:3]] == ["functions.read", "functions.read"]
    assert [json.loads(m["content"][0]["text"])["path"] for m in messages[1:3]] == ["a.py", "b.py"]
    assert [m["content"][0]["text"] for m in messages[3:5]] == ["contents of a.py", "contents of b.py"]
    [tool] = json.loads(document["chat_template_kwargs"])["tools"]
    assert tool["parameters"]["properties"] == {"offset": {"type": "number"}, "path": {"type": "string"}}
    assert tool["parameters"]["required"] == ["path"]


def test_opencode_protocol_links_bundled_parallel_call_observation_to_each_call():
    tools = [
        {
            "type": "function",
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "offset": {"type": "number"}},
                "required": ["path"],
            },
        }
    ]

    converted = opencode_protocol_messages(
        [
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
        ],
        tools,
    )
    messages, metadata = converted
    document = openai_chat_document(messages, "test", **metadata)

    calls = document["messages"][1:3]
    observations = document["messages"][3:5]
    assert [call["recipient"] for call in calls] == ["functions.read", "functions.read"]
    assert [m["name"] for m in observations] == ["functions.read", "functions.read"]
    assert [m["content"][0]["text"] for m in observations] == ["combined output", "combined output"]


def test_opencode_recovers_task_and_declared_tools_from_served_prompt():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                    "required": ["filePath"],
                },
            },
        }
    ]
    prompt = (
        "<|im_start|>system\n<tools>\n"
        + json.dumps(tools[0])
        + "\n</tools>\n<IMPORTANT>Source tool syntax.</IMPORTANT>\n"
        "You are opencode. Follow the task.\n<|im_end|>\n"
        "<|im_start|>user\nFix the real task.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    messages, recovered_tools = opencode_conversation(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": "Done."}], prompt
    )
    assert messages[:2] == [
        {"role": "system", "content": "You are opencode. Follow the task.\n"},
        {"role": "user", "content": "Fix the real task."},
    ]
    assert recovered_tools == tools
