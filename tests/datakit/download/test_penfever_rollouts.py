# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.datakit.chat_normalize import validate_chat_messages
from marin.datakit.download.opencode import opencode_conversation, opencode_protocol_messages
from marin.datakit.download.penfever_rollouts import PenfeverRollout, row_to_chat_doc
from marin.datakit.download.rollout_transforms import openai_chat_document, openai_chat_messages


def _dataset(cohort: str) -> PenfeverRollout:
    return PenfeverRollout(
        cohort_name=cohort,
        teacher="teacher",
        task_source="task",
        hf_dataset_id="test/penfever",
        revision="abc123",
    )


@pytest.mark.parametrize("prompts", [None, [], [[]]])
def test_opencode_filters_rows_without_recorded_initial_prompt(prompts):
    transform = row_to_chat_doc(_dataset("qwen35-122b-131k-opencode"))
    row = {
        "conversations": [{"role": "user", "content": ""}, {"role": "assistant", "content": "Done."}],
        "prompt_token_ids": prompts,
        "instruction": "The displayed instruction cannot recover the served prompt.",
    }
    assert transform(row) == []


@pytest.mark.parametrize("cohort", ["minimax-m27-131k", "qwen35-122b-32k"])
def test_terminal_protocol_becomes_reasoning_call_and_observation(cohort):
    transform = row_to_chat_doc(_dataset(cohort))
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


@pytest.mark.parametrize("cohort", ["minimax-m27-131k", "qwen35-122b-32k"])
@pytest.mark.parametrize("observation", ["<|start_think|>assistant", "<tool_response>contents</tool_response>"])
def test_terminal_cohorts_quarantine_malformed_observations(cohort, observation):
    transform = row_to_chat_doc(_dataset(cohort))
    assert (
        transform(
            {
                "conversations": [
                    {"role": "user", "content": "Task Description:\nInspect the file."},
                    {
                        "role": "assistant",
                        "content": (
                            '{"analysis":"Inspect.","plan":"Read.",'
                            '"commands":[{"keystrokes":"cat file\\n","duration":0.1}]}'
                        ),
                    },
                    {"role": "user", "content": f"New Terminal Output:\n{observation}"},
                    {
                        "role": "assistant",
                        "content": '{"analysis":"Done.","plan":"Stop.","commands":[],"task_complete":true}',
                    },
                ]
            }
        )
        == []
    )


@pytest.mark.parametrize("reasoning", ["", "<think>Continue implementing.</think>\n"])
@pytest.mark.parametrize(
    "continuation",
    ["Implemented the function.", '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'],
)
def test_opencode_continuation_placeholder_does_not_end_assistant_turn(reasoning, continuation):
    converted = opencode_protocol_messages(
        [
            {"role": "user", "content": "Implement the function."},
            {"role": "assistant", "content": reasoning + "(tool use)"},
            {"role": "assistant", "content": continuation},
        ],
        [{"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}],
    )
    messages, _ = converted
    chat = openai_chat_messages(messages)
    validate_chat_messages(chat)
    rendered = [message.to_dict() for message in chat]
    assert all(message["content"] != [{"type": "text", "text": "(tool use)"}] for message in rendered)
    if reasoning:
        assert rendered[1]["content"] == [{"type": "text", "text": "Continue implementing."}]
        assert rendered[1]["channel"] == "analysis"
    if continuation.startswith("<tool_call>"):
        assert rendered[-1]["recipient"] == "functions.read"
        assert json.loads(rendered[-1]["content"][0]["text"]) == {"path": "a.py"}
    else:
        assert rendered[-1]["content"] == [{"type": "text", "text": continuation}]
        assert rendered[-1]["channel"] == "final"


def test_opencode_keeps_terminal_placeholder_when_no_continuation_was_recorded():
    messages, _ = opencode_protocol_messages(
        [{"role": "user", "content": "Implement the function."}, {"role": "assistant", "content": "(tool use)"}],
        [],
    )
    assert messages[-1] == {"role": "assistant", "content": "(tool use)"}


def test_qwen_handoff_merges_user_requests_after_terminal_protocol_parsing():
    [document] = row_to_chat_doc(_dataset("qwen35-122b-32k"))(
        {
            "conversations": [
                {"role": "user", "content": "Task Description:\nWrite the workflow."},
                {
                    "role": "assistant",
                    "content": '{"analysis":"Done.","plan":"Stop.","commands":[],"task_complete":true}',
                },
                {"role": "user", "content": "Are you sure you want to mark the task as complete?"},
                {"role": "user", "content": "Summarize your work for the next agent."},
                {
                    "role": "assistant",
                    "content": '{"analysis":"The workflow is ready.","plan":"Stop.","commands":[],"task_complete":true}',
                },
            ]
        }
    )
    messages = document["messages"]
    users = [message for message in messages if message["role"] == "user"]
    assert len(users) == 2
    assert users[-1]["content"] == [
        {
            "type": "text",
            "text": "Are you sure you want to mark the task as complete?\n\nSummarize your work for the next agent.",
        }
    ]
    assert messages[-1]["content"] == [{"type": "text", "text": "Task complete."}]
    assert any("The workflow is ready." in part["text"] for message in messages for part in message["content"])
