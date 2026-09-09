# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.datakit.chat_normalize import _normalize_chat_record
from marin.datakit.download.agenttrove import row_to_chat_doc as agenttrove_row_to_chat_doc
from marin.datakit.download.coderforge import row_to_chat_doc as coderforge_row_to_chat_doc
from marin.datakit.download.davinci_dev import env_row_to_chat_doc as davinci_row_to_chat_doc
from marin.datakit.download.gpt_oss_rollouts import row_to_chat_doc as gpt_oss_row_to_chat_doc
from marin.datakit.download.nemotron_terminal import row_to_chat_doc as nemotron_terminal_row_to_chat_doc
from marin.datakit.download.numinamath_tir import row_to_chat_doc as numinamath_row_to_chat_doc
from marin.datakit.download.swe_rebench_openhands import row_to_chat_doc as openhands_row_to_chat_doc
from marin.datakit.download.swe_zero_12m import row_to_chat_doc as swe_zero_row_to_chat_doc


@pytest.mark.parametrize("adapter", [agenttrove_row_to_chat_doc, nemotron_terminal_row_to_chat_doc])
def test_terminal_source_filters_inline_calls_without_tool_definitions(adapter) -> None:
    row = {
        "conversations": [
            {"role": "user", "content": "Read the repository."},
            {"role": "assistant", "content": '<tool_call>{"name":"read","arguments":{}}</tool_call>'},
        ],
    }
    assert adapter(row) == []


def test_agenttrove_merges_completion_and_handoff_prompts() -> None:
    row = {
        "conversations": [
            {"role": "user", "content": "Task Description:\nFix the task."},
            {"role": "assistant", "content": '{"commands":[],"task_complete":true}'},
            {"role": "user", "content": "Confirm task completion."},
            {"role": "user", "content": "Summarize the work for the next agent."},
            {"role": "assistant", "content": '{"analysis":"Work summary.","commands":[],"task_complete":true}'},
        ]
    }
    [document] = agenttrove_row_to_chat_doc(row)
    normalized = _normalize_chat_record(document, "messages", "id")
    user_turns = [message for message in normalized["messages"] if message["role"] == "user"]
    assert user_turns[-1]["content"] == [
        {"type": "text", "text": "Confirm task completion.\n\nSummarize the work for the next agent."}
    ]


def test_agenttrove_preserves_tool_observation_before_user_followup() -> None:
    row = {
        "conversations": [
            {"role": "user", "content": "Task Description:\nInspect the repository."},
            {"role": "assistant", "content": '{"commands":[{"keystrokes":"ls\\n"}]}'},
            {"role": "user", "content": "New Terminal Output:\nREADME.md"},
            {"role": "user", "content": "Summarize the work for the next agent."},
            {"role": "assistant", "content": '{"analysis":"README found.","commands":[],"task_complete":true}'},
        ]
    }
    [document] = agenttrove_row_to_chat_doc(row)
    normalized = _normalize_chat_record(document, "messages", "id")
    turns = [
        (message["role"], message["content"][0]["text"])
        for message in normalized["messages"]
        if message["role"] in {"user", "tool"}
    ]
    assert turns == [
        ("user", "Task Description:\nInspect the repository."),
        ("tool", "New Terminal Output:\nREADME.md"),
        ("user", "Summarize the work for the next agent."),
    ]


def test_davinci_filters_source_control_tokens() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Explain <|eot_id|>."},
            {"role": "assistant", "content": "It ends a turn."},
        ],
    }
    assert davinci_row_to_chat_doc(row) == []


def test_coderforge_filters_protocol_wrapped_tool_observations() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Read the repository."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "read", "arguments": {}}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "<tool_response>contents</tool_response>"},
            {"role": "assistant", "content": "Done."},
        ],
    }
    assert coderforge_row_to_chat_doc(row) == []


def test_coderforge_keeps_unsuccessful_trajectory_without_visible_outcome() -> None:
    row = {
        "reward": 0.0,
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "I could not fix it."},
        ],
    }

    [document] = coderforge_row_to_chat_doc(row)
    assert [(m["role"], m["content"][0]["text"]) for m in document["messages"]] == [
        (m["role"], m["content"]) for m in row["messages"]
    ]
    assert document["reward"] == 0.0


def test_davinci_drops_terminal_submit_observation() -> None:
    row = {
        "success": True,
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "The fix is complete.",
                "tool_calls": [
                    {
                        "id": "submit_1",
                        "type": "function",
                        "function": {"name": "submit", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "diff --git a/app.py b/app.py",
                "name": "submit",
                "tool_call_id": "submit_1",
            },
        ],
    }

    [document] = davinci_row_to_chat_doc(row)
    assert [message["role"] for message in document["messages"]] == ["user", "assistant", "assistant"]
    assert document["success"] is True
    assert document["messages"][-1]["recipient"] == "functions.submit"


def test_davinci_drops_nonterminal_trailing_observation() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "bash_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"command": "pytest"}},
                    }
                ],
            },
            {"role": "tool", "content": "tests failed", "name": "bash", "tool_call_id": "bash_1"},
        ]
    }

    assert davinci_row_to_chat_doc(row) == []


def test_gpt_oss_drops_embedded_tokenizer_control_tokens() -> None:
    row = {
        "user_content": 'What does tokenizer.convert_tokens_to_ids("<|eot_id|>") return?',
        "assistant_content": "It returns the end-of-turn token ID.",
    }

    assert gpt_oss_row_to_chat_doc(row) == []


def test_openhands_merges_adjacent_user_context() -> None:
    row = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": None,
                        "properties": {
                            "command": {"type": "string", "enum": None},
                            "unused_arrow_field": None,
                        },
                        "required": ["command"],
                    },
                },
            }
        ],
        "trajectory": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "user", "content": "Repository: example/project"},
            {"role": "assistant", "content": "I fixed it."},
        ],
    }

    [document] = openhands_row_to_chat_doc(row)
    assert [message["role"] for message in document["messages"]] == ["user", "assistant"]
    assert document["messages"][0]["content"] == [
        {"type": "text", "text": "Fix the bug.\n\nRepository: example/project"}
    ]

    [tool] = json.loads(document["chat_template_kwargs"])["tools"]
    assert tool["function"]["parameters"] == {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }


def test_openhands_filters_unclosed_inline_tool_calls() -> None:
    row = {
        "trajectory": [
            {"role": "user", "content": "Inspect the repository."},
            {"role": "assistant", "content": '<tool_call>{"name":"execute_bash"'},
        ],
    }
    assert openhands_row_to_chat_doc(row) == []


def test_swe_zero_filters_protocol_wrapped_observations() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Inspect the repository."},
            {"role": "assistant", "content": "THOUGHT: Read it.\n```bash\ncat README.md\n```"},
            {"role": "user", "content": "Observation: <tool_response>contents</tool_response>"},
            {
                "role": "assistant",
                "content": "THOUGHT: Done.\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            },
        ],
    }
    assert swe_zero_row_to_chat_doc(row) == []


def test_swe_zero_converts_reasoning_bash_and_observation() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Respond with THOUGHT and a bash block."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "THOUGHT: Inspect the file.\n\n```bash\nsed -n '1,80p' app.py\n```",
            },
            {"role": "user", "content": "Observation: file contents"},
            {
                "role": "assistant",
                "content": "THOUGHT: The edit is complete.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            },
            {"role": "user", "content": "Observation: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
        ]
    }

    [document] = swe_zero_row_to_chat_doc(row)
    messages = document["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "assistant",
    ]
    assert messages[2]["channel"] == "analysis"
    assert messages[2]["content"] == [{"type": "text", "text": "Inspect the file."}]
    assert messages[3]["recipient"] == "functions.bash"
    assert messages[3]["content"] == [{"type": "text", "text": '{"command":"sed -n \'1,80p\' app.py"}'}]
    assert messages[4]["content"] == [{"type": "text", "text": "file contents"}]
    assert messages[5]["channel"] == "analysis"
    assert messages[5]["content"] == [{"type": "text", "text": "The edit is complete."}]
    assert messages[-1]["channel"] == "final"
    assert messages[-1]["content"] == [{"type": "text", "text": "Task complete."}]


def test_swe_zero_does_not_treat_inspecting_completion_marker_as_completion() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": (
                    "THOUGHT: Find the protocol.\n\n```bash\ngrep COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT README.md\n```"
                ),
            },
            {"role": "user", "content": "Observation: README.md: completion instructions"},
            {
                "role": "assistant",
                "content": "THOUGHT: Done.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            },
            {"role": "user", "content": "Observation: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
        ]
    }

    [document] = swe_zero_row_to_chat_doc(row)
    messages = document["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "assistant", "tool", "assistant", "assistant"]
    assert messages[2]["recipient"] == "functions.bash"
    assert "grep COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in messages[2]["content"][0]["text"]


def test_numinamath_splits_reasoning_python_and_output() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Compute it."},
            {
                "role": "assistant",
                "content": (
                    "I should calculate it.\n\n```python\nprint(6 * 7)\n```\n"
                    "```output\n42\n```\nTherefore, the answer is 42."
                ),
            },
        ]
    }

    [document] = numinamath_row_to_chat_doc(row)
    messages = document["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "assistant", "tool", "assistant"]
    assert messages[1]["channel"] == "analysis"
    assert messages[1]["content"] == [{"type": "text", "text": "I should calculate it."}]
    assert messages[2]["recipient"] == "functions.python"
    assert messages[2]["content"] == [{"type": "text", "text": '{"code":"print(6 * 7)"}'}]
    assert messages[3]["content"] == [{"type": "text", "text": "42"}]
    assert messages[4]["channel"] == "final"
    assert messages[4]["content"] == [{"type": "text", "text": "Therefore, the answer is 42."}]
