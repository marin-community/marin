# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.datakit.chat_normalize import InvalidToolCallPolicy, _normalize_chat_record
from marin.datakit.chat_render import chat_training_record, render_chat_record
from marin.datakit.download.rollout_transforms import (
    LiteralToolCallFormat,
    ToolCallLiteralFormatError,
    normalize_tool_call_literals,
    openai_chat_document,
    openai_chat_messages,
    render_tool_call,
    render_tool_message,
)
from openai_harmony import Role


def test_qwen_literal_calls_use_shared_marin_serialization_and_retain_other_text():
    quoted = "<tool_call><function=example></function></tool_call>"
    reasoning = f"<|start_think|>Quoted example: {quoted}<|end_think|>\n"
    first = "<tool_call><function=lookup><parameter=query>first</parameter></function></tool_call>"
    parallel = (
        "<tool_call><function=lookup><parameter=query>second</parameter></function>"
        "<function=finish><parameter=value>done</parameter></function></tool_call>"
    )
    calls = [
        {"id": "a", "function": {"name": "lookup", "arguments": {"query": "first"}}},
        {"id": "b", "function": {"name": "lookup", "arguments": {"query": "second"}}},
        {"id": "c", "function": {"name": "finish", "arguments": {"value": "done"}}},
    ]
    source = reasoning + first + "\nBETWEEN_CALLS\n" + parallel + "\nFINAL_TEXT"
    normalized = normalize_tool_call_literals(source, calls, source_format=LiteralToolCallFormat.QWEN3_CODER)
    assert normalized.startswith(reasoning)
    assert "\nBETWEEN_CALLS\n" in normalized and normalized.endswith("\nFINAL_TEXT")
    bodies = normalized.removeprefix(reasoning).split("<tool_call>")[1:]
    parsed = [json.loads(body.split("</tool_call>", 1)[0]) for body in bodies]
    assert parsed == [{"name": call["function"]["name"], "arguments": call["function"]["arguments"]} for call in calls]


def test_hermes_negative_literal_is_not_rewritten_as_a_valid_tool_call():
    malformed = "<tool_call><function=wrong>RAW_LOOP RAW_LOOP</tool_call>"
    calls = [{"id": "a", "function": {"name": "wrong", "arguments": "UNFINISHED_JSON"}}]
    assert normalize_tool_call_literals(malformed, calls, source_format=LiteralToolCallFormat.HERMES) == malformed


def test_qwen_literal_mapping_rejects_an_unmatched_function_before_serialization():
    source = "<tool_call><function=lookup><parameter=query>raw</parameter></function></tool_call>"
    calls = [{"id": "a", "function": {"name": "different", "arguments": {"query": "raw"}}}]
    with pytest.raises(ToolCallLiteralFormatError, match="names differ"):
        normalize_tool_call_literals(source, calls, source_format=LiteralToolCallFormat.QWEN3_CODER)


@pytest.mark.parametrize("name", ["apply_patch", "cat /app/result.json"])
def test_offline_negative_calls_preserve_malformed_arguments_and_unparsed_text(name: str):
    tools = [{"type": "function", "function": {"name": "exec_command", "parameters": {"type": "object"}}}]
    malformed = '{"patch": "UNFINISHED'
    loop = "<tool_call>\n$LANG$LANG$LANG"
    messages = [
        {"role": "user", "content": "Write the answer."},
        {
            "role": "assistant",
            "tool_calls": [{"id": "bad", "function": {"name": name, "arguments": malformed}}],
        },
        {"role": "tool", "tool_call_id": "bad", "content": "Unknown tool apply_patch"},
        {"role": "assistant", "content": "", "unparsed_content": loop},
    ]
    with pytest.raises(ValueError):
        openai_chat_document(messages, "negative", chat_template_kwargs={"tools": tools})
    document = openai_chat_document(
        messages,
        "negative",
        invalid_tool_call_policy=InvalidToolCallPolicy.RETAIN,
        chat_template_kwargs={"tools": tools},
    )
    normalized = _normalize_chat_record(
        document, "messages", "id", invalid_tool_call_policy=InvalidToolCallPolicy.RETAIN
    )
    chat = chat_training_record(normalized)
    assert chat["chat_template_kwargs"]["tools"] == tools
    assert chat["messages"][1]["tool_calls"][0]["function"] == {"name": name, "arguments": malformed}
    assert chat["messages"][-1]["content"] == loop
    rendered = render_chat_record(normalized)["text"]
    assert malformed in rendered and loop in rendered and "Unknown tool apply_patch" in rendered
    with pytest.raises(ValueError, match=r"Tool-call arguments must encode a JSON object|valid tool name"):
        _normalize_chat_record(document, "messages", "id")


def test_source_aliases_and_tool_results_become_harmony_messages():
    messages = openai_chat_messages(
        [
            {"from": "human", "value": "Inspect the repository."},
            {
                "from": "gpt",
                "value": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "bash", "arguments": {"cmd": "pwd"}}}],
            },
            {"role": "function", "content": "/workspace", "tool_call_id": "call_1"},
        ]
    )
    assert [message.author.role for message in messages] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert messages[1].recipient == "functions.bash"
    assert json.loads(messages[1].content[0].to_dict()["text"]) == {"cmd": "pwd"}
    assert messages[2].author.name == "functions.bash"
    assert messages[2].content[0].to_dict()["text"] == "/workspace"


def test_legacy_function_call_becomes_harmony_recipient_and_arguments():
    [message] = openai_chat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "function_call": '{"name":"search","arguments":{"query":"marin"}}',
                "provider_metadata": {"shape": "varies"},
            }
        ]
    )
    assert message.to_dict() == {
        "role": "assistant",
        "name": None,
        "channel": "commentary",
        "recipient": "functions.search",
        "content": [{"type": "text", "text": '{"query":"marin"}'}],
    }


def test_text_content_blocks_preserve_native_instructions_and_tool_history():
    source = [
        {"role": "system", "content": "First instruction.\nSecond instruction."},
        {"role": "user", "content": "Inspect the file."},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [{"id": "one", "function": {"name": "read", "arguments": {"path": "a.py"}}}],
        },
        {"role": "tool", "tool_call_id": "one", "content": "File contents."},
        {"role": "assistant", "content": "<think>Checked.</think>Done."},
    ]
    blocks = [
        {**message, "content": [{"type": "text", "text": word} for word in message["content"].splitlines(True)]}
        for message in source
    ]
    assert [message.to_dict() for message in openai_chat_messages(blocks)] == [
        message.to_dict() for message in openai_chat_messages(source)
    ]


def test_nontext_content_block_is_rejected_without_discarding_context():
    with pytest.raises(ValueError, match="text content blocks"):
        openai_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this image."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ]
        )


def test_response_text_item_before_tool_observation_stays_in_same_assistant_turn():
    messages = openai_chat_messages(
        [
            {"role": "user", "content": "Workspace instructions."},
            {"role": "user", "content": "Inspect the file."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "one", "function": {"name": "read", "arguments": {"path": "a.py"}}}],
            },
            {"role": "assistant", "content": "Check the file.</think>Reading now."},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "tool", "tool_call_id": "one", "content": "File contents."},
            {"role": "assistant", "content": "Done."},
        ],
        assistant_prefill="<think>\n",
    )
    assert [(message.channel, message.recipient) for message in messages] == [
        (None, None),
        ("analysis", None),
        ("commentary", None),
        ("commentary", "functions.read"),
        ("commentary", "assistant"),
        ("final", None),
    ]
    assert messages[0].content[0].to_dict()["text"] == "Workspace instructions.\n\nInspect the file."
    assert messages[1].content[0].to_dict()["text"] == "Check the file."
    assert messages[2].content[0].to_dict()["text"] == "Reading now."


@pytest.mark.parametrize("content", ["<think>unfinished", "<think>a</think>answer<think>b</think>", "a</think>answer"])
def test_source_adapter_rejects_malformed_reasoning(content):
    with pytest.raises(ValueError):
        openai_chat_messages([{"role": "assistant", "content": content}])


@pytest.mark.parametrize("content", ["reasoning</think>answer", "<think>reasoning</think>answer"])
def test_explicit_reasoning_prefill_preserves_analysis_and_final_channels(content):
    messages = openai_chat_messages([{"role": "assistant", "content": content}], assistant_prefill="<think>\n")
    assert [(message.channel, message.content[0].to_dict()["text"]) for message in messages] == [
        ("analysis", "reasoning"),
        ("final", "answer"),
    ]


def test_source_adapter_rejects_repeated_call_ids_before_discarding_them():
    with pytest.raises(ValueError, match="unique"):
        openai_chat_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "same", "function": {"name": "read", "arguments": {"path": "a"}}},
                        {"id": "same", "function": {"name": "read", "arguments": {"path": "b"}}},
                    ],
                }
            ]
        )


def test_render_tool_call_dict_arguments():
    tool_call = {"function": {"name": "bash", "arguments": {"cmd": "ls", "dir": "/tmp"}}}
    assert render_tool_call(tool_call) == "<tool_call:bash>\n  cmd: ls\n  dir: /tmp\n</tool_call:bash>"


def test_render_tool_call_json_string_arguments():
    tool_call = {"function": {"name": "edit", "arguments": '{"path": "a.py"}'}}
    assert render_tool_call(tool_call) == "<tool_call:edit>\n  path: a.py\n</tool_call:edit>"


def test_render_tool_call_malformed_json_kept_as_raw_string():
    # A tool call whose arguments are an unparseable string must not abort the transform.
    tool_call = {"function": {"name": "run", "arguments": "not json"}}
    assert render_tool_call(tool_call) == "<tool_call:run>\n  not json\n</tool_call:run>"


def test_render_tool_message_includes_content_and_tool_calls():
    message = {
        "role": "assistant",
        "content": "checking",
        "tool_calls": [{"function": {"name": "ls", "arguments": {"dir": "/tmp"}}}],
    }
    assert render_tool_message(message) == (
        "<assistant>\nchecking\n<tool_call:ls>\n  dir: /tmp\n</tool_call:ls>\n</assistant>"
    )


def test_render_tool_message_omits_empty_content_line():
    assert render_tool_message({"role": "user", "content": ""}) == "<user>\n</user>"
