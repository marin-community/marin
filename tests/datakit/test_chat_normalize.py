# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.chat_normalize import (
    ChatChannel,
    _normalize_chat_record,
    normalize_chat_to_parquet,
    validate_chat_messages,
)
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.coderforge import SOURCE_CHAT_SCHEMA
from marin.datakit.download.coderforge import transform_chat as transform_coderforge_chat
from openai_harmony import Author, Message, Role


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def test_normalization_preserves_native_harmony_channels_without_interpreting_text():
    messages = [
        Message.from_role_and_content(Role.USER, "Explain the <think> syntax."),
        Message.from_role_and_content(Role.ASSISTANT, "Check the syntax.").with_channel(ChatChannel.ANALYSIS),
        Message.from_role_and_content(Role.ASSISTANT, "THOUGHT: is a literal prefix.").with_channel(
            ChatChannel.COMMENTARY
        ),
        Message.from_role_and_content(Role.ASSISTANT, "<think> marks reasoning in some source formats.").with_channel(
            ChatChannel.FINAL
        ),
    ]
    record = {"messages": [message.to_dict() for message in messages]}
    normalized = _normalize_chat_record(record, "messages", "id")
    assert normalized["messages"] == record["messages"]
    assert _normalize_chat_record(normalized, "messages", "id")["id"] == normalized["id"]


@pytest.mark.parametrize(
    ("analysis", "expected_mode", "expected_instruction"),
    [(None, False, "Reasoning: /nothink"), ("Check the answer.", True, "Reasoning: /think")],
)
def test_normalization_derives_conversation_mode_from_analysis(analysis, expected_mode, expected_instruction):
    messages = [Message.from_role_and_content(Role.USER, "What is two plus two?")]
    if analysis is not None:
        messages.append(Message.from_role_and_content(Role.ASSISTANT, analysis).with_channel(ChatChannel.ANALYSIS))
    messages.append(Message.from_role_and_content(Role.ASSISTANT, "Four.").with_channel(ChatChannel.FINAL))
    record = {
        "messages": [message.to_dict() for message in messages],
        "chat_template_kwargs": {"enable_thinking": not expected_mode},
    }

    normalized = _normalize_chat_record(record, "messages", "id")

    assert json.loads(normalized["chat_template_kwargs"])["enable_thinking"] is expected_mode
    assert expected_instruction in render_chat_record(normalized)["text"]


def test_normalization_rejects_legacy_source_turns():
    with pytest.raises(ValueError, match="Source adapters must emit Harmony"):
        _normalize_chat_record(
            {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]},
            "messages",
            "id",
        )


def test_chat_identity_includes_tool_definitions():
    messages = [
        Message.from_role_and_content(Role.USER, "Run it."),
        Message.from_role_and_content(Role.ASSISTANT, "{}")
        .with_channel(ChatChannel.COMMENTARY)
        .with_recipient("functions.run"),
    ]
    ids = [
        _normalize_chat_record(
            {
                "messages": [message.to_dict() for message in messages],
                "chat_template_kwargs": {
                    "tools": [{"name": "run", "description": description, "parameters": {"type": "object"}}]
                },
            },
            "messages",
            "id",
        )["id"]
        for description in ["First", "Second"]
    ]
    assert ids[0] != ids[1]


def test_harmony_tool_handoff_requires_matching_observations_before_continuation():
    user = Message.from_role_and_content(Role.USER, "Read the file.")
    call = (
        Message.from_role_and_content(Role.ASSISTANT, '{"path":"a.py"}')
        .with_channel(ChatChannel.COMMENTARY)
        .with_recipient("functions.read")
    )
    observation = (
        Message.from_author_and_content(Author.new(Role.TOOL, "functions.read"), "contents")
        .with_channel(ChatChannel.COMMENTARY)
        .with_recipient("assistant")
    )
    final = Message.from_role_and_content(Role.ASSISTANT, "Done.").with_channel(ChatChannel.FINAL)
    record = {"messages": [message.to_dict() for message in [user, call, observation, final]]}
    with pytest.raises(ValueError, match="no explicit definition"):
        _normalize_chat_record(record, "messages", "id")
    record["chat_template_kwargs"] = {
        "tools": [{"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]
    }
    normalized = _normalize_chat_record(record, "messages", "id")
    assert normalized["messages"] == record["messages"]
    assert json.loads(normalized["chat_template_kwargs"])["tools"][0]["name"] == "read"
    with pytest.raises(ValueError, match="observation"):
        validate_chat_messages([user, call, final])
    wrong_observation = (
        Message.from_author_and_content(Author.new(Role.TOOL, "functions.write"), "done")
        .with_channel(ChatChannel.COMMENTARY)
        .with_recipient("assistant")
    )
    with pytest.raises(ValueError, match="match pending calls"):
        validate_chat_messages([user, call, wrong_observation, final])


def test_normalization_filters_repeated_tool_call_after_identical_replies(tmp_path: Path):
    def call(arguments: str) -> Message:
        return (
            Message.from_role_and_content(Role.ASSISTANT, arguments)
            .with_channel(ChatChannel.COMMENTARY)
            .with_recipient("functions.search")
        )

    def reply(text: str) -> Message:
        return (
            Message.from_author_and_content(Author.new(Role.TOOL, "functions.search"), text)
            .with_channel(ChatChannel.COMMENTARY)
            .with_recipient("assistant")
        )

    repeated = [
        Message.from_role_and_content(Role.USER, "Search again."),
        call('{"query":"x","limit":1}'),
        reply("same result"),
        call('{"limit":1,"query":"x"}'),
        reply("same result"),
        call('{"query":"x","limit":1}'),
    ]
    clean = [
        Message.from_role_and_content(Role.USER, "Hello."),
        Message.from_role_and_content(Role.ASSISTANT, "Hi.").with_channel(ChatChannel.FINAL),
    ]
    records = [
        {"messages": [message.to_dict() for message in clean]},
        {
            "messages": [message.to_dict() for message in repeated],
            "chat_template_kwargs": {"tools": [{"name": "search", "parameters": {"type": "object"}}]},
        },
    ]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "data.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

    result = normalize_chat_to_parquet(input_path=str(input_dir), output_path=str(tmp_path / "normalized"))

    normalized = [
        row
        for path in (tmp_path / "normalized" / "outputs" / "main").glob("*.parquet")
        for row in pq.read_table(path).to_pylist()
    ]
    assert len(normalized) == 1
    assert normalized[0]["messages"][0]["content"][0]["text"] == "Hello."
    assert result.counters["normalize_chat/repeated_tool_calls_filtered"] == 1
    assert result.counters.get("normalize_chat/records_quarantined", 0) == 0


def test_normalization_counts_conversations_with_long_final_responses_once(tmp_path: Path):
    messages = [
        Message.from_role_and_content(Role.USER, "First question"),
        Message.from_role_and_content(Role.ASSISTANT, "word " * 2_001).with_channel(ChatChannel.FINAL),
        Message.from_role_and_content(Role.USER, "Second question"),
        Message.from_role_and_content(Role.ASSISTANT, "word " * 2_001).with_channel(ChatChannel.FINAL),
    ]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "data.jsonl").write_text(json.dumps({"messages": [message.to_dict() for message in messages]}))

    result = normalize_chat_to_parquet(input_path=str(input_dir), output_path=str(tmp_path / "normalized"))

    assert result.counters["normalize_chat/conversations_with_final_over_2k_estimated_tokens"] == 1


@pytest.mark.parametrize("replies", [("43% complete", "65% complete"), ("same result", "same result")])
def test_tool_call_repetition_requires_unchanged_feedback_and_sequential_calls(replies):
    user = Message.from_role_and_content(Role.USER, "Search.")
    call = (
        Message.from_role_and_content(Role.ASSISTANT, '{"query":"x"}')
        .with_channel(ChatChannel.COMMENTARY)
        .with_recipient("functions.search")
    )
    observations = [
        Message.from_author_and_content(Author.new(Role.TOOL, "functions.search"), text)
        .with_channel(ChatChannel.COMMENTARY)
        .with_recipient("assistant")
        for text in replies
    ]
    messages = [user, call, observations[0], call, observations[1], call]
    if replies[0] == replies[1]:
        messages = [
            user,
            call,
            call,
            call,
            observations[0],
            observations[1],
            observations[0],
            Message.from_role_and_content(Role.ASSISTANT, "Done.").with_channel(ChatChannel.FINAL),
        ]
    record = {
        "messages": [message.to_dict() for message in messages],
        "chat_template_kwargs": {"tools": [{"name": "search", "parameters": {"type": "object"}}]},
    }

    assert _normalize_chat_record(record, "messages", "id")["messages"] == record["messages"]


@pytest.mark.parametrize(
    "tail",
    [
        Message.from_role_and_content(Role.ASSISTANT, "answer"),
        Message.from_role_and_content(Role.ASSISTANT, "answer").with_channel("unknown"),
        Message.from_role_and_content(Role.ASSISTANT, "{}")
        .with_channel(ChatChannel.FINAL)
        .with_recipient("functions.run"),
        Message.from_role_and_content(Role.ASSISTANT, "[]")
        .with_channel(ChatChannel.COMMENTARY)
        .with_recipient("functions.run"),
        Message.from_role_and_content(Role.ASSISTANT, "answer")
        .with_channel(ChatChannel.FINAL)
        .with_recipient("unknown"),
    ],
)
def test_harmony_validation_rejects_invalid_channels_and_calls(tail):
    with pytest.raises(ValueError):
        validate_chat_messages([Message.from_role_and_content(Role.USER, "Question"), tail])


@pytest.mark.parametrize("channel", [ChatChannel.ANALYSIS, ChatChannel.COMMENTARY])
def test_harmony_validation_requires_final_or_tool_call_at_record_end(channel):
    messages = [
        Message.from_role_and_content(Role.USER, "Question"),
        Message.from_role_and_content(Role.ASSISTANT, "Still working").with_channel(channel),
    ]
    with pytest.raises(ValueError, match="final answer or tool call"):
        validate_chat_messages(messages)


@pytest.mark.parametrize("channel", [ChatChannel.ANALYSIS, ChatChannel.COMMENTARY])
def test_harmony_validation_requires_final_before_next_user(channel):
    messages = [
        Message.from_role_and_content(Role.USER, "First question"),
        Message.from_role_and_content(Role.ASSISTANT, "Still working").with_channel(channel),
        Message.from_role_and_content(Role.USER, "Second question"),
        Message.from_role_and_content(Role.ASSISTANT, "Answer").with_channel(ChatChannel.FINAL),
    ]
    with pytest.raises(ValueError, match="final answer before the next user turn"):
        validate_chat_messages(messages)


def test_normalize_chat_to_parquet_keeps_varying_tool_schemas_arrow_stable(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    records = []
    for name, arguments in (("read", {"path": "a.py"}), ("search", {"query": "Marin", "limit": 3})):
        messages = [
            Message.from_role_and_content(Role.USER, f"Use {name}."),
            Message.from_role_and_content(Role.ASSISTANT, json.dumps(arguments))
            .with_channel(ChatChannel.COMMENTARY)
            .with_recipient(f"functions.{name}"),
        ]
        records.append(
            {
                "messages": [message.to_dict() for message in messages],
                "chat_template_kwargs": {"tools": [{"name": name, "parameters": {"type": "object"}}]},
            }
        )
    (input_dir / "data.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    normalize_chat_to_parquet(input_path=str(input_dir), output_path=str(output_dir))
    normalized = [
        row for path in (output_dir / "outputs" / "main").glob("*.parquet") for row in pq.read_table(path).to_pylist()
    ]
    calls = [Message.from_dict(row["messages"][-1]) for row in normalized]
    assert {call.recipient: json.loads(call.content[0].to_dict()["text"]) for call in calls} == {
        "functions.read": {"path": "a.py"},
        "functions.search": {"query": "Marin", "limit": 3},
    }
    assert all(call.channel == ChatChannel.COMMENTARY for call in calls)
    assert {json.loads(row["chat_template_kwargs"])["tools"][0]["name"] for row in normalized} == {"read", "search"}


@pytest.mark.parametrize("valid_count", [20, 1])
def test_normalization_quarantines_bad_harmony_and_enforces_source_health(tmp_path: Path, valid_count: int):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    valid = {
        "messages": [
            Message.from_role_and_content(Role.USER, "Question").to_dict(),
            Message.from_role_and_content(Role.ASSISTANT, "Answer").with_channel(ChatChannel.FINAL).to_dict(),
        ]
    }
    invalid = {"messages": [*valid["messages"], valid["messages"][-1]]}
    (input_dir / "data.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in [*[valid] * valid_count, invalid])
    )
    if valid_count == 1:
        with pytest.raises(ValueError, match="above the 5% health limit"):
            normalize_chat_to_parquet(input_path=str(input_dir), output_path=str(output_dir))
        return
    result = normalize_chat_to_parquet(input_path=str(input_dir), output_path=str(output_dir))
    normalized = [
        row for path in (output_dir / "outputs" / "main").glob("*.parquet") for row in pq.read_table(path).to_pylist()
    ]
    assert len(normalized) == 1
    assert result.counters["normalize_chat/records_validated"] == valid_count
    assert result.counters["normalize_chat/records_quarantined"] == 1


def test_source_writer_preserves_tool_fields_first_seen_after_plain_conversations(tmp_path: Path):
    input_dir = tmp_path / "input"
    processed_dir = tmp_path / "processed"
    normalized_dir = tmp_path / "normalized"
    input_dir.mkdir()
    records = []
    for index in range(9):
        assistant = {"role": "assistant", "content": "Answer"}
        if index == 8:
            assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "run", "arguments": {"command": "ls"}}},
                ],
            }
        records.append(
            {
                "messages": json.dumps(
                    [
                        {"role": "user", "content": f"Question {index}"},
                        assistant,
                    ]
                ),
                "reward": 0.75,
                "tools": json.dumps(
                    [
                        {
                            "type": "function",
                            "function": {
                                "name": "run",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
                                    "required": ["command"],
                                },
                            },
                        }
                    ]
                ),
            }
        )
    pq.write_table(pa.Table.from_pylist(records), input_dir / "data.parquet")
    transform_coderforge_chat(str(input_dir), str(processed_dir))
    processed = [row for path in processed_dir.glob("*.parquet") for row in pq.read_table(path).to_pylist()]
    call = next(row for row in processed if row["messages"][0]["content"][0]["text"] == "Question 8")
    assert call["messages"][-1]["recipient"] == "functions.run"
    assert call["messages"][-1]["content"] == [{"type": "text", "text": '{"command":"ls"}'}]
    normalize_chat_to_parquet(
        input_path=str(processed_dir), output_path=str(normalized_dir), output_schema=SOURCE_CHAT_SCHEMA
    )
    normalized = [
        row
        for path in (normalized_dir / "outputs" / "main").glob("*.parquet")
        for row in pq.read_table(path).to_pylist()
    ]
    normalized_call = next(row for row in normalized if row["messages"][0]["content"][0]["text"] == "Question 8")
    assert normalized_call["messages"][-1]["recipient"] == "functions.run"
    assert normalized_call["reward"] == 0.75
    [tool] = json.loads(normalized_call["chat_template_kwargs"])["tools"]
    assert tool["function"]["parameters"]["properties"]["timeout"] == {"type": "number"}
    assert tool["function"]["parameters"]["required"] == ["command"]


@pytest.mark.parametrize("source_id", [None, "upstream-record-7"])
def test_normalization_retains_source_provenance_across_repeated_normalization(source_id):
    record = {
        "id": "intermediate-content-hash",
        "messages": [
            Message.from_role_and_content(Role.USER, "Hello").to_dict(),
            Message.from_role_and_content(Role.ASSISTANT, "Hi").with_channel(ChatChannel.FINAL).to_dict(),
        ],
    }
    if source_id is not None:
        record["source_id"] = source_id
    normalized = _normalize_chat_record(record, "messages", "id")
    assert normalized["source_id"] == (source_id or "intermediate-content-hash")
    assert _normalize_chat_record(normalized, "messages", "id")["source_id"] == normalized["source_id"]
