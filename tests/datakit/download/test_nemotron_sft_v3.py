# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from itertools import pairwise

import msgspec
import pytest
from marin.datakit.download import nemotron_chat_prompts
from marin.datakit.download.nemotron_chat_prompts import restore_chat_row
from marin.datakit.download.nemotron_sft_v3 import (
    _JsonlByteRange,
    _load_jsonl_byte_range,
    load_jsonl_with_skips,
    row_to_chat_doc,
)


def test_v3_chat_reconstruction_preserves_the_original_seed_prompt():
    prompt = "Explain this TypeScript error."
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    row = {
        "uuid": "wildchat-example",
        "metadata": {"seed_dataset": "allenai/WildChat-1M", "seed_prompt_sha256": digest},
        "messages": [
            {"role": "system", "content": None},
            {"role": "user", "content": None},
            {"role": "assistant", "content": "The API changed."},
        ],
    }

    restored = restore_chat_row(row, {("allenai/WildChat-1M", digest): ("You are helpful.", prompt)})

    assert row["messages"][1]["content"] is None
    assert restored["messages"][:2] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": prompt},
    ]
    assert restore_chat_row(restored, {}) == restored


def test_v3_chat_reconstruction_removes_absent_system_turn():
    row = {
        "metadata": {
            "seed_dataset": "allenai/WildChat-1M",
            "seed_prompt_sha256": "digest",
            "train_turns": [False, False, True],
        },
        "messages": [
            {"role": "system", "content": None},
            {"role": "user", "content": None},
            {"role": "assistant", "content": "Answer"},
        ],
    }

    restored = restore_chat_row(row, {("allenai/WildChat-1M", "digest"): (None, "Original prompt")})

    assert [message["role"] for message in restored["messages"]] == ["user", "assistant"]
    assert restored["metadata"]["train_turns"] == [False, True]


def test_v3_chat_enrichment_joins_seed_conversations(tmp_path, monkeypatch):
    prompt = "Why does useHistory fail?"
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    source = tmp_path / "input" / "data" / "chat.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "uuid": "seeded-chat",
                "metadata": {"seed_dataset": "allenai/WildChat-1M", "seed_prompt_sha256": digest},
                "messages": [{"role": "system", "content": None}, {"role": "user", "content": None}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        nemotron_chat_prompts,
        "load_dataset",
        lambda _dataset, **_kwargs: [
            {"conversation": [{"role": "system", "content": "Help with coding."}, {"role": "user", "content": prompt}]}
        ],
    )

    nemotron_chat_prompts.restore_chat_prompts(str(tmp_path / "input"), str(tmp_path / "output"))

    restored = json.loads((tmp_path / "output" / "data" / "chat.jsonl").read_text())
    assert [message["content"] for message in restored["messages"]] == ["Help with coding.", prompt]


def test_v3_chat_enrichment_skips_null_source_user_turn(tmp_path, monkeypatch):
    prompt = "First prompt with content"
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    source = tmp_path / "input" / "data" / "chat.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "uuid": "seeded-chat",
                "metadata": {"seed_dataset": "allenai/WildChat-1M", "seed_prompt_sha256": digest},
                "messages": [{"role": "user", "content": None}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        nemotron_chat_prompts,
        "load_dataset",
        lambda _dataset, **_kwargs: [
            {
                "conversation": [
                    {"role": "user", "content": None},
                    {"role": "assistant", "content": "Answer"},
                    {"role": "user", "content": prompt},
                ]
            }
        ],
    )

    nemotron_chat_prompts.restore_chat_prompts(str(tmp_path / "input"), str(tmp_path / "output"))

    restored = json.loads((tmp_path / "output" / "data" / "chat.jsonl").read_text())
    assert restored["messages"][0]["content"] == prompt


def test_v3_chat_enrichment_excludes_unrecoverable_prompts(tmp_path, monkeypatch):
    prompt = "Available prompt"
    found_digest = hashlib.sha256(prompt.encode()).hexdigest()
    missing_digest = hashlib.sha256(b"Withheld from public source").hexdigest()
    rows = [
        {
            "uuid": uuid,
            "metadata": {"seed_dataset": "allenai/WildChat-1M", "seed_prompt_sha256": digest},
            "messages": [{"role": "user", "content": None}],
        }
        for uuid, digest in [("found", found_digest), ("missing-1", missing_digest), ("missing-2", missing_digest)]
    ]
    rows.append({"uuid": "unprotected", "messages": [{"role": "user", "content": "Keep this"}]})
    source = tmp_path / "input" / "data" / "chat.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        nemotron_chat_prompts,
        "load_dataset",
        lambda _dataset, **_kwargs: [{"conversation": [{"role": "user", "content": prompt}]}],
    )

    nemotron_chat_prompts.restore_chat_prompts(str(tmp_path / "input"), str(tmp_path / "output"))

    output = tmp_path / "output"
    restored = [json.loads(line) for line in (output / "data" / "chat.jsonl").read_text().splitlines()]
    assert [row["uuid"] for row in restored] == ["found", "unprotected"]
    assert restored[0]["messages"][0]["content"] == prompt
    assert json.loads((output / "restoration_report.json").read_text()) == {
        "source_rows": 4,
        "written_rows": 2,
        "excluded_rows": {"allenai/WildChat-1M": 2, "lmsys/lmsys-chat-1m": 0},
        "missing_prompt_hashes": {"allenai/WildChat-1M": 1, "lmsys/lmsys-chat-1m": 0},
    }


def test_v3_chat_retains_source_training_turn_annotation():
    row = {
        "uuid": "chat-example",
        "messages": [
            {"role": "user", "content": "Initial prompt"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Follow up"},
            {"role": "assistant", "reasoning_content": "I should clarify.", "content": "Final answer"},
        ],
        "metadata": {"train_turns": [False, False, False, True]},
    }

    documents = row_to_chat_doc(row, family="instruction_following_chat_v3", partition_name="chat")

    assert len(documents) == 1
    document = documents[0]
    assert [message["role"] for message in document["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "assistant",
    ]
    assert document["source_train_turns"] == [False, False, False, True]
    assert document["source_id"] == "chat-example"


def test_v2_anonymized_name_is_retained():
    row = {"messages": [{"role": "user", "content": "Describe NAME_1"}, {"role": "assistant", "content": "Answer"}]}

    documents = row_to_chat_doc(row, family="instruction_following_chat_v2", partition_name="reasoning_off")

    assert len(documents) == 1
    assert documents[0]["messages"][0]["content"][0]["text"] == "Describe NAME_1"


def test_repeated_source_lines_are_filtered():
    row = {
        "messages": [
            {"role": "user", "content": "A question"},
            {"role": "assistant", "content": "same line\n" * 257},
        ]
    }

    assert row_to_chat_doc(row, family="instruction_following_chat_v2", partition_name="reasoning_off") == []


def test_json_backspace_in_math_markup_is_restored():
    row = {
        "messages": [
            json.loads(r'{"role":"user","content":"Put it in \boxed{}."}'),
            json.loads(r'{"role":"assistant","reasoning_content":"Use \boxed{}.","content":"Done."}'),
        ]
    }

    documents = row_to_chat_doc(row, family="math_v4", partition_name="train")

    assert len(documents) == 1
    texts = [part["text"] for message in documents[0]["messages"] for part in message["content"]]
    assert r"\boxed{}" in texts[0]
    assert r"\boxed{}" in texts[1]
    assert all("\b" not in text for text in texts)


def test_swe_agentless_source_format_suffix_does_not_drop_issue():
    suffix = (
        "\n\nOutput format requirement: Please put your reasoning tokens in a separate code block, starting "
        "with <think> and ending with </think>, and the solution tokens in a separate code block, starting with "
        "<solution> and ending with </solution>."
    )
    row = {
        "messages": [
            {"role": "user", "content": "Fix the import error." + suffix},
            {"role": "assistant", "reasoning_content": "The import is missing.", "content": "Add the import."},
        ]
    }

    documents = row_to_chat_doc(row, family="swe_v2", partition_name="agentless")

    assert len(documents) == 1
    assert documents[0]["messages"][0]["content"][0]["text"] == "Fix the import error."
    assert row["messages"][0]["content"].endswith(suffix)


def test_opencode_tool_schema_is_available_to_chat_template():
    row = {
        "messages": [
            json.dumps({"role": "user", "content": "List files"}),
            json.dumps(
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call-1", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}],
                }
            ),
            json.dumps({"role": "tool", "name": "bash", "tool_call_id": "call-1", "content": "README.md"}),
            json.dumps({"role": "assistant", "content": "Found README.md"}),
        ],
        "tools": [
            {
                "id": "bash",
                "description": "Run a command",
                "inputSchema": {"jsonSchema": {"type": "object", "properties": {"command": {"type": "string"}}}},
            }
        ],
    }

    documents = row_to_chat_doc(row, family="opencode_v1", partition_name="bash_only_tool")

    assert len(documents) == 1
    kwargs = json.loads(documents[0]["chat_template_kwargs"])
    assert kwargs["tools"][0]["function"]["name"] == "bash"
    assert kwargs["tools"][0]["function"]["parameters"]["properties"]["command"]["type"] == "string"


@pytest.mark.parametrize("tools", ["", "   "])
def test_science_vendor_empty_tools_string_keeps_valid_chat(tools):
    row = {
        "messages": [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": "42"},
        ],
        "tools": tools,
    }

    documents = row_to_chat_doc(row, family="science_v2", partition_name="vendor")

    assert len(documents) == 1
    assert [message["role"] for message in documents[0]["messages"]] == ["user", "assistant"]
    assert "tools" not in json.loads(documents[0]["chat_template_kwargs"])


def test_load_jsonl_with_skips_only_ignores_pinned_bad_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": 1}\n{bad json}\n{"id": 2}\n')

    assert list(load_jsonl_with_skips(str(path), frozenset({2}))) == [{"id": 1}, {"id": 2}]
    with pytest.raises(msgspec.DecodeError):
        list(load_jsonl_with_skips(str(path), frozenset()))


def test_jsonl_byte_ranges_read_each_row_once_across_line_boundaries(tmp_path):
    rows = [{"id": index, "text": "λ" * index + ("x" * 100 if index == 5 else "")} for index in range(12)]
    path = tmp_path / "math.jsonl"
    data = b"".join(msgspec.json.encode(row) + b"\n" for row in rows)
    path.write_bytes(data)
    bounds = [len(data) * index // 17 for index in range(18)]
    actual = [
        row
        for start, stop in pairwise(bounds)
        for row in _load_jsonl_byte_range(_JsonlByteRange(str(path), start, stop))
    ]
    assert actual == rows

    first_line_end = data.index(b"\n") + 1
    boundary_rows = [
        *_load_jsonl_byte_range(_JsonlByteRange(str(path), 0, first_line_end)),
        *_load_jsonl_byte_range(_JsonlByteRange(str(path), first_line_end, len(data))),
    ]
    assert boundary_rows == rows
