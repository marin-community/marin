# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import msgspec
import pytest
from marin.datakit.download import nemotron_chat_prompts
from marin.datakit.download.nemotron_chat_prompts import restore_chat_row
from marin.datakit.download.nemotron_sft_v3 import load_jsonl_with_skips, row_to_chat_doc


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


def test_load_jsonl_with_skips_only_ignores_pinned_bad_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": 1}\n{bad json}\n{"id": 2}\n')

    assert list(load_jsonl_with_skips(str(path), frozenset({2}))) == [{"id": 1}, {"id": 2}]
    with pytest.raises(msgspec.DecodeError):
        list(load_jsonl_with_skips(str(path), frozenset()))
