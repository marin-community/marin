# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.download.openresearcher import (
    HF_DATASET_ID,
    HF_FULL_REVISION,
    HF_REVISION,
    download_openresearcher_step,
    load_openresearcher_rows,
    row_to_doc,
    transform,
)


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _message(
    role: str,
    text: str,
    *,
    channel: str = "",
    recipient: str = "",
    name: str = "",
    content_type: str = "",
) -> dict:
    return {
        "role": role,
        "channel": channel,
        "recipient": recipient,
        "name": name,
        "content_type": content_type,
        "content": [_text_block(text)],
    }


def _system_message() -> dict:
    return {
        "role": "system",
        "content": [
            {
                "type": "system_content",
                "model_identity": "You are ChatGPT, a large language model trained by OpenAI.",
                "knowledge_cutoff": "2024-06",
                "reasoning_effort": "high",
                "conversation_start_date": "2026-06-07",
                "channel_config": {
                    "valid_channels": ["analysis", "final"],
                    "channel_required": True,
                },
                "tools": {
                    "browser": {
                        "tools": [
                            {"name": "search", "description": "Search the web."},
                            {"name": "open", "description": "Open a result."},
                            {"name": "find", "description": "Find text."},
                        ]
                    }
                },
            }
        ],
    }


def _messages(include_system: bool = True) -> list[dict]:
    messages = []
    if include_system:
        messages.append(_system_message())
    messages.extend(
        [
            _message("developer", "Use browser evidence before answering."),
            _message("user", "Did Neil Patrick Harris narrate Henry Huggins?"),
            _message(
                "assistant",
                '{"query":"Henry Huggins Neil Patrick Harris narrator","topn":5,"source":"web"}',
                channel="analysis",
                recipient="browser.search",
                content_type="code",
            ),
            _message(
                "tool",
                "[0] Henry Huggins audiobook narrated by Neil Patrick Harris.",
                channel="analysis",
                name="browser.search",
            ),
            _message(
                "assistant",
                "Explanation: the audiobook page lists him.\n\nExact Answer: Yes\n\nConfidence: 98%",
                channel="final",
            ),
        ]
    )
    return messages


def _valid_row(include_system: bool = True, **overrides) -> dict:
    row = {
        "qid": "example-1",
        "question": "Did Neil Patrick Harris narrate Henry Huggins?",
        "answer": "Yes",
        "chunk_idx": 0,
        "attempt": 0,
        "error": None,
        "status": "success",
        "messages": _messages(include_system=include_system),
        "source_seed": "seed_42",
    }
    row.update(overrides)
    return row


def test_load_openresearcher_rows_rejects_unknown_or_missing_seed_config():
    with pytest.raises(ValueError, match="Could not find seed config"):
        list(load_openresearcher_rows("/tmp/raw/seed_cache/train-00000-of-00003.parquet"))

    with pytest.raises(ValueError, match="Could not find seed config"):
        list(load_openresearcher_rows("/tmp/raw/seed_99/train-00000-of-00003.parquet"))


def test_row_to_doc_preserves_deep_research_metadata_and_transcript():
    row = _valid_row()
    row["messages"][2]["content"] = [
        _text_block("Did Neil Patrick Harris"),
        {"type": "image", "text": None},
        _text_block("narrate Henry Huggins?"),
    ]

    [doc] = row_to_doc(row)

    assert doc["source"] == HF_DATASET_ID
    assert doc["source_revision"] == HF_FULL_REVISION
    assert doc["license"] == "mit"
    assert doc["source_seed"] == "seed_42"
    assert doc["qid"] == "example-1"
    assert doc["answer"] == "Yes"
    assert doc["exact_answer"] == "Yes"
    assert doc["answer_match"] == "match"
    assert doc["message_count"] == 6
    assert doc["assistant_message_count"] == 2
    assert doc["tool_message_count"] == 1
    assert doc["browser_search_count"] == 1
    assert json.loads(doc["messages_json"]) == row["messages"]
    tool_schema = json.loads(doc["tool_schema_json"])
    assert {tool["name"] for tool in tool_schema["browser"]["tools"]} == {"search", "open", "find"}
    assert "<openresearcher_metadata>" in doc["text"]
    assert "answer_match: match" in doc["text"]
    assert "Did Neil Patrick Harris\nnarrate Henry Huggins?" in doc["text"]
    assert "<assistant" in doc["text"]
    assert 'to="browser.search"' in doc["text"]
    assert "Henry Huggins Neil Patrick Harris narrator" in doc["text"]
    assert "<tool" in doc["text"]
    assert 'name="browser.search"' in doc["text"]
    assert "audiobook narrated by Neil Patrick Harris" in doc["text"]


def test_row_id_includes_source_seed_for_repeated_qids():
    seed_42 = row_to_doc(_valid_row(source_seed="seed_42"))[0]
    seed_43 = row_to_doc(_valid_row(source_seed="seed_43"))[0]

    assert seed_42["qid"] == seed_43["qid"]
    assert seed_42["id"] != seed_43["id"]
    assert seed_42["id"].startswith("seed_42:example-1:chunk-0:")
    assert seed_43["id"].startswith("seed_43:example-1:chunk-0:")


def test_row_to_doc_keeps_answer_mismatches_as_diagnostic_metadata():
    [doc] = row_to_doc(_valid_row(answer="No"))

    assert doc["answer"] == "No"
    assert doc["exact_answer"] == "Yes"
    assert doc["answer_match"] == "mismatch"
    assert "answer_match: mismatch" in doc["text"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"messages": None},
        {"messages": []},
        {"messages": [_message("user", "hi"), "bad-message"]},
        {"question": ""},
        {"question": "   "},
        {"status": "failed"},
    ],
    ids=[
        "messages-missing",
        "messages-empty",
        "malformed-message",
        "question-empty",
        "question-whitespace",
        "non-success-status",
    ],
)
def test_row_to_doc_drops_structurally_invalid_rows(overrides):
    assert row_to_doc(_valid_row(**overrides)) == []


def test_download_openresearcher_step_selects_requested_seed_train_parquets():
    processed = download_openresearcher_step(seed_configs=("seed_42", "seed_43"), step_suffix="sample")
    download = processed.deps[0]

    assert download.hash_attrs["hf_dataset_id"] == HF_DATASET_ID
    assert download.hash_attrs["revision"] == HF_REVISION
    assert download.hash_attrs["hf_urls_glob"] == ["seed_42/train-*.parquet", "seed_43/train-*.parquet"]
    assert processed.hash_attrs["seed_configs"] == ["seed_42", "seed_43"]


def test_download_openresearcher_step_rejects_invalid_seed_configs():
    with pytest.raises(ValueError, match="at least one"):
        download_openresearcher_step(seed_configs=())

    with pytest.raises(ValueError, match="Unknown OpenResearcher seed"):
        download_openresearcher_step(seed_configs=("seed_99",))


def test_transform_reads_seeded_parquet_and_writes_valid_docs(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "seed_42"
    raw_dir.mkdir(parents=True)
    valid_row = _valid_row(include_system=False)
    failed_row = copy.deepcopy(valid_row)
    failed_row["status"] = "failed"
    table = pa.Table.from_pylist([valid_row, failed_row])
    pq.write_table(table, raw_dir / "train-00000-of-00001.parquet")

    output_dir = tmp_path / "processed"
    transform(str(tmp_path / "raw"), str(output_dir))

    rows = [row for path in output_dir.rglob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert rows == row_to_doc(_valid_row(include_system=False, source_seed="seed_42"))
