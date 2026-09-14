# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.chat_normalize import _normalize_chat_record
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.glm53_format_following import WILDCHAT, wildchat_document
from marin.datakit.download.referenced_completion import resolve_reference_file


def test_wildchat_source_join_preserves_selected_turn_and_literal_answer(tmp_path: Path):
    source = {
        "conversation_hash": "conversation-1",
        "conversation": [
            {"role": "user", "content": "Earlier question", "turn_identifier": "0"},
            {"role": "assistant", "content": "Earlier answer", "turn_identifier": "1"},
            {"role": "user", "content": "  Return a tag.  ", "turn_identifier": "2"},
        ],
    }
    row = {
        "source_dataset": WILDCHAT.source_repo,
        "dataset_revision": WILDCHAT.source_revision,
        "source_file": "data/train-00000-of-00001.parquet",
        "source_split": "train",
        "source_row_group": 1,
        "source_row_in_group": 0,
        "source_turn_index": 2,
        "conversation_hash": "conversation-1",
        "turn_identifier": "2",
        "prompt_sha256": hashlib.sha256(b"Return a tag.").hexdigest(),
        "source_id": hashlib.sha256(b"return a tag.").hexdigest(),
        "pair_id": "pair-1",
        "format_instruction": "Use literal tool_call tags.",
        "answer": '<tool_call>{"answer": 42}</tool_call>',
    }
    path = tmp_path / row["source_file"]
    path.parent.mkdir()
    decoy = {**source, "conversation_hash": "wrong-conversation"}
    pq.write_table(pa.Table.from_pylist([decoy, source]), path, row_group_size=1)
    [doc] = resolve_reference_file(row["source_file"], iter([row]), config=WILDCHAT, source_root=str(tmp_path))
    assert doc["messages"][0]["content"] == [{"type": "text", "text": "Return a tag.\n\nUse literal tool_call tags."}]
    assert len(doc["messages"]) == 2
    assert doc["messages"][1]["channel"] == "final"
    assert doc["messages"][1].get("recipient") is None
    rendered = render_chat_record(_normalize_chat_record(doc, "messages", "id"))["text"]
    assert row["answer"] in rendered
    assert "Earlier answer" not in rendered
    with pytest.raises(ValueError, match="native chat control tokens"):
        wildchat_document({**row, "answer": "<|eot_id|>"}, source)
    with pytest.raises(ValueError, match="reference mismatch"):
        wildchat_document({**row, "prompt_sha256": "wrong"}, source)
    with pytest.raises(ValueError, match="Unexpected source"):
        list(
            resolve_reference_file(
                row["source_file"],
                iter([{**row, "dataset_revision": "main"}]),
                config=WILDCHAT,
                source_root=str(tmp_path),
            )
        )
