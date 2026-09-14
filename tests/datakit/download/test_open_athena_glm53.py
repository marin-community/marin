# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.chat_normalize import _normalize_chat_record
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.open_athena_glm53 import (
    COMPACTION_TEMPLATE,
    COMPACTIONS,
    OPENCODE_REVISION,
    WILDCHAT,
    compaction_document,
    resolve_reference_file,
    wildchat_document,
)


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


def test_compaction_keeps_original_trajectory_in_user_prompt():
    history = [
        {"role": "user", "content": "Fix café parsing."},
        {"role": "assistant", "content": '<tool_call>{"name":"shell"}</tool_call>'},
        {"role": "user", "content": "Tests passed."},
    ]
    source = {"conversations": history}
    row = {
        "trace_id": hashlib.sha256(json.dumps(history, ensure_ascii=False).encode()).hexdigest(),
        "opencode_revision": OPENCODE_REVISION,
        "compaction": "## Objective\n- Fix café parsing.",
    }
    doc = compaction_document(row, source)
    prompt = doc["messages"][0]["content"][0]["text"]
    assert prompt.startswith(
        "Here is the conversation so far:\n\n<conversation>\n[User]: Fix café parsing.\n\n"
        '[Assistant]: <tool_call>{"name":"shell"}</tool_call>'
    )
    assert "[User]: Tests passed.\n</conversation>\n\nCreate a new anchored summary" in prompt
    assert prompt.endswith(COMPACTION_TEMPLATE)
    assert len(doc["messages"]) == 2
    assert doc["messages"][1]["content"] == [{"type": "text", "text": row["compaction"]}]
    assert _normalize_chat_record(doc, "messages", "id")["source"] == "open-athena/agenttrove-glm53-compactions"
    with pytest.raises(ValueError, match="history hash mismatch"):
        compaction_document(row, {"conversations": history[:-1]})
    with pytest.raises(ValueError, match="prompt revision"):
        compaction_document({**row, "opencode_revision": "main"}, source)


def test_compaction_rejects_negative_source_coordinates(tmp_path: Path):
    row = {
        "source_dataset": COMPACTIONS.source_repo,
        "dataset_revision": COMPACTIONS.source_revision,
        "source_file": "data/train-00000-of-00001.parquet",
        "source_split": "train",
        "source_row_group": 0,
        "source_row_in_group": -1,
    }
    with pytest.raises(ValueError, match="Negative source row coordinate"):
        list(resolve_reference_file(row["source_file"], iter([row]), config=COMPACTIONS, source_root=str(tmp_path)))
