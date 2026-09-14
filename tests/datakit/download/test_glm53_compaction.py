# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
from marin.datakit.chat_normalize import _normalize_chat_record
from marin.datakit.download.glm53_compaction import (
    COMPACTION_TEMPLATE,
    COMPACTIONS,
    OPENCODE_REVISION,
    compaction_document,
)
from marin.datakit.download.referenced_completion import resolve_reference_file


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
