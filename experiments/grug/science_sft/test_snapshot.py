# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the sealed science SFT snapshot artifact."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.chat_normalize import CHAT_SCHEMA

from experiments.grug.science_sft.snapshot import snapshot


def test_snapshot_copies_and_verifies_each_source_before_sealing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "snapshot"
    names = ["biology/qa", "math/textbook"]
    for index, name in enumerate(names):
        row = {
            "id": str(index),
            "messages": [
                {"role": "user", "name": None, "channel": None, "recipient": None,
                 "content": [{"type": "text", "text": "question"}]},
                {"role": "assistant", "name": None, "channel": "final", "recipient": None,
                 "content": [{"type": "text", "text": "answer"}]},
            ],
            "source": name,
            "source_id": str(index),
            "chat_template_kwargs": None,
        }
        pq.write_table(pa.Table.from_pylist([row], schema=CHAT_SCHEMA), source / f"{name.replace('/', '__')}__0.parquet")

    manifest = snapshot(str(source), str(destination), names, workers=2)

    assert manifest["file_count"] == 2
    assert manifest["conversations"] == 2
    assert manifest["rows_by_source"] == {"biology__qa": 1, "math__textbook": 1}
    assert (destination / "_READY").read_text() == "verified\n"
    assert json.loads((destination / "snapshot-manifest.json").read_text()) == manifest
    for name in names:
        assert pq.read_table(destination / "outputs/main" / f"{name.replace('/', '__')}__0.parquet").num_rows == 1
