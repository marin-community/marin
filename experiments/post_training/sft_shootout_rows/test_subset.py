# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import json

import pytest

from experiments.post_training.sft_shootout_rows.build import write_split
from experiments.post_training.sft_shootout_rows.subset import intersection_rows, read_split, source_index, task_keys
from experiments.post_training.sweep_data.prepare_agentic import CONTEXT, output_hashes


def test_subset_preserves_whole_rows_masks_and_epoch_identity(tmp_path):
    entries = []
    for family, task, tokens in (("swesmith", "shared", 2), ("tmax", "shared", 3), ("swesmith", "excluded", 4)):
        row = json.dumps(
            {"input_ids": [17] * 8 + [128001] * (CONTEXT - 8), "loss_weights": [1] * tokens + [0] * (CONTEXT - tokens)}
        )
        entries.append((row, dict(family=family, task_id=task, supervised_tokens=tokens, context_tokens=8)))
    source_path = tmp_path / "train-epoch-0-prebuilt.jsonl.gz"
    index, split = write_split(source_path, entries, "source")
    index = [dict(epoch=0, **entry) for entry in index]
    (tmp_path / "train-index.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in index))
    manifest = dict(arm="plain", dry_run=False, splits={"epoch-0": split}, output_hashes=output_hashes(tmp_path))
    raw = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    (tmp_path / "_SUCCESS").touch()
    loaded, loaded_index = source_index(str(tmp_path), hashlib.sha256(raw).hexdigest())
    reference = [dict(epoch=0, family=family, task_id="shared") for family in ("swesmith", "tmax")]
    shared = task_keys(loaded_index) & task_keys(reference)
    selected = intersection_rows(read_split(str(tmp_path), loaded, loaded_index, source_path.name, "epoch-0"), shared)
    expected_lines = [entries[0][0], entries[1][0]]
    assert [line for line, _ in selected] == expected_lines
    for epoch in range(3):
        path = tmp_path / f"subset-{epoch}.jsonl.gz"
        output_index, output_split = write_split(path, selected, f"epoch-{epoch}")
        with gzip.open(path, "rt") as stream:
            lines = stream.read().splitlines()
        assert sorted(lines[:2]) == sorted(expected_lines)
        assert len(lines) == 16
        assert all(not any(json.loads(line)["loss_weights"]) for line in lines[2:])
        assert output_split["supervised_total"] == 5
        assert {(entry["family"], entry["task_id"]) for entry in output_index} == shared


@pytest.mark.parametrize("duplicate", [False, True])
def test_subset_refuses_missing_or_duplicate_shared_rows(duplicate):
    entry = ("{}", dict(family="swesmith", task_id="shared"))
    with pytest.raises(ValueError, match="exactly one"):
        intersection_rows([entry, entry] if duplicate else [], {("swesmith", "shared")})
