# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record

from experiments.post_training.bfcl_rl import final_smoke_data
from experiments.post_training.bfcl_rl.collect import MODELS
from experiments.post_training.bfcl_rl.data import DATASET_COMMIT, PARTITION_MANIFEST_SHA256, BFCLPartition, TaskIdentity
from experiments.post_training.bfcl_rl.final_smoke_data import (
    FreshSmokeDataConfig,
    fresh_smoke_rows,
    run_fresh_smoke_data,
)
from experiments.post_training.bfcl_rl.preference_union import PREFERENCE_COLUMNS
from experiments.post_training.bfcl_rl.recovery_data import recovery_cache_value, write_recovery_cache


def pair(tokens: int, source_index: int) -> dict:
    return {
        "chosen_input_ids": [10, 20, tokens, 40],
        "chosen_assistant_masks": [0, 0, 1, 0],
        "rejected_input_ids": [10, 20, tokens + 1000, 50],
        "rejected_assistant_masks": [0, 0, 1, 0],
        "source_row_index": source_index,
    }


def test_fresh_canary_preserves_masks_and_avoids_repeating_a_frozen_pair():
    fresh = pair(30, -1)
    frozen = [pair(30, 123)] + [pair(value, value) for value in range(31, 95)]
    rows = fresh_smoke_rows([fresh], frozen)
    assert rows == [fresh, *frozen[1:64]]
    assert len({tuple(row["chosen_input_ids"]) for row in rows}) == 64
    assert all(row["chosen_assistant_masks"] == [0, 0, 1, 0] for row in rows)
    assert all(row["rejected_assistant_masks"] == [0, 0, 1, 0] for row in rows)


def test_curated_artifact_to_canary_parquet_preserves_rows_and_masks(tmp_path, monkeypatch):
    tokenizer = MODELS["student"]
    partition = BFCLPartition(DATASET_COMMIT, (TaskIdentity("bfcl-unseen", "unseen", "digest"),), ())
    # The audited remote release is the only fake I/O boundary; artifact, cache and Parquet reads are real.
    monkeypatch.setattr(final_smoke_data, "load_audited_partition", lambda _: partition)
    fresh_path = str(tmp_path / "fresh")
    chosen = {"task_source_id": "unseen", "task_digest": "digest", "harness": "opencode", "outcome": "correct"}
    report = {
        "dataset_commit": DATASET_COMMIT,
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "student_tokenizer": f"{tokenizer.model}@{tokenizer.revision}",
        "student": {"model_revision": tokenizer.revision},
        "max_length": 40960,
        "preferences": [{"chosen": chosen, "rejected": {**chosen, "outcome": "incorrect"}}],
    }
    fresh = pair(30, -1)
    write_recovery_cache([{column: fresh[column] for column in PREFERENCE_COLUMNS}], report, fresh_path)
    cache = recovery_cache_value(fresh_path)
    write_record(
        ArtifactRecord(output_path=fresh_path, result_type=result_type_name(type(cache)), result=cache.result_payload())
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "selection.json").write_text(json.dumps({"seen_task_ids": ["seen"]}))
    rows = [pair(30, 123)] + [pair(value, value) for value in range(31, 1742)]
    for row in rows:
        row.update(task_source_id="unseen", task_digest="digest", token_mask_sha256="fixture", provenance_json="{}")
    schema = pa.schema(
        [(column, pa.list_(pa.int32())) for column in PREFERENCE_COLUMNS]
        + [("source_row_index", pa.int32())]
        + [(column, pa.string()) for column in ("task_source_id", "task_digest", "token_mask_sha256", "provenance_json")]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), frozen / "full-batches.parquet")
    output = str(tmp_path / "output")
    result = run_fresh_smoke_data(FreshSmokeDataConfig(fresh_path, str(frozen), "release", output))
    actual = pq.read_table(tmp_path / "output/smoke.parquet").to_pylist()
    assert result.num_pairs == 64 and result.fresh_pairs == 1
    assert [row["chosen_input_ids"][2] for row in actual] == list(range(30, 94))
    assert actual[0]["source_row_index"] == -1
    assert actual[0]["chosen_input_ids"] == fresh["chosen_input_ids"]
    assert actual[0]["rejected_input_ids"] == fresh["rejected_input_ids"]
    assert all(row["chosen_assistant_masks"] == [0, 0, 1, 0] for row in actual)
    assert all(row["rejected_assistant_masks"] == [0, 0, 1, 0] for row in actual)
