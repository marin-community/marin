# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import numpy as np
import pytest
from levanter.store.cache import CacheLedger, TreeCache
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record

from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256, BFCLPartition, TaskIdentity
from experiments.post_training.bfcl_rl.preference_union import combine_preference_caches
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache, write_recovery_cache


@pytest.fixture
def partition():
    return BFCLPartition(
        "dataset-revision",
        (TaskIdentity("task", "task-id", "task-digest"),),
        (TaskIdentity("holdout", "holdout-id", "holdout-digest"),),
    )


def preference_cache(path, run_id, rejected_tokens):
    identity = {"run_id": run_id, "model_source_identity": "student-checkpoint", "model_revision": "student-revision"}
    archive = str(path / "student.zip")
    branch = {
        "task_source_id": "task-id",
        "task_digest": "task-digest",
        "harness": "opencode",
        "repetition": 0,
        "model_revision": "student-revision",
    }
    report = {
        "dataset_commit": "dataset-revision",
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "student_tokenizer": "student@student-revision",
        "max_length": 32,
        "conditions_digest": "conditions",
        "scoring": "assistant-only",
        "repeated_tool_call_policy": "retain",
        "student": identity,
        "collections": [{"identity": identity, "conditions_digest": "conditions", "archives": [archive]}],
        "preferences": [
            {
                "chosen": {**branch, "outcome": "correct", "trajectory_uri": "teacher.zip#record"},
                "rejected": {**branch, "outcome": "incorrect", "trajectory_uri": archive + "#student-record"},
            }
        ],
    }
    row = {
        "chosen_input_ids": [1, 2, 3],
        "chosen_assistant_masks": [0, 1, 1],
        "rejected_input_ids": rejected_tokens,
        "rejected_assistant_masks": [0, *([1] * (len(rejected_tokens) - 1))],
    }
    write_recovery_cache([row], report, str(path))
    artifact = RecoveryPreferenceCache(
        path=str(path),
        num_preferences=1,
        tokenizer_uri="student",
        tokenizer_revision="student-revision",
        max_length=32,
        selection_manifest_uri=str(path / "selection.json"),
    )
    write_record(
        ArtifactRecord(
            output_path=str(path),
            result_type=result_type_name(RecoveryPreferenceCache),
            result=artifact.result_payload(),
        )
    )
    return RecoveryPreferenceCache.raw_load(str(path)), row


def test_union_preserves_exact_rows_and_manifest_provenance(tmp_path, partition):
    first, first_row = preference_cache(tmp_path / "first", "run-a", [1, 4, 5])
    second, second_row = preference_cache(tmp_path / "second", "run-b", [1, 6, 7, 8])
    output = tmp_path / "union"
    result = combine_preference_caches((first, second), partition, str(output))
    cache = TreeCache.load(str(output / "train"), {key: np.zeros(0, np.int32) for key in first_row})
    rows = cache.get_batch_sync([0, 1])
    for actual, expected in zip(rows, (first_row, second_row), strict=True):
        for column in expected:
            np.testing.assert_array_equal(actual[column], expected[column])
    assert result.num_preferences == 2
    assert json.loads((output / "train/.stats.json").read_text()) == {"total_elements": 2, "total_tokens": 13}
    manifest = json.loads((output / "selection.json").read_text())
    assert [source["manifest_uri"] for source in manifest["sources"]] == [
        first.selection_manifest_uri,
        second.selection_manifest_uri,
    ]
    assert [pair["rejected"]["trajectory_uri"] for pair in manifest["preferences"]] == [
        str(tmp_path / "first/student.zip") + "#student-record",
        str(tmp_path / "second/student.zip") + "#student-record",
    ]
    ledger = CacheLedger.load(str(output / "train"))
    assert ledger.metadata.preprocessor_metadata == {
        "preference_provenance_uri": str(output / "selection.json"),
        "preference_provenance_sha256": hashlib.sha256((output / "selection.json").read_bytes()).hexdigest(),
    }
    assert (output / "train/.success").exists()


def test_union_rejects_overlapping_snapshots_even_with_different_archive_paths(tmp_path, partition):
    first, _ = preference_cache(tmp_path / "early", "same-run", [1, 4, 5])
    expanded, _ = preference_cache(tmp_path / "later", "same-run", [1, 4, 5])
    output = tmp_path / "union"
    with pytest.raises(ValueError, match="Overlapping student"):
        combine_preference_caches((first, expanded), partition, str(output))
    assert not output.exists()


def test_union_rejects_manifest_changed_after_cache_completion(tmp_path, partition):
    source, _ = preference_cache(tmp_path / "source", "run-a", [1, 4, 5])
    manifest_path = tmp_path / "source/selection.json"
    report = json.loads(manifest_path.read_text())
    report["preferences"][0]["rejected"]["task_source_id"] = "holdout-id"
    manifest_path.write_text(json.dumps(report))
    output = tmp_path / "union"
    with pytest.raises(ValueError, match="Selection hash"):
        combine_preference_caches((source,), partition, str(output))
    assert not output.exists()


def test_union_rejects_holdout_even_with_consistent_provenance_hash(tmp_path, partition):
    source, row = preference_cache(tmp_path / "source", "run-a", [1, 4, 5])
    report = json.loads((tmp_path / "source/selection.json").read_text())
    for branch in report["preferences"][0].values():
        branch.update(task_source_id="holdout-id", task_digest="holdout-digest")
    bad_source = tmp_path / "bad-source"
    write_recovery_cache([row], report, str(bad_source))
    source = source.model_copy(
        update={"path": str(bad_source), "selection_manifest_uri": str(bad_source / "selection.json")}
    )
    output = tmp_path / "union"
    with pytest.raises(ValueError, match="outside the audited complement"):
        combine_preference_caches((source,), partition, str(output))
    assert not output.exists()
