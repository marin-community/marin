# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re

from experiments.evals.synthetic_patch_diff_ppl import (
    EXAMPLES_PER_CONFIG,
    SYNTHETIC_PATCH_DIFF_HF_DATASET_ID,
    SyntheticPatchDiffSubset,
    iter_synthetic_patch_diff_records,
    synthetic_patch_diff_raw_validation_sets,
    synthetic_patch_diff_record,
    write_local_sample,
)

EXPECTED_PATCH_KEYS = {
    "synthetic_patch_diff_ppl/commit_message_metadata",
    "synthetic_patch_diff_ppl/failing_test_trace_to_patch",
    "synthetic_patch_diff_ppl/file_path_line_refs",
    "synthetic_patch_diff_ppl/gh_pr_event_patch",
    "synthetic_patch_diff_ppl/review_comment_threads",
    "synthetic_patch_diff_ppl/unified_diff_hunks",
}

EXPECTED_PATCH_SUBSETS = (
    "unified_diff_hunks",
    "file_path_line_refs",
    "review_comment_threads",
    "failing_test_trace_to_patch",
    "commit_message_metadata",
    "gh_pr_event_patch",
)


def test_synthetic_patch_diff_raw_validation_sets_use_supervised_target_only_format():
    datasets = synthetic_patch_diff_raw_validation_sets()

    assert set(datasets) == EXPECTED_PATCH_KEYS
    for key, dataset in datasets.items():
        subset = key.rsplit("/", maxsplit=1)[-1]
        assert dataset.hf_dataset_id == SYNTHETIC_PATCH_DIFF_HF_DATASET_ID
        assert dataset.hf_dataset_name == subset
        assert dataset.input_key == "input"
        assert dataset.target_key == "target"
        assert dataset.split == "validation"
        assert "loss:target_only" in dataset.tags
        assert f"subset:{subset}" in dataset.tags
        assert f"examples:{EXAMPLES_PER_CONFIG}" in dataset.tags


def test_synthetic_patch_diff_record_has_required_schema_and_patch_target():
    record = synthetic_patch_diff_record(SyntheticPatchDiffSubset.FAILING_TEST_TRACE_TO_PATCH, row_index=3)

    assert set(record) == {"id", "subset", "task", "seed", "input", "target", "metadata"}
    assert record["subset"] == "failing_test_trace_to_patch"
    assert record["task"] == "failing_test_trace_to_patch"
    assert "Write the minimal patch hunk" not in str(record["input"])
    assert str(record["input"]).startswith("Failing test:")
    assert "@@ -1,2 +1,2 @@" in str(record["target"])
    assert str(record["target"]).endswith("\n")
    assert isinstance(record["metadata"], dict)
    assert record["metadata"]["eval_only"] is True


def test_iter_records_covers_each_subset_with_requested_count():
    records = list(iter_synthetic_patch_diff_records(examples_per_config=1))
    instruction_fragments = ("Complete ", "Continue ", "Given ", "Write ", "Task:", "User:", "Assistant:")

    assert [record["subset"] for record in records] == list(EXPECTED_PATCH_SUBSETS)
    assert all(record["input"] for record in records)
    assert all(not any(fragment in record["input"] for fragment in instruction_fragments) for record in records)
    assert all(record["target"] for record in records)


def test_patch_diff_examples_preserve_patch_artifact_shapes():
    records_by_subset = {record["subset"]: record for record in iter_synthetic_patch_diff_records(examples_per_config=1)}

    unified_diff = records_by_subset["unified_diff_hunks"]
    assert str(unified_diff["input"]).startswith("File: src/payments/reconcile_0.py\nBug:")
    assert str(unified_diff["target"]).startswith("+    if not items:\n")

    line_ref = records_by_subset["file_path_line_refs"]
    assert str(line_ref["input"]).startswith("Diagnostic: retry loop exits one attempt early")
    assert str(line_ref["target"]).startswith("lib/service/worker_0.ts:39 uses `< maxAttempts`")

    gh_event = records_by_subset["gh_pr_event_patch"]
    assert str(gh_event["input"]).startswith('{"event":"pull_request"')
    assert str(gh_event["target"]).startswith("@@ -22,7 +22,7 @@")


def test_unified_diff_hunk_header_counts_match_body():
    record = synthetic_patch_diff_record(SyntheticPatchDiffSubset.UNIFIED_DIFF_HUNKS, row_index=0)
    hunk = f"{record['input']}{record['target']}"
    header = next(line for line in hunk.splitlines() if line.startswith("@@ "))
    old_count, new_count = (int(count) for count in re.match(r"@@ -\d+,(\d+) \+\d+,(\d+) @@", header).groups())
    body = hunk.split(f"{header}\n", maxsplit=1)[1].splitlines()

    assert sum(not line.startswith("+") for line in body) == old_count
    assert sum(not line.startswith("-") for line in body) == new_count


def test_write_local_sample_creates_jsonl_per_hf_config(tmp_path):
    write_local_sample(tmp_path, examples_per_config=1)

    for subset in EXPECTED_PATCH_SUBSETS:
        sample_path = tmp_path / f"{subset}.jsonl"
        assert sample_path.exists()
        rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["subset"] == subset
        assert set(rows[0]) == {"id", "subset", "task", "seed", "input", "target", "metadata"}
