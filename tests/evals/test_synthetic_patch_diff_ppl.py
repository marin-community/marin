# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.evals.synthetic_patch_diff_ppl import (
    EXAMPLES_PER_CONFIG,
    SYNTHETIC_PATCH_DIFF_HF_DATASET_ID,
    SYNTHETIC_PATCH_DIFF_PPL_SLICES,
    SyntheticPatchDiffSubset,
    iter_synthetic_patch_diff_records,
    synthetic_patch_diff_raw_validation_sets,
    synthetic_patch_diff_record,
    write_local_sample,
)


def test_synthetic_patch_diff_raw_validation_sets_use_supervised_target_only_format():
    datasets = synthetic_patch_diff_raw_validation_sets()

    assert set(datasets) == {slice_.registry_key for slice_ in SYNTHETIC_PATCH_DIFF_PPL_SLICES}
    for slice_ in SYNTHETIC_PATCH_DIFF_PPL_SLICES:
        dataset = datasets[slice_.registry_key]
        assert dataset.hf_dataset_id == SYNTHETIC_PATCH_DIFF_HF_DATASET_ID
        assert dataset.hf_dataset_name == slice_.hf_config_name
        assert dataset.input_key == "input"
        assert dataset.target_key == "target"
        assert dataset.split == "validation"
        assert "loss:target_only" in dataset.tags
        assert f"subset:{slice_.subset.value}" in dataset.tags
        assert f"examples:{EXAMPLES_PER_CONFIG}" in dataset.tags


def test_synthetic_patch_diff_record_has_required_schema_and_patch_target():
    record = synthetic_patch_diff_record(SyntheticPatchDiffSubset.FAILING_TEST_TRACE_TO_PATCH, row_index=3)

    assert set(record) == {"id", "subset", "task", "seed", "input", "target", "metadata"}
    assert record["subset"] == "failing_test_trace_to_patch"
    assert record["task"] == "failing_test_trace_to_patch"
    assert "Write the minimal patch hunk" in str(record["input"])
    assert "@@ -1,2 +1,2 @@" in str(record["target"])
    assert str(record["target"]).endswith("\n")
    assert isinstance(record["metadata"], dict)
    assert record["metadata"]["eval_only"] is True


def test_iter_records_covers_each_subset_with_requested_count():
    records = list(iter_synthetic_patch_diff_records(examples_per_config=1))

    assert [record["subset"] for record in records] == [subset.value for subset in SyntheticPatchDiffSubset]
    assert all(record["input"] for record in records)
    assert all(record["target"] for record in records)


def test_write_local_sample_creates_jsonl_per_hf_config(tmp_path):
    write_local_sample(tmp_path, examples_per_config=1)

    for subset in SyntheticPatchDiffSubset:
        sample_path = tmp_path / f"{subset.value}.jsonl"
        assert sample_path.exists()
        rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["subset"] == subset.value
        assert set(rows[0]) == {"id", "subset", "task", "seed", "input", "target", "metadata"}
