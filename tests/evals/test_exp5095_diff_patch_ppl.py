# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

from experiments import exp5095_diff_patch_ppl as diff_patch
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset


def test_prefixed_diff_patch_validation_sets_prefixes_each_slice() -> None:
    raw_diff = RawTextEvaluationDataset(input_path="raw/diff_patch/swe_bench_raw_diff.jsonl.gz")
    commit_plus_diff = RawTextEvaluationDataset(input_path="raw/diff_patch/commitpack_msg_plus_diff.jsonl.gz")

    prefixed = diff_patch.prefixed_diff_patch_validation_sets(
        {
            "swe_bench_raw_diff": raw_diff,
            "commitpack_msg_plus_diff": commit_plus_diff,
        }
    )

    assert prefixed == {
        os.path.join(diff_patch.DIFF_PATCH_PREFIX, "swe_bench_raw_diff"): raw_diff,
        os.path.join(diff_patch.DIFF_PATCH_PREFIX, "commitpack_msg_plus_diff"): commit_plus_diff,
    }


def test_diff_patch_raw_validation_sets_reads_active_registry(monkeypatch) -> None:
    issue_to_patch = RawTextEvaluationDataset(input_path="raw/diff_patch/swe_bench_issue_to_patch.jsonl.gz")

    monkeypatch.setattr(diff_patch, "ACTIVE_DIFF_PATCH_DATASETS", {"swe_bench_issue_to_patch": issue_to_patch})

    assert diff_patch.diff_patch_raw_validation_sets() == {
        os.path.join(diff_patch.DIFF_PATCH_PREFIX, "swe_bench_issue_to_patch"): issue_to_patch
    }
