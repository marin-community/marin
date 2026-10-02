# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
from rigging.provenance import LAUNCH_PROVENANCE_ENV, Provenance

from levanter.inference.benchmark import (
    BatchMeasurement,
    TokenWorkload,
    measure_batches,
    source_provenance,
    summarize_batch,
)


def test_benchmark_excludes_compile_and_warmup_from_steady_state():
    workload = TokenWorkload(prompts=[[1, 2], [3]], output_tokens=3)
    # Independent recorded measurements: cold=60 seconds, warmup=10, then 2 and 3.
    measurements = iter(
        BatchMeasurement(elapsed=elapsed, first_token=[0.5, 1.0], tokens=[[4, 5, 6], [7, 8, 9]])
        for elapsed in [60, 10, 2, 3, 90]
    )
    result = measure_batches(workload, lambda _: next(measurements), warmup_batches=1, measured_batches=2)
    assert result.first_batch_including_compile.elapsed == 60
    assert result.warmup[0].elapsed == 10
    result = dataclasses.asdict(result)
    # Six generated tokens in 2 and 3 seconds; prompts must not enter the numerator.
    assert result["validation_tokens"] == [[4, 5, 6], [7, 8, 9]]
    assert result["validation_output_sha256"] == result["samples"][0]["output_sha256"]
    assert result["median_output_tokens_per_second"] == 2.5
    assert result["samples"][0]["mean_time_after_first_token_per_output_token"] == 0.625


def test_benchmark_rejects_dropped_requests_and_truncated_generations():
    workload = TokenWorkload(prompts=[[1, 2], [3]], output_tokens=3)
    for outputs in [[[4, 5, 6]], [[4, 5, 6], [7]]]:
        with pytest.raises(ValueError, match="Incomplete generation"):
            summarize_batch(workload, BatchMeasurement(elapsed=2, first_token=[1, 1], tokens=outputs))


def test_source_bundle_does_not_claim_a_clean_git_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(LAUNCH_PROVENANCE_ENV, raising=False)
    absent = source_provenance()
    assert absent.revision is None and absent.dirty is None
    supplied = source_provenance("0123456789abcdef")
    assert supplied.revision == "0123456789abcdef"
    assert supplied.dirty is None
    published = Provenance(tree_hash="tree123", base_commit="commit456", dirty=True, branch=None, built_by=None)
    monkeypatch.setenv(LAUNCH_PROVENANCE_ENV, published.to_json())
    inherited = source_provenance()
    assert (inherited.revision, inherited.tree_hash, inherited.dirty) == ("commit456", "tree123", True)
