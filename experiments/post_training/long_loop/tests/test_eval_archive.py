# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from finestore.reader import ReadView
from marin.evaluation.records import EvalRef, EvalRunRecord, HardwareRef, ModelRef, Provenance, RunStatus, read_record

from experiments.post_training.long_loop.eval_archive import publish_evaluation


def test_evaldash_archive_keeps_failed_trial_without_counting_it_as_zero_reward(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            "uid": "a",
            "data_source": "bench",
            "env_class": "taskcompendium",
            "verification": {"status": "verified", "score": 0.8, "passed": True},
        },
        {
            "uid": "b",
            "data_source": "bench",
            "env_class": "taskcompendium",
            "verification": {"status": "error", "score": None},
        },
    ]
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    (source / "bench.jsonl").write_bytes(raw)
    record = EvalRunRecord(
        run_id="smoke",
        group_id="smoke",
        created_at="2026-11-09T00:00:00Z",
        user="test",
        model=ModelRef(name="test", location="test", backend="vllm"),
        evaluation=EvalRef(name="smoke", mechanism="skyrl"),
        hardware=HardwareRef(platform="local", accelerator="none", region_or_cluster=None),
        status=RunStatus.FAILED,
        error="one verifier failed",
        results_path=str(tmp_path / "results"),
        metrics={},
        jobs={},
        log_tails={},
        provenance=Provenance(git_sha="a" * 40, launch_host="local"),
    )
    path = publish_evaluation(source=str(source), record=record, records_prefix=str(tmp_path / "evals"))
    assert publish_evaluation(source=str(source), record=record, records_prefix=str(tmp_path / "evals")) == path
    stored = read_record(path)
    assert stored.metrics == {"bench": {"reward": 0.8}}
    assert stored.coverage["bench"].n_attempted == 2
    assert stored.coverage["bench"].n_scored == 1
    assert stored.status == RunStatus.FAILED
    view = ReadView(stored.results_path)
    assert view.read_blob("sources/bench.jsonl") == raw
    assert view.is_sealed()
