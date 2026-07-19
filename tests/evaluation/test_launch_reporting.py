# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from iris.client import Job
from marin.evaluation.records import (
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    write_record,
)

from experiments.evaluation.launch import GroupRunRef, SubmittedGroup, wait_and_report


class _CompletedJob:
    def wait(self, *, timeout: float, raise_on_failure: bool) -> None:
        pass


def test_wait_and_report_reads_object_store_record_without_database(tmp_path, capsys) -> None:
    run_id = "test-run"
    (tmp_path / run_id).mkdir()
    record = EvalRunRecord(
        run_id=run_id,
        group_id="test-group",
        created_at="2026-07-19T00:00:00+00:00",
        user="tester",
        model=ModelRef(name="test-model", location="test-location", backend="vllm"),
        evaluation=EvalRef(
            name="test-eval",
            mechanism="evalchemy",
            tasks=(EvalTaskRef(name="test-task", num_fewshot=0),),
        ),
        hardware=HardwareRef(platform="tpu", accelerator="v6e-8", region_or_cluster="us-central1"),
        status=RunStatus.SUCCEEDED,
        error=None,
        results_path=str(tmp_path / run_id / "results"),
        metrics={"test-task": {"acc,none": 0.75}},
        jobs={},
        log_tails={},
        provenance=Provenance(git_sha="abc123", evalchemy_image="image", launch_host="host"),
    )
    write_record(record, str(tmp_path))
    group = SubmittedGroup(
        group_id="test-group",
        job=cast(Job, _CompletedJob()),
        records_prefix=str(tmp_path),
        model_key="test-model",
        runs=(GroupRunRef(run_id=run_id, eval_key="test-eval"),),
    )

    wait_and_report([group])

    output = capsys.readouterr().out
    assert "test-run  [succeeded]  test-model / test-eval" in output
    assert "test-task" in output
    assert "0.7500" in output
