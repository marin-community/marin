# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.domain_phase_mix import analyze_delphi_tpp40_bridge_acceptance as acceptance
from experiments.domain_phase_mix import collect_delphi_tpp40_bridge_idempotence as collector

JOB_ID = "/calvinxu/dm-delphi-tpp40-europe-v6e8-bridge-idempotence-test"


def _successful_rows() -> list[dict[str, str]]:
    outer, inner, _ = collector._expected_rerun_command(
        acceptance.COMMAND_FILES["europe"]["training"],
        rerun_job_id=JOB_ID,
    )
    submit_argv = ["iris", "--config", "marin.yaml", "job", "run", *outer, "--", *inner]
    return [
        {
            "job_id": JOB_ID,
            "parent_job_id": "",
            "state": str(collector.JOB_STATE_SUCCEEDED),
            "submitted_at_ms": "2000",
            "finished_at_ms": "3000",
            "exit_code": "",
            "num_tasks": "1",
            "succeeded_task_count": "1",
            "zero_exit_succeeded_task_count": "1",
            "name": JOB_ID,
            "entrypoint_json": json.dumps({"run_command": {"argv": inner}}),
            "bundle_id": "a" * 64,
            "submit_argv_json": json.dumps(submit_argv),
        }
    ]


def test_validate_rerun_job_accepts_exact_command_with_new_parent_name() -> None:
    evidence = collector.validate_rerun_job(
        _successful_rows(),
        job_id=JOB_ID,
        command_path=acceptance.COMMAND_FILES["europe"]["training"],
        captured_at_ms=1000,
    )

    assert evidence["state"] == "succeeded"
    assert evidence["child_job_count"] == 0
    assert evidence["bundle_id"] == "a" * 64
    assert evidence["raw_job_exit_code"] is None
    assert evidence["successful_zero_exit_parent_task_count"] == 1


def test_validate_rerun_job_rejects_any_child_job() -> None:
    rows = _successful_rows()
    rows.append({"job_id": f"{JOB_ID}/child", "parent_job_id": JOB_ID})

    with pytest.raises(ValueError, match="submitted 1 child jobs"):
        collector.validate_rerun_job(
            rows,
            job_id=JOB_ID,
            command_path=acceptance.COMMAND_FILES["europe"]["training"],
            captured_at_ms=1000,
        )


def test_validate_rerun_job_rejects_changed_launcher_arguments() -> None:
    rows = _successful_rows()
    submit_argv = json.loads(rows[0]["submit_argv_json"])
    submit_argv[submit_argv.index("--max-concurrent") + 1] = "3"
    rows[0]["submit_argv_json"] = json.dumps(submit_argv)

    with pytest.raises(ValueError, match="changed its launcher arguments"):
        collector.validate_rerun_job(
            rows,
            job_id=JOB_ID,
            command_path=acceptance.COMMAND_FILES["europe"]["training"],
            captured_at_ms=1000,
        )


def test_validate_rerun_job_rejects_parent_that_predates_snapshot() -> None:
    with pytest.raises(ValueError, match="predates the before snapshot"):
        collector.validate_rerun_job(
            _successful_rows(),
            job_id=JOB_ID,
            command_path=acceptance.COMMAND_FILES["europe"]["training"],
            captured_at_ms=2000,
        )


def test_validate_rerun_job_rejects_missing_zero_exit_parent_task() -> None:
    rows = _successful_rows()
    rows[0]["zero_exit_succeeded_task_count"] = "0"

    with pytest.raises(ValueError, match="lacks one successful zero-exit parent task"):
        collector.validate_rerun_job(
            rows,
            job_id=JOB_ID,
            command_path=acceptance.COMMAND_FILES["europe"]["training"],
            captured_at_ms=1000,
        )
