# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the terminal evidence the dashboard ranks and serves."""

import json

from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.rl_progress.server import ingest
from infra.marina.apps.rl_progress.server.app import comparison_fields


def test_terminal_quick_loop_record_uses_coordinator_time_and_preserves_grader_stats(tmp_path, monkeypatch):
    def write(relative: str, value: dict) -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    terminal_uri = "s3://marin-us-east-02a/marin/users/user/checkpoints/snowball-inventory/2026.11.19/terminal.json"
    original_run_root = ingest.run_root

    def resolve_fixture(uri: str) -> StoragePath:
        original_run_root(uri)
        return StoragePath(str(tmp_path))

    monkeypatch.setattr(ingest, "run_root", resolve_fixture)
    entry = {
        "run_id": "users/user/checkpoints/snowball-inventory-2026.11.19",
        "attempt_id": "a12345678901",
        "coordinator_job_id": "/user/quick-rl-coordinator",
        "terminal_manifest_uri": terminal_uri,
        "expected_step": 3,
        "source_revision": "source",
        "data_identity": "training",
        "validation_identity": "validation",
        "initializer_identity": "checkpoint",
    }
    write(
        "terminal.json",
        {
            "result": {
                "run_id": entry["run_id"],
                "attempt_id": entry["attempt_id"],
                "state": "succeeded",
                "iris_job_id": "/user/child",
            }
        },
    )
    identity = {"membership_sha256": "panel", "sampling": {"temperature": 0.0}}
    report = {
        "evaluation": {"3": [{"data_source": "code", "success_rate": 0.75}]},
        "training": {},
        "evaluation_metadata": {
            "0": {"attempted": 20, "scored": 20, "reward": 0.25, "identity": identity},
            "3": {"attempted": 20, "scored": 20, "reward": 0.75, "identity": identity},
        },
        "evaluation_duration_seconds": {"0": 8.0, "3": 12.0},
    }
    report_path = f"exports/grader_reports/{entry['attempt_id']}/grader_report.json"
    write(report_path, report)
    coordinator = job_pb2.JobStatus(state=job_pb2.JOB_STATE_SUCCEEDED)
    coordinator.submitted_at.epoch_ms = 100_000
    coordinator.started_at.epoch_ms = 110_000
    coordinator.finished_at.epoch_ms = 210_000

    result = ingest.terminal_record(entry, coordinator)

    assert result["coordinator_wall_seconds"] == 110.0
    assert result["elapsed_seconds"] == 90.0
    assert result["timing_scope"] == "coordinator_wall_minus_validation"
    assert result["validation_seconds"] == {"baseline": 8.0, "final": 12.0}
    assert result["grader_report"] == report
    assert result["final"]["reward"] == 0.75
    assert comparison_fields(result)["pareto_eligible"] is True

    other_attempt_id = "a45678901234"
    stale_attempt = ingest.terminal_record(entry | {"attempt_id": other_attempt_id}, coordinator)
    assert stale_attempt["status"] == "failed"
    assert stale_attempt["grader_report"] is None
    assert comparison_fields(stale_attempt)["pareto_eligible"] is False

    partial_report = {"training": {"execution": [{"data_source": "code", "grader_failures": {"timeout": 4}}]}}
    write(f"exports/grader_reports/{other_attempt_id}/grader_report.json", partial_report)
    coordinator.state = job_pb2.JOB_STATE_FAILED
    failed_attempt = ingest.terminal_record(entry | {"attempt_id": other_attempt_id}, coordinator)
    assert failed_attempt["status"] == "failed"
    assert failed_attempt["grader_report"] == partial_report
    coordinator.state = job_pb2.JOB_STATE_SUCCEEDED

    report["evaluation_metadata"]["3"]["identity"] = identity | {"membership_sha256": "different-panel"}
    write(report_path, report)
    different_panel = ingest.terminal_record(entry, coordinator)
    assert comparison_fields(different_panel)["pareto_eligible"] is False
