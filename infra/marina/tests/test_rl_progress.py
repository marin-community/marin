# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from infra.marina.tests.rl_progress_fixtures import APPLET_ID, grader_report, progress_record

pytestmark = pytest.mark.integration


def _projection(record: dict) -> dict:
    projected = record.copy()
    projected["has_grader_report"] = projected.pop("grader_report", None) is not None
    return projected


def test_grader_report_survives_republication_and_is_served_by_attempt(rl_progress_client):
    client, _, _ = rl_progress_client
    api = f"/a/{APPLET_ID}/api"
    record = progress_record()
    report = grader_report()
    assert client.post(f"{api}/runs", json=record | {"grader_report": report}).status_code == 200
    assert client.post(f"{api}/runs", json=record | {"attempt_id": "native-second"}).status_code == 200
    assert client.post(f"{api}/runs", json=record).status_code == 200
    first, second = client.get(f"{api}/runs").json()["runs"]
    assert first["has_grader_report"] is True and "grader_report" not in first
    assert second["has_grader_report"] is False
    query = {"run_id": record["run_id"], "attempt_id": record["attempt_id"]}
    assert client.get(f"{api}/graders", params=query).json() == {"label": record["label"], "report": report}
    assert client.get(f"{api}/graders", params=query | {"attempt_id": "missing"}).status_code == 404


def test_progress_publication_updates_attempts_and_serves_valid_records(rl_progress_client):
    client, request, _ = rl_progress_client
    query_path = f"/a/{APPLET_ID}/query"
    api_path = f"/a/{APPLET_ID}/api/runs"
    insert_sql = "INSERT INTO rl_runs (run_id, attempt_id, record) VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))"

    def publish(record):
        response = client.post(api_path, json=record)
        assert response.status_code == 200, response.text
        return response.json()

    record = progress_record()
    sparse = {key: value for key, value in record.items() if value is not None}
    sparse.pop("evaluations")
    sparse.pop("failed_facts")
    running = record | {"status": "running", "completed_at": None, "elapsed_seconds": None, "gate_result": None}
    assert publish(running)["status"] == "running"
    assert publish(sparse) == record
    assert publish(running) == record
    failed = record | {
        "attempt_id": "native-second",
        "started_at": "2026-11-03T00:00:00+00:00",
        "completed_at": "2026-11-03T01:00:00+00:00",
        "status": "failed",
        "optimizer_steps": 0,
    }
    publish(failed)
    changed = copy.deepcopy(record)
    changed.update(issue=456, label="corrected label", model="snowball", evaldash_run_ids=["eval-start", "eval-finish"])
    changed["gate_result"]["comparisons"][0]["run"] = 0.9
    assert publish(changed) == changed
    for omit_issue in (False, True):
        missing_issue = copy.deepcopy(changed)
        for field in ("issue", "gate", "gate_result", "model", "evaldash_run_ids"):
            if omit_issue:
                missing_issue.pop(field)
            else:
                missing_issue[field] = None
        assert publish(missing_issue) == changed
    changed["evaldash_run_ids"] = ["eval-finish"]
    assert publish(changed) == changed
    unknown = record | {
        "run_id": "test/unknown",
        "attempt_id": "unknown",
        "elapsed_seconds": None,
        "run_date": None,
        "started_at": None,
        "completed_at": None,
        "optimizer_steps": None,
        "evidence": ["Checkpoint lineage timing is unknown: a native time window is missing."],
    }
    legacy = {key: value for key, value in unknown.items() if key not in {"model", "evaldash_run_ids", "issue"}}
    request(
        "http://localhost",
        "POST",
        query_path,
        json_body={
            "sql": insert_sql,
            "parameters": {
                "run_id": legacy["run_id"],
                "attempt_id": "unknown",
                "record": json.dumps(legacy | {"issue": "opted_out", "future_fact": "stored"}),
            },
        },
    )
    for run_id, invalid in (
        ("test/bad-status", record | {"run_id": "test/bad-status", "attempt_id": "bad", "status": "unknown"}),
        ("test/wrong-sql-key", record | {"attempt_id": "bad"}),
        ("test/bad-shape", ["unexpected"]),
        (
            "test/bad-result",
            record | {"run_id": "test/bad-result", "attempt_id": "bad", "gate_result": {"comparisons": "invalid"}},
        ),
    ):
        request(
            "http://localhost",
            "POST",
            query_path,
            json_body={
                "sql": insert_sql,
                "parameters": {"run_id": run_id, "attempt_id": "bad", "record": json.dumps(invalid)},
            },
        )
    huge_extra = json.dumps(
        record | {"run_id": "test/huge-extra", "attempt_id": "bad", "engine": {"value": "large-number"}}
    ).replace('"large-number"', "1" + "0" * 400 + ".5")
    request(
        "http://localhost",
        "POST",
        query_path,
        json_body={
            "sql": insert_sql,
            "parameters": {"run_id": "test/huge-extra", "attempt_id": "bad", "record": huge_extra},
        },
    )
    huge_gate = copy.deepcopy(record)
    huge_gate.update(run_id="test/huge-gate", attempt_id="bad")
    huge_gate["gate_result"]["invariants"][0]["value"] = "large-number"
    request(
        "http://localhost",
        "POST",
        query_path,
        json_body={
            "sql": insert_sql,
            "parameters": {
                "run_id": "test/huge-gate",
                "attempt_id": "bad",
                "record": json.dumps(huge_gate).replace('"large-number"', "1" + "0" * 400 + ".5"),
            },
        },
    )
    payload = client.get(f"/a/{APPLET_ID}/api/runs").json()
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert rows["native-first"] == _projection(changed)
    assert rows["native-second"] == _projection(failed)
    assert rows["unknown"] == _projection(legacy | {"issue": None, "future_fact": "stored"})
    assert len(rows) == 3
    skipped = {row["run_id"]: row["error"] for row in payload["skipped_records"]}
    assert "gate_result.comparisons" in skipped["test/bad-result"]
    assert "status: literal_error" in skipped["test/bad-status"]
    assert set(skipped) == {
        "test/bad-shape",
        "test/bad-result",
        "test/huge-extra",
        "test/huge-gate",
        "test/wrong-sql-key",
        "test/bad-status",
    }

    for run_id in ("test/bad-result", "test/huge-gate"):
        repaired = record | {"run_id": run_id, "attempt_id": "bad", "gate_result": None}
        assert publish(repaired | {"issue": None, "gate": None}) == repaired
        recovered = client.get(api_path).json()
        assert _projection(repaired) in recovered["runs"]
        assert run_id not in {row["run_id"] for row in recovered["skipped_records"]}

    before = client.get(api_path).json()
    for invalid in (
        changed | {"started_at": "2026-11-02T01:00:00+01:00"},
        changed | {"evaluation": None},
        changed | {"elapsed_seconds": float("nan")},
        changed | {"engine": {"value": float("inf")}},
        changed | {"label": "\x00"},
        changed | {"issue": "opted_out"},
        changed | {"gate_result": {"comparisons": "invalid"}},
    ):
        response = client.post(api_path, content=json.dumps(invalid), headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["detail"]
        assert client.get(api_path).json() == before


def test_release_archives_v9_rows_and_versions_retained_attempts(rl_progress_client):
    client, request, publish_release = rl_progress_client
    query_path = f"/a/{APPLET_ID}/query"

    def query(sql, parameters=None):
        return request("http://localhost", "POST", query_path, json_body={"sql": sql, "parameters": parameters or {}})

    # Shape and measurements captured from the hosted v9 control listing.
    legacy = {
        "run_id": "checkpoints/snowball-rl/v9",
        "attempt_id": "av9",
        "evaluation": {
            "name": "test_panel",
            "task_count": 1500,
            "version": "rl-data/task-selection/snowball-test-panel@2026.11.06.1:88ee12a7",
        },
        "baseline": {"reward": 0.5280307370790129, "scored": 1500, "attempted": 1500},
        "final": {"reward": 0.6195039305522064, "scored": 1500, "attempted": 1500},
        "preset": "learning-medium",
    }
    retained = progress_record()
    retained.pop("format_version")
    archive = legacy | {"attempt_id": "previously-archived"}
    query("CREATE TABLE rl_archived_runs (LIKE rl_runs INCLUDING ALL)")
    query(
        "INSERT INTO rl_archived_runs (run_id,attempt_id,record) VALUES (:run_id,:attempt_id,CAST(:record AS jsonb))",
        {"run_id": archive["run_id"], "attempt_id": archive["attempt_id"], "record": json.dumps(archive)},
    )
    for row in (legacy, retained):
        query(
            "INSERT INTO rl_runs (run_id,attempt_id,record) VALUES (:run_id,:attempt_id,CAST(:record AS jsonb))",
            {"run_id": row["run_id"], "attempt_id": row["attempt_id"], "record": json.dumps(row)},
        )

    publish_release()

    assert query("SELECT record FROM rl_legacy_runs")["rows"] == [{"record": legacy}]
    assert query("SELECT record FROM rl_archived_runs")["rows"] == [{"record": archive}]
    payload = client.get(f"/a/{APPLET_ID}/api/runs").json()
    assert payload["runs"] == [_projection(retained | {"format_version": 1})]
    assert payload["skipped_records"] == []
