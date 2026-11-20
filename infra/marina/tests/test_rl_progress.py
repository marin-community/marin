# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from infra.marina.tests.rl_progress_fixtures import APPLET_ID, progress_record

pytestmark = pytest.mark.integration


def test_grader_report_survives_republication_and_is_served_by_attempt(rl_progress_client):
    client, _ = rl_progress_client
    api = f"/a/{APPLET_ID}/api"
    record = progress_record()
    report = {"evaluation": {"eval_lite/0": [{"environment": "script", "success_rate": 0.5}]}}
    assert client.post(f"{api}/runs", json=record | {"grader_report": report}).status_code == 200
    assert client.post(f"{api}/runs", json=record | {"attempt_id": "native-second"}).status_code == 200
    assert client.post(f"{api}/runs", json=record).status_code == 200
    first, second = client.get(f"{api}/runs").json()["runs"]
    assert first["has_grader_report"] is True and "grader_report" not in first
    assert second.get("has_grader_report") is None
    query = {"run_id": record["run_id"], "attempt_id": record["attempt_id"]}
    assert client.get(f"{api}/graders", params=query).json() == {"label": record["label"], "report": report}
    assert client.get(f"{api}/graders", params=query | {"attempt_id": "missing"}).status_code == 404


def test_progress_publication_updates_attempts_and_serves_valid_records(rl_progress_client):
    client, request = rl_progress_client
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
    assert publish(sparse) == record
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
                "record": json.dumps(legacy | {"issue": "opted_out"}),
            },
        },
    )
    for run_id, invalid in (
        ("test/legacy", record | {"run_id": "test/legacy", "attempt_id": "bad", "evaluation": None}),
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
    assert rows["native-first"] == changed
    assert rows["native-second"] == failed
    assert rows["unknown"] == legacy | {"issue": None}
    assert len(rows) == 3
    skipped = {row["run_id"]: row["error"] for row in payload["skipped_records"]}
    assert "gate_result.comparisons" in skipped["test/bad-result"]
    assert "evaluation: extra_forbidden" in skipped["test/legacy"]
    assert set(skipped) == {
        "test/bad-shape",
        "test/bad-result",
        "test/huge-extra",
        "test/huge-gate",
        "test/wrong-sql-key",
        "test/legacy",
    }

    for run_id in ("test/bad-result", "test/huge-gate"):
        repaired = record | {"run_id": run_id, "attempt_id": "bad", "gate_result": None}
        assert publish(repaired | {"issue": None, "gate": None}) == repaired
        recovered = client.get(api_path).json()
        assert repaired in recovered["runs"]
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
