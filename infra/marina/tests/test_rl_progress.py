# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from infra.marina.tests.rl_progress_fixtures import APPLET_ID

pytestmark = pytest.mark.integration


def test_progress_publication_updates_attempts_and_serves_valid_records(rl_progress_client):
    client, request = rl_progress_client
    query_path = f"/a/{APPLET_ID}/query"
    api_path = f"/a/{APPLET_ID}/api/runs"
    insert_sql = "INSERT INTO rl_runs (run_id, attempt_id, record) VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))"

    def publish(record):
        response = client.post(api_path, json=record)
        assert response.status_code == 200, response.text
        return response.json()

    record = {
        "run_id": "test/control",
        "attempt_id": "native-first",
        "label": "learning-medium lr=2e-06",
        "series": "learning-medium",
        "issue": 123,
        "gate": "test/gate",
        "run_date": "2026-11-02T00:00:00Z",
        "started_at": "2026-11-02T00:00:00Z",
        "completed_at": "2026-11-02T01:00:00Z",
        "elapsed_seconds": 3600,
        "timing_scope": "native_job_start_to_finish",
        "status": "succeeded",
        "optimizer_steps": 25,
        "source_revision": "source-1",
        "data_identity": "data-1",
        "initializer_identity": "model-1",
        "gate_result": {
            "comparisons": [
                {"metric": "score", "role": "objective", "run": 0.4, "control": 0.2, "n_run": 1, "n_control": 4}
            ],
            "invariants": [{"name": "planned_updates", "value": 25, "limit": 25, "holds": True, "replicate": 0}],
            "verdict": "pass",
            "label": "improves",
            "reason": None,
            "attempt_ids": ["native-first"],
            "replicates": 1,
        },
        "artifact_uri": "s3://example/test",
        "evidence": [],
    }
    assert publish(record) == record
    failed = record | {
        "attempt_id": "native-second",
        "started_at": "2026-11-03T00:00:00+00:00",
        "completed_at": "2026-11-03T01:00:00+00:00",
        "status": "failed",
        "optimizer_steps": 0,
    }
    publish(failed)
    changed = copy.deepcopy(record)
    changed.update(issue=456, label="corrected label")
    changed["gate_result"]["comparisons"][0]["run"] = 0.9
    assert publish(changed) == changed
    for omit_issue in (False, True):
        missing_issue = copy.deepcopy(changed)
        if omit_issue:
            missing_issue.pop("issue")
        else:
            missing_issue["issue"] = None
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
    publish(unknown)
    for run_id, invalid in (
        ("test/legacy", record | {"evaluation": None}),
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
    payload = client.get(f"/a/{APPLET_ID}/api/runs").json()
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert rows["native-first"] == changed
    assert rows["native-second"] == failed
    assert rows["unknown"] == unknown
    assert len(rows) == 3
    assert rows["unknown"]["evidence"] == unknown["evidence"]
    assert "comparator" not in rows["native-first"]["gate_result"]["invariants"][0]
    skipped = {row["run_id"]: row["error"] for row in payload["skipped_records"]}
    assert "gate_result.comparisons" in skipped["test/bad-result"]
    assert set(skipped) == {"test/bad-shape", "test/bad-result", "test/huge-extra", "test/wrong-sql-key", "test/legacy"}

    before = client.get(api_path).json()
    for invalid in (
        changed | {"started_at": "2026-11-02T01:00:00+01:00"},
        changed | {"evaluation": None},
        changed | {"elapsed_seconds": float("nan")},
        changed | {"engine": {"value": float("inf")}},
        changed | {"label": "\x00"},
        changed | {"gate_result": {"comparisons": "invalid"}},
    ):
        response = client.post(api_path, content=json.dumps(invalid), headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["detail"]
        assert client.get(api_path).json() == before
