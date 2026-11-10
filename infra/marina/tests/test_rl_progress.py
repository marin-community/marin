# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json

import pytest

from experiments.post_training import publish_rl_progress as publication


def test_progress_publication_updates_attempts_and_serves_valid_records(rl_progress_client, monkeypatch):
    client, request = rl_progress_client
    query_path = f"/a/{publication.APPLET_ID}/query"
    membership = [f"task-{index}" for index in range(100)]
    record = {
        "run_id": "test/control",
        "attempt_id": "native-first",
        "label": "learning-medium lr=2e-06",
        "series": "learning-medium",
        "issue": 123,
        "gate": "test/gate",
        "verdict": None,
        "min_scored_fraction": 0.99,
        "run_date": "2026-11-02T00:00:00Z",
        "date_basis": "submission_acknowledged",
        "started_at": "2026-11-02T00:00:00Z",
        "completed_at": "2026-11-02T01:00:00Z",
        "elapsed_seconds": 3600,
        "timing_scope": "native_job_start_to_finish",
        "status": "succeeded",
        "optimizer_steps": 25,
        "source_revision": "source-1",
        "data_identity": "data-1",
        "initializer_identity": "model-1",
        "evaluation": {
            "name": "test-panel",
            "version": "1",
            "task_count": 100,
            "task_set_sha256": "a" * 64,
            "membership_sha256": (
                hashlib.sha256(json.dumps(sorted(membership), separators=(",", ":")).encode()).hexdigest()
            ),
            "protocol": {"temperature": 0},
            "protocol_sha256": hashlib.sha256(b'{"temperature":0}').hexdigest(),
        },
        "baseline": {"reward": 0.2, "attempted": 100, "scored": 99},
        "final": {"reward": 0.4, "attempted": 100, "scored": 99},
        "artifact_uri": "s3://example/test",
        "evidence": [],
    }
    url = publication.publish_result(record, "http://localhost")
    assert url.endswith("#run=test%2Fcontrol&attempt=native-first")
    publication.publish_result(copy.deepcopy(record), "http://localhost")
    failed = record | {
        "attempt_id": "native-second",
        "started_at": "2026-11-03T00:00:00+00:00",
        "completed_at": "2026-11-03T01:00:00+00:00",
        "status": "failed",
        "optimizer_steps": 0,
    }
    publication.publish_result(failed, "http://localhost")
    changed = copy.deepcopy(record)
    changed.update(issue=456, label="corrected label", verdict="fail")
    changed["final"]["reward"] = 0.9
    publication.publish_result(changed, "http://localhost")
    partial = copy.deepcopy(record)
    partial.update(run_id="test/partial", attempt_id="partial")
    partial["final"]["scored"] = 98
    publication.publish_result(partial, "http://localhost")
    unknown = record | {
        "run_id": "test/unknown",
        "attempt_id": "unknown",
        "verdict": "invalid",
        "elapsed_seconds": None,
        "run_date": None,
        "started_at": None,
        "completed_at": None,
        "baseline": None,
        "final": None,
        "evaluation": None,
        "optimizer_steps": None,
    }
    publication.publish_result(unknown, "http://localhost")
    for run_id, invalid in (
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
                "sql": publication.INSERT,
                "parameters": {"run_id": run_id, "attempt_id": "bad", "record": json.dumps(invalid)},
            },
        )
    huge_extra = json.dumps(
        record | {"run_id": "test/huge-extra", "attempt_id": "bad", "extra": "large-number"}
    ).replace('"large-number"', "1" + "0" * 400 + ".5")
    request(
        "http://localhost",
        "POST",
        query_path,
        json_body={
            "sql": publication.INSERT,
            "parameters": {"run_id": "test/huge-extra", "attempt_id": "bad", "record": huge_extra},
        },
    )
    payload = client.get(f"/a/{publication.APPLET_ID}/api/runs").json()
    assert len(payload["runs"]) == 4
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert rows["native-first"]["final"]["reward"] == 0.9
    assert rows["native-first"] == changed
    assert rows["native-second"] == failed
    assert rows["partial"] == partial and rows["unknown"] == unknown
    skipped = {row["run_id"]: row["error"] for row in payload["skipped_records"]}
    assert "gate_result.comparisons" in skipped["test/bad-result"]
    assert set(skipped) == {"test/bad-shape", "test/bad-result", "test/huge-extra", "test/wrong-sql-key"}

    def rejected_record(origin, method, path, **kwargs):
        if path.endswith("/api/runs"):
            return {
                "runs": [],
                "skipped_records": [
                    {"run_id": changed["run_id"], "attempt_id": changed["attempt_id"], "error": "unsupported result"}
                ],
            }
        return request(origin, method, path, **kwargs)

    monkeypatch.setattr("marina.client.marina_request", rejected_record)
    with pytest.raises(RuntimeError, match=r"test/control/native-first.*unsupported result"):
        publication.publish_result(changed, "http://localhost")
