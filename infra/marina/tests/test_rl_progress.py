# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json

import pytest

from experiments.post_training import publish_rl_progress as publication


def test_progress_publication_retains_attempts_and_serves_valid_comparisons(rl_progress_client):
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
    first = publication.publish_result(record, "http://localhost")
    assert first["inserted"] and first["verified"]
    assert publication.publish_result(copy.deepcopy(record), "http://localhost")["inserted"] is False
    changed = copy.deepcopy(record)
    changed["final"]["reward"] = 0.9
    with pytest.raises(ValueError, match="different record"):
        publication.publish_result(changed, "http://localhost")
    failed = record | {
        "attempt_id": "native-second",
        "started_at": "2026-11-03T00:00:00Z",
        "completed_at": "2026-11-03T01:00:00Z",
        "status": "failed",
        "optimizer_steps": 0,
    }
    assert publication.publish_result(failed, "http://localhost")["verified"]
    below = copy.deepcopy(record)
    below.update(run_id="test/under-coverage", attempt_id="native-third")
    below["final"]["scored"] = 98
    publication.publish_result(below, "http://localhost")
    other_membership = copy.deepcopy(record)
    other_membership.update(run_id="test/different-membership", attempt_id="native-fourth")
    other_membership["evaluation"]["membership_sha256"] = "b" * 64
    publication.publish_result(other_membership, "http://localhost")
    partial_baseline = copy.deepcopy(record)
    partial_baseline.update(run_id="test/partial-baseline", attempt_id="native-fifth")
    partial_baseline["baseline"]["scored"] = 98
    publication.publish_result(partial_baseline, "http://localhost")
    for field, value in (
        ("label", "arbitrary label"),
        ("issue", 987),
        ("timing_scope", "launcher_start_to_finish"),
    ):
        distinct = copy.deepcopy(record)
        distinct.update(run_id="test/" + field, attempt_id=field)
        distinct[field] = value
        publication.publish_result(distinct, "http://localhost")
    unknown = copy.deepcopy(record)
    unknown.update(run_id="test/unknown-protocol", attempt_id="unknown")
    unknown["evaluation"].update(protocol=None, protocol_sha256=None)
    publication.publish_result(unknown, "http://localhost")
    unknown_time = record | {
        "run_id": "test/unknown-time",
        "attempt_id": "unknown-time",
        "elapsed_seconds": None,
        "run_date": None,
        "started_at": None,
        "completed_at": None,
    }
    publication.publish_result(unknown_time, "http://localhost")
    publication.publish_result(
        record
        | {
            "run_id": "test/resumed",
            "attempt_id": "resumed",
            "resume_from": "source@1",
            "baseline": None,
            "final": None,
        },
        "http://localhost",
    )
    overflow = copy.deepcopy(record)
    overflow.update(run_id="test/overflow", attempt_id="bad")
    overflow["baseline"]["reward"], overflow["final"]["reward"] = -1e308, 1e308
    for run_id, invalid in (
        ("test/overflow", overflow),
        ("test/wrong-sql-key", record | {"attempt_id": "bad"}),
        ("test/bad-shape", ["unexpected"]),
    ):
        request(
            "http://localhost",
            "POST",
            query_path,
            json_body={
                "sql": publication.INSERT,
                "parameters": {
                    "run_id": run_id,
                    "attempt_id": "bad",
                    "record": json.dumps(invalid),
                },
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
    for verdict in ("fail", "invalid"):
        publication.publish_result(
            record | {"run_id": "test/" + verdict, "attempt_id": verdict, "verdict": verdict}, "http://localhost"
        )
    api = client.get(f"/a/{publication.APPLET_ID}/api/runs")
    assert api.status_code == 200
    payload = api.json()
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert len(rows) == 13
    assert rows["resumed"]["provisional"] and not rows["resumed"]["pareto_eligible"]
    assert rows["fail"]["pareto_eligible"]
    assert not rows["invalid"]["pareto_eligible"] and rows["invalid"]["provisional"]
    good = rows["native-first"]
    assert good["final"]["reward"] == 0.4
    assert good["partial_coverage"] and good["pareto_eligible"] and good["trusted_gain_eligible"]
    assert not good["provisional"] and good["gain_per_hour"] == pytest.approx(0.2)
    assert rows["native-second"]["status"] == "failed"
    assert rows["native-second"]["final"]["reward"] == 0.4
    assert not rows["native-second"]["pareto_eligible"]
    assert rows["native-third"]["provisional"] and not rows["native-third"]["pareto_eligible"]
    assert rows["native-fourth"]["comparison_key"] != good["comparison_key"]
    assert rows["native-fifth"]["pareto_eligible"] and not rows["native-fifth"]["trusted_gain_eligible"]
    assert rows["label"]["comparison_key"] == rows["issue"]["comparison_key"] == good["comparison_key"]
    assert rows["timing_scope"]["comparison_key"] != good["comparison_key"]
    assert rows["unknown-time"]["final"]["reward"] == 0.4
    assert rows["unknown-time"]["elapsed_seconds"] is None and rows["unknown-time"]["gain_per_hour"] is None
    assert rows["unknown-time"]["provisional"] and not rows["unknown-time"]["pareto_eligible"]
    assert rows["unknown"]["comparison_key"] is None
    assert rows["unknown"]["provisional"] and not rows["unknown"]["pareto_eligible"]
    assert not rows["unknown"]["trusted_gain_eligible"]
    skipped = {row["run_id"]: row["error"] for row in payload["skipped_records"]}
    assert set(skipped) == {"test/bad-shape", "test/overflow", "test/huge-extra", "test/wrong-sql-key"}
