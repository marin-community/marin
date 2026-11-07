# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest
from marina.applets import AppletStore, package_applet, read_applet_package
from marina.database_setup import ensure_applet_provisioning
from marina.db import UrlDatabase
from marina.server import MarinaConfig, create_app
from pydantic import ValidationError
from starlette.testclient import TestClient

from experiments.post_training import publish_rl_progress as publication


def test_progress_publication_retains_attempts_and_serves_valid_comparisons(tmp_path, database_url, monkeypatch):
    database = UrlDatabase(database_url)
    store = AppletStore(database)
    ensure_applet_provisioning(store.engine)
    store.migrate()
    source = Path(__file__).parents[1] / "apps/rl_progress"
    store.publish(read_applet_package(package_applet(source)), "test", UUID(publication.APPLET_ID))
    client = TestClient(
        create_app(MarinaConfig(data_root=str(tmp_path), iap_audience=None, database=database)),
        client=("127.0.0.1", 40000),
    )
    query_path = f"/a/{publication.APPLET_ID}/query"

    def request(origin, method, path, *, json_body=None):
        response = client.request(method, path, json=json_body)
        response.raise_for_status()
        return response.json()

    monkeypatch.setattr(publication, "marina_request", request)
    request(
        "http://localhost",
        "POST",
        query_path,
        json_body={
            "sql": (
                "CREATE TABLE rl_runs (run_id text, attempt_id text, record jsonb NOT NULL, "
                "imported_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (run_id, attempt_id))"
            ),
            "parameters": {},
        },
    )
    membership = [f"task-{index}" for index in range(100)]
    record = {
        "run_id": "test/control",
        "attempt_id": "native-first",
        "label": "learning-medium lr=2e-06",
        "campaign": "experiment",
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
        "baseline": None,
        "final": None,
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
        ("task_set_sha256", "c" * 64),
        ("protocol_sha256", None),
        ("timing_scope", "launcher_start_to_finish"),
    ):
        distinct = copy.deepcopy(record)
        distinct.update(run_id="test/" + field, attempt_id=field)
        if field in {"task_set_sha256", "protocol_sha256"}:
            distinct["evaluation"][field] = value
            if field == "protocol_sha256":
                distinct["evaluation"]["protocol"] = {"temperature": 1}
                distinct["evaluation"][field] = hashlib.sha256(b'{"temperature":1}').hexdigest()
        else:
            distinct[field] = value
        publication.publish_result(distinct, "http://localhost")
    malformed = copy.deepcopy(record)
    malformed.update(run_id="test/malformed", attempt_id="bad")
    malformed["evaluation"]["task_ids"] = ["duplicate"] * 100
    with pytest.raises(ValidationError, match="membership"):
        publication.publish_result(malformed, "http://localhost")
    for run_id, invalid in (
        (malformed["run_id"], malformed),
        ("test/bad-shape", ["unexpected"]),
        (
            "test/bad-timestamp",
            record | {"run_id": "test/bad-timestamp", "attempt_id": "bad", "started_at": "yesterday"},
        ),
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
    api = client.get(f"/a/{publication.APPLET_ID}/api/runs")
    assert api.status_code == 200
    payload = api.json()
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert len(rows) == 10
    good = rows["native-first"]
    assert good["final"]["reward"] == 0.4
    assert good["partial_coverage"] and good["pareto_eligible"] and good["trusted_gain_eligible"]
    assert not good["provisional"] and good["gain_per_hour"] == pytest.approx(0.2)
    assert rows["native-second"]["status"] == "failed"
    assert rows["native-second"]["gain_per_hour"] is None
    assert not rows["native-second"]["pareto_eligible"]
    assert rows["native-third"]["provisional"] and not rows["native-third"]["pareto_eligible"]
    assert rows["native-fourth"]["comparison_key"] != good["comparison_key"]
    assert rows["native-fifth"]["pareto_eligible"] and not rows["native-fifth"]["trusted_gain_eligible"]
    assert rows["label"]["comparison_key"] == rows["issue"]["comparison_key"] == good["comparison_key"]
    assert (
        len(
            {
                rows[key]["comparison_key"]
                for key in ("native-first", "task_set_sha256", "protocol_sha256", "timing_scope")
            }
        )
        == 4
    )
    skipped = {row["run_id"]: row["error"] for row in payload["skipped_records"]}
    assert set(skipped) == {"test/malformed", "test/bad-shape", "test/bad-timestamp"}
    assert "evaluation" in skipped["test/malformed"]
