# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import sys

import pytest
from pydantic import ValidationError

from experiments.post_training.inventory import publish_loop_result as publication


@pytest.fixture
def result():
    return {
        "run_id": "test/control",
        "attempt_id": "attempt-1",
        "label": "control",
        "run_date": "2026-11-02T00:00:00Z",
        "date_basis": "submission_acknowledged",
        "started_at": "2026-11-02T00:00:00Z",
        "completed_at": "2026-11-02T01:00:00Z",
        "elapsed_seconds": 3600,
        "timing_scope": "coordinator_submission_to_terminal",
        "status": "succeeded",
        "optimizer_steps": 25,
        "source_revision": "source-1",
        "data_identity": "data-1",
        "initializer_identity": "model-1",
        "evaluation": {
            "name": "fast",
            "version": "1",
            "task_count": 100,
            "task_set_sha256": "a" * 64,
            "protocol_sha256": None,
        },
        "baseline": {"reward": 0.2, "attempted": 100, "scored": 100},
        "final": {"reward": 0.4, "attempted": 100, "scored": 100},
        "artifact_uri": "s3://example/test",
        "evidence": ["synthetic unit-test result"],
    }


@pytest.fixture
def api_result(result):
    return result | {
        "partial_coverage": False,
        "comparison_key": None,
        "provisional": True,
        "pareto_eligible": False,
    }


class ReportTransport:
    def __init__(self, api_runs=()):
        self.rows = {}
        self.requests = []
        self.api_runs = api_runs

    def __call__(self, origin, method, path, *, json_body=None):
        self.requests.append((origin, method, path))
        if path.startswith("/api/marina/applets/"):
            return {"authenticated_url": f"https://applets.marina.oa.dev/a/{publication.APPLET_ID}/"}
        if method == "POST":
            parameters = json_body["parameters"]
            key = (parameters["run_id"], parameters["attempt_id"])
            if "record" in parameters:
                if key in self.rows:
                    return {"row_count": 0}
                self.rows[key] = json.loads(parameters["record"])
                return {"row_count": 1}
            return {"rows": [{"record": copy.deepcopy(self.rows[key])}]}
        return {"runs": list(self.api_runs)}


def test_publication_retries_are_idempotent_and_conflicts_preserve_original(monkeypatch, result, api_result):
    transport = ReportTransport([api_result])
    monkeypatch.setattr(publication, "marina_request", transport)
    first = publication.publish_result(result)
    repeated = publication.publish_result(copy.deepcopy(result))
    assert first["inserted"] is True
    assert repeated["inserted"] is False
    assert first["verified"] and repeated["verified"]
    changed = copy.deepcopy(result)
    changed["final"]["reward"] = 0.9
    with pytest.raises(ValueError, match="different record"):
        publication.publish_result(changed)
    assert list(transport.rows.values()) == [result]
    assert all(origin == "https://applets.marina.oa.dev" for origin, method, _ in transport.requests if method == "POST")


def test_incomplete_success_rejected_before_io_but_failed_attempt_remains_visible(monkeypatch, result, api_result):
    transport = ReportTransport([api_result])
    monkeypatch.setattr(publication, "marina_request", transport)
    result["baseline"]["scored"] = 99
    with pytest.raises(ValidationError, match="complete baseline"):
        publication.publish_result(result)
    assert transport.requests == []
    result["status"] = "failed"
    api_result.update(result, partial_coverage=True)
    receipt = publication.publish_result(result)
    assert receipt["verified"]
    assert next(iter(transport.rows.values()))["baseline"]["scored"] == 99


def test_nonfinite_measurement_never_reaches_publication(monkeypatch, result):
    transport = ReportTransport()
    monkeypatch.setattr(publication, "marina_request", transport)
    result["final"]["reward"] = float("nan")
    with pytest.raises(ValidationError):
        publication.publish_result(result)
    assert transport.rows == {}
    assert transport.requests == []


def test_invalid_evaluation_identity_cannot_poison_report_api(monkeypatch, result):
    transport = ReportTransport()
    monkeypatch.setattr(publication, "marina_request", transport)
    result["evaluation"]["task_ids"] = ["replaced"] * 100
    result["evaluation"]["membership_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="membership"):
        publication.publish_result(result)
    assert transport.rows == {}
    assert transport.requests == []


def test_pre_evaluation_failure_publishes_no_fabricated_measurements(monkeypatch, result, api_result):
    transport = ReportTransport([api_result])
    monkeypatch.setattr(publication, "marina_request", transport)
    result.update(
        status="failed",
        baseline=None,
        final=None,
        optimizer_steps=0,
        date_basis="launch_started",
        timing_scope="loop_launcher_to_terminal",
    )
    result["evaluation"]["task_set_sha256"] = None
    api_result.update(result, partial_coverage=True)
    receipt = publication.publish_result(result)
    assert receipt["verified"]
    stored = next(iter(transport.rows.values()))
    assert stored["baseline"] is None and stored["final"] is None
    assert stored["date_basis"] == "launch_started"


def test_publication_without_deployment_client_reports_missing_dependency(monkeypatch, result):
    monkeypatch.setitem(sys.modules, "marina", None)
    monkeypatch.setitem(sys.modules, "marina.client", None)
    with pytest.raises(RuntimeError, match="requires the Marina client"):
        publication.publish_result(result)
