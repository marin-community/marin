# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import sys

import pytest
from pydantic import ValidationError

from experiments.post_training.inventory import publish_loop_result as publication
from experiments.post_training.inventory.publication_test_support import ReportTransport, result_record


@pytest.fixture
def result():
    return result_record()


def test_publication_retries_are_idempotent_and_conflicts_preserve_original(monkeypatch, result):
    transport = ReportTransport()
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


def test_incomplete_success_rejected_before_io_but_failed_attempt_remains_visible(monkeypatch, result):
    transport = ReportTransport()
    monkeypatch.setattr(publication, "marina_request", transport)
    result["baseline"]["scored"] = 99
    with pytest.raises(ValidationError, match="complete baseline"):
        publication.publish_result(result)
    assert transport.requests == []
    result["status"] = "failed"
    receipt = publication.publish_result(result)
    assert receipt["verified"]
    assert next(iter(transport.rows.values()))["baseline"]["scored"] == 99

    for scored, verdict in ((99, "pass"), (98, "fail"), (None, "invalid")):
        record = copy.deepcopy(result)
        record.update(status="succeeded", verdict=verdict, min_scored_fraction=0.99, attempt_id=f"coverage-{scored}")
        if scored is None:
            record.update(baseline=None, final=None)
        else:
            record["baseline"]["scored"] = record["final"]["scored"] = scored
        assert publication.publish_result(record)["verified"]
        assert transport.rows[(record["run_id"], record["attempt_id"])] == record


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


def test_pre_evaluation_failure_publishes_no_fabricated_measurements(monkeypatch, result):
    transport = ReportTransport()
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
