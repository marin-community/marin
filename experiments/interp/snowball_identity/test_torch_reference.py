# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The torch reference entrypoint stops before the load on any failed check and never reports a bare exit 0."""

import json

import pytest

from experiments.interp.snowball_identity import torch_reference
from experiments.interp.snowball_identity.worker import Queue


def read(queue: Queue, *parts: str) -> dict:
    value = queue.read(*parts)
    assert value is not None, parts
    return value


SPEC = {"command": "router_rounding", "positions": "P", "positions_sha256": "S"}


def setup(tmp_path, monkeypatch, preflight=(), audit_error=None, results=None, check=()):
    calls = []
    monkeypatch.setattr(torch_reference.preflight, "source_preflight", lambda weights, expect: list(preflight))

    def audit(weights, hashes, receipt):
        calls.append("audit")
        if audit_error:
            raise audit_error

    def serve(queue, weights, idle_minutes):
        calls.append("serve")
        for job_id, outcome in (results or {}).items():
            queue.write({"spec": SPEC}, "results", job_id, outcome)
        return sum(outcome == "error.json" for outcome in (results or {}).values())

    monkeypatch.setattr(torch_reference, "source_audit", audit)
    monkeypatch.setattr(torch_reference, "serve", serve)
    monkeypatch.setattr(torch_reference.preflight, "check_router_rounding_result", lambda *args: list(check))
    queue = Queue(str(tmp_path / "q"))
    code = torch_reference.run(queue, "W", {"files": {}}, idle_minutes=2, layers=26, k=4)
    return code, calls, queue


def test_failed_preflight_stops_before_the_audit_and_load(tmp_path, monkeypatch):
    code, calls, queue = setup(tmp_path, monkeypatch, preflight=["config.json: sha256 x"])
    assert code == 2 and calls == [] and read(queue, "preflight.json")["failures"] == ["config.json: sha256 x"]


def test_failed_source_audit_stops_before_the_load(tmp_path, monkeypatch):
    code, calls, queue = setup(tmp_path, monkeypatch, audit_error=ValueError("Nonfinite source tensor: x"))
    assert code == 2 and calls == ["audit"] and "Nonfinite" in read(queue, "source-audit-error.json")["error"]


def test_exit_zero_needs_a_checked_result(tmp_path, monkeypatch):
    code, calls, queue = setup(tmp_path, monkeypatch, results={"r1": "done.json"})
    assert code == 0 and calls == ["audit", "serve"] and read(queue, "final-check.json")["failures"] == []


def test_missing_bad_or_failed_results_exit_one(tmp_path, monkeypatch):
    assert setup(tmp_path / "a", monkeypatch)[0] == 1  # no result at all
    code, _, queue = setup(tmp_path / "b", monkeypatch, results={"r1": "done.json"}, check=["no done.json"])
    assert code == 1 and read(queue, "final-check.json")["failures"] == ["r1: no done.json"]
    assert setup(tmp_path / "c", monkeypatch, results={"r1": "error.json"})[0] == 1
    assert json.loads((tmp_path / "c" / "q" / "final-check.json").read_text())["checks"] == {"r1": []}


def test_missing_pod_s3_settings_stop_before_any_read(tmp_path, monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    queue = Queue(str(tmp_path / "q"))
    monkeypatch.setattr(torch_reference.preflight, "source_preflight", lambda *args: pytest.fail("read"))
    assert torch_reference.run(queue, "s3://bucket/model", {"files": {}}, 2, 26, 4) == 2
    assert "AWS_ENDPOINT_URL" in read(queue, "source-audit-error.json")["error"]
