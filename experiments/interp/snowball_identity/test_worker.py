# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The worker on a local queue with the tiny random GrugMoE."""

import json

import pytest

from experiments.interp.snowball_identity import worker
from experiments.interp.snowball_identity.tiny_model import FakeTokenizer


@pytest.fixture
def queue(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "load_harness", lambda weights: harness)
    monkeypatch.setattr(worker, "load_tokenizer", lambda weights: FakeTokenizer())
    monkeypatch.setattr(worker, "stop_tokens", lambda weights: set())
    return worker.Queue(str(tmp_path / "queue"))


def test_serve_runs_specs_and_reports_errors(queue, tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "a", "text": "t5 t17 t3 t99", "targets": ["t40", "t41"]}) + "\n")
    worker.submit(queue, {"id": "cap", "command": "capture", "args": ["--prompts", str(prompts)]}, wait=False)
    worker.submit(queue, {"id": "lens", "command": "lens", "prompts": str(prompts)}, wait=False)
    worker.submit(queue, {"id": "bad", "command": "lens"}, wait=False)
    with pytest.raises(ValueError, match="set by the worker"):
        worker.submit(queue, {"command": "capture", "args": ["--output", "x"]}, wait=False)
    with pytest.raises(ValueError, match="Unknown command"):
        worker.submit(queue, {"command": "exec"}, wait=False)
    with pytest.raises(ValueError, match="already in use"):
        worker.submit(queue, {"id": "cap", "command": "lens"}, wait=False)
    queue.write({"command": "lens"}, "running", "orphan.json")

    worker.serve(queue, "weights", idle_minutes=0)

    assert queue.read("results", "orphan", "error.json")["error"].startswith("the worker running this spec")
    assert queue.read("results", "cap", "done.json")["seconds"] >= 0
    assert queue.read("results", "lens", "done.json")
    error = queue.read("results", "bad", "error.json")
    assert "KeyError" in error["error"] and error["spec"]["id"] == "bad"
    (row,) = [json.loads(line) for line in (tmp_path / "queue/results/cap/capture.jsonl").read_text().splitlines()]
    assert set(row["targets"]) == {"t40", "t41"}
    status = worker.status(queue)
    assert status["inbox"] == [] and status["running"] == []
    assert status["worker"]["state"] == "stopped" and status["worker"]["jobs_done"] == 3


def test_stop_file(queue):
    worker.submit(queue, {"id": "later", "command": "lens", "prompts": "unused"}, wait=False)
    queue.write({}, "stop.json")
    worker.serve(queue, "weights", idle_minutes=60)
    assert worker.status(queue)["inbox"] == ["later"]
    assert queue.read("stop.json") is None and queue.read("worker.json")["reason"] == "stop.json"
