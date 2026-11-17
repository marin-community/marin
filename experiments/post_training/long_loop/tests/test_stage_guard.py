# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig
from iris.client.workload_codec import job_status_from_proto, task_status_from_proto
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.timing import Timestamp

from experiments.post_training.long_loop import stage_guard
from experiments.post_training.long_loop.stage_guard import StageGuard, run_guarded_stage


class StageController:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.jobs = {}
        self.requests = []
        self.cancelled = []

    def submit(self, request, *, adopt_existing):
        assert not adopt_existing
        self.requests.append(request)
        identity = "/benfeuer/exp3-smoke/" + request.name
        self.jobs[identity] = next(self.outcomes)
        return SimpleNamespace(job_id=identity)

    def job(self, identity):
        name = identity.to_wire()
        assert name in self.jobs  # A parent or sibling must never be cancelled.

        def status():
            state, error = self.jobs[name]
            return job_status_from_proto(job_pb2.JobStatus(job_id=name, state=state, error=error))

        return SimpleNamespace(status=status, cancel=lambda: self.cancelled.append(name))

    def list_tasks(self, identity):
        state, _ = self.jobs[identity.to_wire()]
        tasks = []
        for index in range(2):
            task = job_pb2.TaskStatus(task_id=identity.to_wire() + f"/{index}")
            task.submitted_at.epoch_ms = 1
            task.started_at.epoch_ms = 60000
            if state != job_pb2.JOB_STATE_RUNNING:
                task.finished_at.epoch_ms = 120000
            tasks.append(task_status_from_proto(task))
        return tasks


def bind_controller(monkeypatch, outcomes):
    controller = StageController(outcomes)
    monkeypatch.setattr(stage_guard, "current_client", lambda: controller)
    monkeypatch.setattr(
        stage_guard,
        "iris_ctx",
        lambda: SimpleNamespace(client=controller, job_id=JobName.from_wire("/benfeuer/exp3-smoke")),
    )
    monkeypatch.setattr(Timestamp, "now", lambda: Timestamp.from_seconds(120))
    return controller


def request():
    return JobRequest(
        name="sft",
        entrypoint=Entrypoint.from_binary("python", ["-V"]),
        resources=ResourceConfig.with_gpu("H100", count=8, replicas=2),
    )


def test_aggregate_watchdog_text_retries_twice_with_fresh_identities_and_charges_each_worker_once(tmp_path, monkeypatch):
    controller = bind_controller(
        monkeypatch,
        [
            (job_pb2.JOB_STATE_FAILED, "No progress after process_started"),
            (job_pb2.JOB_STATE_FAILED, "process exited with exit code 124"),
            (job_pb2.JOB_STATE_SUCCEEDED, ""),
        ],
    )
    receipt = run_guarded_stage(request(), StageGuard(1.0, 10, 2), str(tmp_path))
    assert len(controller.requests) == 3
    assert len({item.name for item in controller.requests}) == 3
    assert receipt["h100_hours"] == pytest.approx(0.8)
    assert not controller.cancelled
    assert [json.loads((tmp_path / f"attempt-{i}.json").read_text())["result"] for i in range(3)] == [
        "failed",
        "failed",
        "succeeded",
    ]


def test_budget_guard_records_spend_and_cancels_only_the_active_child(tmp_path, monkeypatch):
    controller = bind_controller(monkeypatch, [(job_pb2.JOB_STATE_RUNNING, "")])
    with pytest.raises(RuntimeError):
        run_guarded_stage(request(), StageGuard(0.3, 10, 2), str(tmp_path))
    assert controller.cancelled == ["/benfeuer/exp3-smoke/sft"]
    receipt = json.loads((tmp_path / "attempt-0.json").read_text())
    assert receipt["result"] == "budget_exhausted"
    assert receipt["h100_hours"] == pytest.approx(16 / 60)
    assert receipt["h100_hour_limit"] == pytest.approx(0.3)
    assert len(controller.requests) == 1


def test_other_failures_do_not_replay_the_stage(tmp_path, monkeypatch):
    controller = bind_controller(monkeypatch, [(job_pb2.JOB_STATE_FAILED, "invalid checkpoint shape")])
    with pytest.raises(RuntimeError, match="invalid checkpoint shape"):
        run_guarded_stage(request(), StageGuard(1.0, 10, 2), str(tmp_path))
    assert len(controller.requests) == 1
    assert not controller.cancelled
