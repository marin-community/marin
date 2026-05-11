# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import contextmanager

import pytest
from iris.rpc import job_pb2
from rigging.redaction import REDACTED_VALUE

from scripts.workflows import iris_monitor


def _pod(name: str, *, phase: str = "Running", ready: bool = True, deleting: bool = False) -> dict:
    metadata = {"name": name}
    if deleting:
        metadata["deletionTimestamp"] = "2026-05-06T12:00:00Z"
    return {
        "metadata": metadata,
        "status": {
            "phase": phase,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


def _statuses(*pods: dict) -> list[iris_monitor.K8sPodStatus]:
    return iris_monitor._controller_pods_from_json(json.dumps({"items": list(pods)}))


def _job(job_id: str, state: str) -> job_pb2.JobStatus:
    return job_pb2.JobStatus(job_id=job_id, state=job_pb2.JobState.Value(state))


class _FakeClient:
    def __init__(self, polls: list[list[job_pb2.JobStatus]]) -> None:
        self._polls = iter(polls)

    def list_jobs(self, *, prefix=None):
        return next(self._polls)


def test_settled_coreweave_controller_requires_exactly_one_ready_pod() -> None:
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new"))) == "iris-controller-new"

    assert iris_monitor._settled_controller_pod_name(_statuses()) is None
    assert (
        iris_monitor._settled_controller_pod_name(
            _statuses(
                _pod("iris-controller-old", deleting=True),
                _pod("iris-controller-new"),
            )
        )
        is None
    )
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new", ready=False))) is None
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new", phase="Pending"))) is None


def test_wait_for_child_job_times_out_when_no_child_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the parent is queued long enough that no child reaches RUNNING, fail fast with a queue timeout."""
    parent_id = "/runner/parent"
    child_id = f"{parent_id}/grug-train-canary-tpu-1"
    fake = _FakeClient(
        [
            [_job(parent_id, "JOB_STATE_RUNNING"), _job(child_id, "JOB_STATE_PENDING")],
            [_job(parent_id, "JOB_STATE_RUNNING"), _job(child_id, "JOB_STATE_PENDING")],
        ]
    )

    @contextmanager
    def fake_open(**_kwargs):
        yield fake

    monkeypatch.setattr(iris_monitor, "_open_iris_client", fake_open)
    monkeypatch.setattr(iris_monitor.time, "sleep", lambda _s: None)
    times = iter([0.0, 100.0, 5000.0])
    monkeypatch.setattr(iris_monitor.time, "monotonic", lambda: next(times))

    with pytest.raises(TimeoutError, match="No child reached RUNNING"):
        iris_monitor.wait_for_child_job(
            parent_id,
            iris_config=None,
            controller_url=None,
            poll_interval=0,
            queue_timeout=3000,
            repo_root=iris_monitor._REPO_ROOT,
        )


def test_redact_pod_doc_redacts_env_values_and_preserves_context():
    pod = {
        "metadata": {"name": "worker-0"},
        "spec": {
            "containers": [
                {
                    "name": "runner",
                    "image": "registry.example/iris-runner:sha",
                    "resources": {"limits": {"nvidia.com/gpu": "8"}},
                    "env": [
                        {"name": "AWS_ACCESS_KEY_ID", "value": "AKIA_TEST_ACCESS"},
                        # Low-entropy secret only caught via name-based lift.
                        {"name": "WANDB_API_KEY", "value": "wandb-test-secret"},
                        {
                            "name": "IRIS_JOB_ENV",
                            "value": json.dumps(
                                {
                                    "AWS_SECRET_ACCESS_KEY": "nested-secret-key",
                                    "HF_TOKEN": "nested-hf-token",
                                    "LOG_LEVEL": "debug",
                                }
                            ),
                        },
                        {"name": "NORMAL_ENV", "value": "normal-env-value"},
                        {
                            "name": "HF_TOKEN",
                            "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}},
                        },
                    ],
                }
            ]
        },
    }

    redacted = iris_monitor._redact_pod_doc(pod)
    env_by_name = {entry["name"]: entry for entry in redacted["spec"]["containers"][0]["env"]}

    assert env_by_name["AWS_ACCESS_KEY_ID"]["value"] == REDACTED_VALUE
    assert env_by_name["WANDB_API_KEY"]["value"] == REDACTED_VALUE
    assert env_by_name["NORMAL_ENV"]["value"] == "normal-env-value"

    nested = json.loads(env_by_name["IRIS_JOB_ENV"]["value"])
    assert nested == {
        "AWS_SECRET_ACCESS_KEY": REDACTED_VALUE,
        "HF_TOKEN": REDACTED_VALUE,
        "LOG_LEVEL": "debug",
    }

    # valueFrom entries pass through untouched and never gain a phantom `value`.
    assert "value" not in env_by_name["HF_TOKEN"]
    assert env_by_name["HF_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "hf-token"

    # Non-env pod context stays intact.
    assert redacted["spec"]["containers"][0]["image"] == "registry.example/iris-runner:sha"
    assert redacted["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "8"
