# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-task ``status_message``: the one-liner explaining why a k8s task is stuck
in BUILDING (Kueue admission verdict, image-pull error), harvested from the pod and
its Kueue Workload and carried on ``TaskUpdate.status_message``. This is the signal
that turns a silent "Building 18m" into an explained wait on the dashboard and, once
mirrored, on a federating hub."""

from iris.cluster.backends.k8s.tasks import (
    _KUEUE_POD_GROUP_NAME,
    _TASK_CONTAINER_NAME,
    _pod_status_message,
    _task_update_from_pod,
)
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.types import JobName
from iris.rpc import job_pb2

# The real Kueue message an over-large GPU request produces (cpu=160 on 128-vCPU H100
# nodes under InfiniBand TAS) — the motivating incident.
_KUEUE_UNADMITTED_MSG = (
    "couldn't assign flavors to pod set main: topology \"infiniband\" doesn't allow to "
    'fit any of 1 pod(s). Total nodes: 32; excluded: resource "cpu": 32'
)


def _gated_pod(name: str = "iris-job-0-0", pod_group: str = "wl-abc") -> dict:
    """A Pending pod blocked on a Kueue scheduling gate (no container has started)."""
    return {
        "metadata": {"name": name, "labels": {_KUEUE_POD_GROUP_NAME: pod_group}},
        "status": {
            "phase": "Pending",
            "containerStatuses": [],
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "SchedulingGated",
                    "message": "Scheduling is blocked due to non-empty scheduling gates",
                }
            ],
        },
    }


def _unadmitted_workload(name: str = "wl-abc", msg: str = _KUEUE_UNADMITTED_MSG) -> dict:
    return {
        "metadata": {"name": name},
        "spec": {"queueName": "cw-use02a-lq"},
        "status": {"conditions": [{"type": "QuotaReserved", "status": "False", "reason": "Pending", "message": msg}]},
    }


def _imagepull_pod(name: str = "iris-job-0-0") -> dict:
    return {
        "metadata": {"name": name},
        "status": {
            "phase": "Pending",
            "containerStatuses": [
                {
                    "name": _TASK_CONTAINER_NAME,
                    "state": {
                        "waiting": {"reason": "ImagePullBackOff", "message": 'Back-off pulling image "ghcr.io/nope"'}
                    },
                }
            ],
        },
    }


def test_status_message_surfaces_kueue_admission_verdict():
    """A SchedulingGated pod's message carries the Workload's 'couldn't assign flavors'
    verdict — an over-large GPU request Kueue can never admit, previously invisible."""
    msg = _pod_status_message(_gated_pod(), _unadmitted_workload())
    assert "SchedulingGated" in msg
    assert "couldn't assign flavors" in msg
    assert 'excluded: resource "cpu"' in msg
    assert "cw-use02a-lq" in msg


def test_status_message_when_workload_not_yet_created():
    """Gated but no Workload seen yet: still explain the wait rather than go silent."""
    msg = _pod_status_message(_gated_pod(), None)
    assert "SchedulingGated" in msg
    assert "Kueue" in msg


def test_status_message_surfaces_image_pull_error():
    """An image-pull failure surfaces the container waiting reason + registry detail."""
    msg = _pod_status_message(_imagepull_pod(), None)
    assert "ImagePullBackOff" in msg
    assert "ghcr.io/nope" in msg


def test_status_message_empty_for_healthy_running_pod():
    """A running pod with a live container has nothing to say — the message clears."""
    pod = {
        "metadata": {"name": "iris-job-0-0"},
        "status": {"phase": "Running", "containerStatuses": [{"name": _TASK_CONTAINER_NAME, "state": {"running": {}}}]},
    }
    assert _pod_status_message(pod, None) == ""


def test_task_update_carries_status_message_while_building():
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    update = _task_update_from_pod(entry, _gated_pod(), _unadmitted_workload())
    assert update.new_state == job_pb2.TASK_STATE_BUILDING
    assert update.status_message is not None
    assert "couldn't assign flavors" in update.status_message


def test_task_update_clears_status_message_on_running_and_terminal():
    """Running/terminal updates set status_message to "" so a stale BUILDING reason
    does not linger on a task that has moved on."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    running = {
        "metadata": {"name": "iris-job-0-0"},
        "status": {"phase": "Running", "containerStatuses": [{"name": _TASK_CONTAINER_NAME, "state": {"running": {}}}]},
    }
    assert _task_update_from_pod(entry, running).status_message == ""

    failed = {
        "metadata": {"name": "iris-job-0-0"},
        "status": {
            "phase": "Failed",
            "containerStatuses": [{"name": _TASK_CONTAINER_NAME, "state": {"terminated": {"exitCode": 1}}}],
        },
    }
    assert _task_update_from_pod(entry, failed).status_message == ""
