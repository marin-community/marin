# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes scheduling messages observed at the provider boundary."""

from copy import deepcopy

from iris.cluster.backends.k8s.tasks import K8sTaskProvider
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from .conftest import (
    gated_pod,
    imagepull_pod,
    make_batch,
    make_run_req,
    pod_config,
    singleton_gated_pod,
    singleton_unadmitted_workload,
    unadmitted_workload,
)


def _observe_pod(pod: dict, workload: dict | None = None):
    k8s = InMemoryK8sService(namespace="iris")
    provider = K8sTaskProvider(kubectl=k8s, pods=pod_config(), cluster_scan_interval=0.0)
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    try:
        provider.sync(make_batch(tasks_to_run=[make_run_req("/job/0")]))
        applied = k8s.list_json(K8sResource.PODS)[0]
        observed = deepcopy(pod)
        observed["kind"] = "Pod"
        observed_metadata = observed.setdefault("metadata", {})
        observed_metadata["name"] = applied["metadata"]["name"]
        observed_metadata["labels"] = {
            **applied["metadata"]["labels"],
            **observed_metadata.get("labels", {}),
        }
        k8s.seed_resource(K8sResource.PODS, observed_metadata["name"], observed)
        if workload is not None:
            workload_name = workload.get("metadata", {}).get("name", "workload")
            k8s.seed_resource(K8sResource.WORKLOADS, workload_name, deepcopy(workload))
        updates = provider.sync(make_batch(running_tasks=[entry]))
        assert len(updates) == 1
        return updates[0]
    finally:
        provider.close()


def test_status_message_surfaces_kueue_admission_verdict():
    update = _observe_pod(gated_pod(), unadmitted_workload())
    assert update.status_message is not None
    assert "SchedulingGated" in update.status_message
    assert "couldn't assign flavors" in update.status_message
    assert 'excluded: resource "cpu"' in update.status_message
    assert "cw-use02a-lq" in update.status_message


def test_pod_status_singleton_surfaces_kueue_admission_verdict():
    update = _observe_pod(singleton_gated_pod(), singleton_unadmitted_workload())
    assert update.status_message is not None
    assert "couldn't assign flavors" in update.status_message
    assert 'excluded: resource "cpu"' in update.status_message


def test_status_message_when_workload_not_yet_created():
    update = _observe_pod(gated_pod())
    assert update.status_message is not None
    assert "SchedulingGated" in update.status_message
    assert "Kueue" in update.status_message


def test_status_message_surfaces_image_pull_error():
    update = _observe_pod(imagepull_pod())
    assert update.status_message is not None
    assert "ImagePullBackOff" in update.status_message
    assert "ghcr.io/nope" in update.status_message


def test_status_message_empty_for_healthy_running_pod():
    pod = {
        "metadata": {},
        "status": {"phase": "Running", "containerStatuses": [{"name": "task", "state": {"running": {}}}]},
    }
    assert _observe_pod(pod).status_message == ""


def test_task_update_carries_status_message_while_building():
    update = _observe_pod(gated_pod(), unadmitted_workload())
    assert update.new_state == job_pb2.TASK_STATE_BUILDING
    assert update.status_message is not None
    assert "couldn't assign flavors" in update.status_message


def test_task_update_clears_status_message_on_running_and_terminal():
    running = {
        "metadata": {},
        "status": {"phase": "Running", "containerStatuses": [{"name": "task", "state": {"running": {}}}]},
    }
    assert _observe_pod(running).status_message == ""

    failed = {
        "metadata": {},
        "status": {
            "phase": "Failed",
            "containerStatuses": [{"name": "task", "state": {"terminated": {"exitCode": 1}}}],
        },
    }
    assert _observe_pod(failed).status_message == ""
