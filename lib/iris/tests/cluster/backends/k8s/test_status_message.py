# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes scheduling messages observed at the provider boundary."""

from iris.rpc import job_pb2

from .conftest import (
    gated_pod,
    imagepull_pod,
    observe_pod_update,
    singleton_gated_pod,
    singleton_unadmitted_workload,
    unadmitted_workload,
)


def test_status_message_surfaces_kueue_admission_verdict():
    update = observe_pod_update(gated_pod(), unadmitted_workload())
    assert update.status_message is not None
    assert "SchedulingGated" in update.status_message
    assert "couldn't assign flavors" in update.status_message
    assert 'excluded: resource "cpu"' in update.status_message
    assert "cw-use02a-lq" in update.status_message


def test_pod_status_singleton_surfaces_kueue_admission_verdict():
    update = observe_pod_update(singleton_gated_pod(), singleton_unadmitted_workload())
    assert update.status_message is not None
    assert "couldn't assign flavors" in update.status_message
    assert 'excluded: resource "cpu"' in update.status_message


def test_status_message_when_workload_not_yet_created():
    update = observe_pod_update(gated_pod())
    assert update.status_message is not None
    assert "SchedulingGated" in update.status_message
    assert "Kueue" in update.status_message


def test_status_message_surfaces_image_pull_error():
    update = observe_pod_update(imagepull_pod())
    assert update.status_message is not None
    assert "ImagePullBackOff" in update.status_message
    assert "ghcr.io/nope" in update.status_message


def test_status_message_empty_for_healthy_running_pod():
    pod = {
        "metadata": {},
        "status": {"phase": "Running", "containerStatuses": [{"name": "task", "state": {"running": {}}}]},
    }
    assert observe_pod_update(pod).status_message == ""


def test_task_update_carries_status_message_while_building():
    update = observe_pod_update(gated_pod(), unadmitted_workload())
    assert update.new_state == job_pb2.TASK_STATE_BUILDING
    assert update.status_message is not None
    assert "couldn't assign flavors" in update.status_message


def test_task_update_clears_status_message_on_running_and_terminal():
    running = {
        "metadata": {},
        "status": {"phase": "Running", "containerStatuses": [{"name": "task", "state": {"running": {}}}]},
    }
    assert observe_pod_update(running).status_message == ""

    failed = {
        "metadata": {},
        "status": {
            "phase": "Failed",
            "containerStatuses": [{"name": "task", "state": {"terminated": {"exitCode": 1}}}],
        },
    }
    assert observe_pod_update(failed).status_message == ""
