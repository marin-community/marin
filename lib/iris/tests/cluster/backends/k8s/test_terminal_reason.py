# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-attempt terminal reasons observed through Kubernetes reconciliation."""

from copy import deepcopy

from iris.cluster.backends.k8s.tasks import K8sTaskProvider
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.types import JobName

from .conftest import make_batch, make_run_req, pod_config

TERMINAL_REASON_MAX_CHARS = 500


def _init_status(name: str, exit_code: int | None, reason: str = "Error", message: str = "") -> dict:
    if exit_code is None:
        return {"name": name, "state": {}}
    return {"name": name, "state": {"terminated": {"exitCode": exit_code, "reason": reason, "message": message}}}


def _task_terminated(message: str) -> dict:
    return {"name": "task", "state": {"terminated": {"exitCode": 1, "reason": "Error", "message": message}}}


def _pod(*, init: list[dict] | None = None, containers: list[dict] | None = None) -> dict:
    return {
        "status": {
            "phase": "Failed",
            "initContainerStatuses": init or [],
            "containerStatuses": containers or [],
        }
    }


def _terminal_reason(pod: dict) -> str | None:
    k8s = InMemoryK8sService(namespace="iris")
    provider = K8sTaskProvider(kubectl=k8s, pods=pod_config(), cluster_scan_interval=0.0)
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    try:
        provider.sync(make_batch(tasks_to_run=[make_run_req("/job/0")]))
        applied = k8s.list_json(K8sResource.PODS)[0]
        observed = deepcopy(pod)
        observed["kind"] = "Pod"
        observed["metadata"] = deepcopy(applied["metadata"])
        k8s.seed_resource(K8sResource.PODS, observed["metadata"]["name"], observed)
        updates = provider.sync(make_batch(running_tasks=[entry]))
        assert len(updates) == 1
        return updates[0].terminal_reason
    finally:
        provider.close()


def test_reports_stage_workdir_404():
    pod = _pod(init=[_init_status("stage-workdir", 1, "Error", "Bundle fetch abc failed: HTTP Error 404")])
    assert _terminal_reason(pod) == "Init:Error stage-workdir: Bundle fetch abc failed: HTTP Error 404"


def test_ignores_log_shipper_sidecar():
    assert _terminal_reason(_pod(init=[_init_status("log-shipper", 1, "Error", "sidecar crashed")])) is None


def test_none_when_init_completed():
    assert _terminal_reason(_pod(init=[_init_status("stage-workdir", 0, "Completed")])) is None


def test_none_when_init_still_running():
    assert _terminal_reason(_pod(init=[_init_status("stage-workdir", None)])) is None


def test_none_when_no_init_statuses():
    assert _terminal_reason(_pod()) is None


def test_omits_empty_message():
    assert _terminal_reason(_pod(init=[_init_status("stage-workdir", 2, "Error", "")])) == "Init:Error stage-workdir"


def test_terminal_reason_prefers_init_over_task_container():
    pod = _pod(init=[_init_status("stage-workdir", 1, "Error", "init boom")], containers=[_task_terminated("task boom")])
    assert _terminal_reason(pod) == "Init:Error stage-workdir: init boom"


def test_terminal_reason_falls_back_to_task_container():
    assert _terminal_reason(_pod(containers=[_task_terminated("task boom")])) == "task boom"


def test_terminal_reason_is_bounded():
    pod = _pod(init=[_init_status("stage-workdir", 1, "Error", "x" * 2000)])
    assert len(_terminal_reason(pod) or "") == TERMINAL_REASON_MAX_CHARS
