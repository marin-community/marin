# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the k8s source: flattening of canned API-server JSON, pagination,
error classification, and the fleet's always-one-row-per-cluster alert contract."""

from dataclasses import asdict
from datetime import UTC, datetime

import httpx
import pytest
from conftest import (
    IRIS_DEPLOY,
    KUEUE_DEPLOY,
    KUEUE_SLICES,
    bridge_config,
    deployment,
    healthy_k8s_routes,
    k8s_api,
    make_k8s_source,
    pod,
)
from github_source import GithubSource
from k8s_source import K8sError, K8sErrorClass, K8sFleet
from server import create_app
from starlette.testclient import TestClient
from wandb_source import WandbSource

GPU_RESOURCE = "nvidia.com/gpu"
STALE_DELETION_TIMESTAMP = "2000-01-01T00:00:00Z"


def _workload(name: str, queue: str, *, conditions: list | None = None, created: str = "2026-07-19T00:00:00Z") -> dict:
    return {
        "metadata": {"namespace": "iris", "name": name, "creationTimestamp": created},
        "spec": {"queueName": queue},
        "status": {"conditions": conditions or []},
    }


def _namespace(name: str) -> dict:
    return {"metadata": {"name": name}}


def _termination_candidate(
    name: str,
    *,
    node: str | None = None,
    timestamp: str = STALE_DELETION_TIMESTAMP,
    phase: str = "Running",
    finalizers: list[str] | None = None,
) -> dict:
    manifest = pod("iris", name)
    manifest["metadata"]["deletionTimestamp"] = timestamp
    if finalizers:
        manifest["metadata"]["finalizers"] = finalizers
    manifest["spec"]["containers"] = [{"name": "task", "resources": {}}]
    if node is not None:
        manifest["spec"]["nodeName"] = node
    manifest["status"]["phase"] = phase
    return manifest


def _with_gpu(manifest: dict, quantity: str) -> dict:
    manifest["spec"]["containers"][0]["resources"]["limits"] = {GPU_RESOURCE: quantity}
    return manifest


def _with_task_attempt(manifest: dict, task_attempt: str) -> dict:
    manifest["spec"]["containers"][0]["env"] = [{"name": "IRIS_TASK_ID", "value": task_attempt}]
    return manifest


# --- K8sSource --------------------------------------------------------------


def test_control_plane_flattens_components_and_webhooks():
    routes = healthy_k8s_routes()
    routes[KUEUE_DEPLOY] = deployment("kueue-system", "kueue-controller-manager", ready=0)
    routes["/api/v1/namespaces/kueue-system/pods"] = [
        pod("kueue-system", "kueue-controller-manager-abc", waiting="CrashLoopBackOff", restarts=7)
    ]
    routes[KUEUE_SLICES] = [
        # nil ready counts as ready per the EndpointSlice contract; False does not.
        {"endpoints": [{"conditions": {"ready": True}}, {"conditions": {}}]},
        {"endpoints": [{"conditions": {"ready": False}}]},
    ]
    rows = make_k8s_source(k8s_api(routes)).control_plane()

    kueue = rows[0]
    assert kueue == {
        "kind": "component",
        "component": "kueue-system/kueue-controller-manager",
        "ready": 0,
        "desired": 1,
        "restarts": 7,
        "waiting_reason": "CrashLoopBackOff",
    }
    assert rows[-1] == {"kind": "webhook", "component": "kueue-system/kueue-webhook-service", "ready_endpoints": 2}


def test_missing_deployment_reads_as_degraded_not_healthy():
    routes = healthy_k8s_routes()
    del routes[IRIS_DEPLOY]
    rows = make_k8s_source(k8s_api(routes)).control_plane()
    iris = next(row for row in rows if row["component"] == "iris/iris-controller")
    assert iris["ready"] == 0 and iris["desired"] == 1 and iris["waiting_reason"] == "Missing"


def test_list_follows_continue_pagination():
    pages = [
        {"items": [pod("iris", "task-1", waiting="CrashLoopBackOff")], "metadata": {"continue": "tok"}},
        {"items": [pod("iris", "task-2", waiting="ImagePullBackOff")], "metadata": {}},
    ]
    seen_continues = []

    def pods(request: httpx.Request) -> httpx.Response:
        seen_continues.append(request.url.params.get("continue"))
        return httpx.Response(200, json=pages[len(seen_continues) - 1])

    routes = {"/api/v1/namespaces": [_namespace("iris")], "/api/v1/namespaces/iris/pods": pods}
    rows = make_k8s_source(k8s_api(routes)).crashloops()
    assert [row["pod"] for row in rows] == ["task-1", "task-2"]
    assert seen_continues == [None, "tok"]


def test_429_is_retried_once_after_retry_after():
    responses = [httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, json={"gitVersion": "v1"})]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    source = make_k8s_source(handler)
    assert isinstance(source.probe(), int)
    assert not responses


@pytest.mark.parametrize(
    ("failure", "expected_class"),
    [
        (lambda request: httpx.Response(401, json={}), K8sErrorClass.AUTH),
        (lambda request: httpx.Response(403, json={}), K8sErrorClass.AUTH),
        (lambda request: httpx.Response(500, json={}), K8sErrorClass.HTTP),
        (lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused", request=request)), K8sErrorClass.NETWORK),
        (lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=request)), K8sErrorClass.TIMEOUT),
    ],
)
def test_failures_are_classified(failure, expected_class):
    with pytest.raises(K8sError) as excinfo:
        make_k8s_source(failure).probe()
    assert excinfo.value.error_class == expected_class


def test_missing_token_is_an_auth_error_without_a_network_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent without a token")

    with pytest.raises(K8sError) as excinfo:
        make_k8s_source(handler, token=None).probe()
    assert excinfo.value.error_class == K8sErrorClass.AUTH


def test_crashloop_scope_separates_watched_components_from_workloads():
    routes = {
        "/api/v1/namespaces": [_namespace("iris")],
        "/api/v1/namespaces/iris/pods": [
            pod("iris", "iris-controller-7f9-x2", waiting="CrashLoopBackOff", restarts=3),
            pod("iris", "some-user-task-0", waiting="ImagePullBackOff"),
            pod("iris", "healthy-task-0"),
        ],
    }
    rows = make_k8s_source(k8s_api(routes)).crashloops()
    assert [(row["pod"], row["scope"], row["reason"]) for row in rows] == [
        ("iris-controller-7f9-x2", "control-plane", "CrashLoopBackOff"),
        ("some-user-task-0", "workload", "ImagePullBackOff"),
    ]


def test_provider_namespaces_are_excluded_from_pod_scans():
    # Only the iris pods route exists: a scan reaching cw-* or kube-* would 404
    # and raise, so a passing scan proves the exclusion.
    routes = {
        "/api/v1/namespaces": [_namespace("cw-exporters"), _namespace("kube-system"), _namespace("iris")],
        "/api/v1/namespaces/iris/pods": [pod("iris", "task-0", waiting="CrashLoopBackOff")],
    }
    assert [row["pod"] for row in make_k8s_source(k8s_api(routes)).crashloops()] == ["task-0"]


def test_pending_splits_gated_from_pending_and_sorts_oldest_first():
    unschedulable = [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}]
    gated = [{"type": "PodScheduled", "status": "False", "reason": "SchedulingGated"}]
    routes = {
        "/api/v1/namespaces": [_namespace("iris")],
        "/api/v1/namespaces/iris/pods": [
            pod("iris", "young-gated", created="2026-07-19T12:00:00Z", conditions=gated),
            pod("iris", "old-stuck", created="2026-07-01T00:00:00Z", conditions=unschedulable),
        ],
    }
    rows = make_k8s_source(k8s_api(routes)).pending()
    assert [(row["pod"], row["state"]) for row in rows] == [
        ("old-stuck", "pending"),
        ("young-gated", "scheduling_gated"),
    ]
    assert rows[0]["reason"] == "Unschedulable"
    assert rows[0]["age_seconds"] > rows[1]["age_seconds"]


def test_kueue_counts_unadmitted_per_queue_skipping_admitted_and_finished():
    admitted = [{"type": "Admitted", "status": "True"}]
    finished = [{"type": "Finished", "status": "True"}]
    routes = {
        "/apis/kueue.x-k8s.io/v1beta2/workloads": [
            _workload("running", "q1", conditions=admitted),
            _workload("done", "q1", conditions=finished),
            _workload("waiting-old", "q1", created="2026-07-01T00:00:00Z"),
            _workload("waiting-new", "q1", created="2026-07-19T00:00:00Z"),
            _workload("waiting-other", "q2"),
        ]
    }
    rows = make_k8s_source(k8s_api(routes)).kueue()
    assert [(row["queue"], row["unadmitted"]) for row in rows] == [("q1", 2), ("q2", 1)]
    assert rows[0]["oldest_age_seconds"] > rows[1]["oldest_age_seconds"]


def test_warning_events_flatten_newest_first():
    routes = {
        "/api/v1/events": [
            {
                "involvedObject": {"kind": "Pod", "name": "task-0", "namespace": "iris"},
                "reason": "FailedScheduling",
                "message": "0/5 nodes are available",
                "count": 4,
                "lastTimestamp": "2026-07-19T10:00:00Z",
            },
            {
                "involvedObject": {
                    "kind": "Deployment",
                    "name": "kueue-controller-manager",
                    "namespace": "kueue-system",
                },
                "reason": "BackOff",
                "message": "x" * 500,
                "lastTimestamp": "2026-07-19T11:00:00Z",
            },
        ]
    }
    rows = make_k8s_source(k8s_api(routes)).warning_events()
    assert [row["object"] for row in rows] == ["Deployment/kueue-controller-manager", "Pod/task-0"]
    assert rows[1]["count"] == 4
    assert len(rows[0]["message"]) == 200


def test_terminating_classifies_node_and_api_cleanup_cases():
    recent = datetime.now(UTC).isoformat()
    labels = {"iris.task_id": "sanitized.task", "iris.job_id": "sanitized.job"}
    stuck_gpu = _with_task_attempt(
        _with_gpu(_termination_candidate("stuck-gpu", node="node-a"), "4"), "/user/full-job/0:3"
    )
    stuck_gpu["metadata"].update({"deletionGracePeriodSeconds": 30, "labels": labels})
    routes = {
        "/api/v1/namespaces": [_namespace("iris")],
        "/api/v1/namespaces/iris/pods": [
            stuck_gpu,
            _termination_candidate("finalizer", node="node-b", finalizers=["z", "a"]),
            _termination_candidate("terminal", node="node-c", phase="Succeeded"),
            _termination_candidate("unbound"),
            _with_gpu(_termination_candidate("invalid", node="node-d", timestamp="not-a-time"), "1"),
            _termination_candidate("within-threshold", node="node-e", timestamp=recent),
        ],
    }
    rows = make_k8s_source(k8s_api(routes)).termination_candidates()
    by_name = {row.pod: row for row in rows}

    assert set(by_name) == {"stuck-gpu", "finalizer", "terminal", "unbound", "invalid"}
    stuck_gpu = asdict(by_name["stuck-gpu"])
    assert stuck_gpu.pop("cluster") == "cw-a"
    overdue_seconds = stuck_gpu.pop("overdue_seconds")
    assert stuck_gpu == {
        "namespace": "iris",
        "pod": "stuck-gpu",
        "node": "node-a",
        "phase": "Running",
        "deletion_timestamp": STALE_DELETION_TIMESTAMP,
        "deletion_grace_seconds": 30,
        "gpu_count": 4,
        "task_attempt": "/user/full-job/0:3",
        "task_label": "sanitized.task",
        "job_label": "sanitized.job",
        "priority_class": "",
        "finalizers": "",
        "classification": "node_cleanup",
    }
    assert overdue_seconds > 0
    assert by_name["finalizer"].classification == "finalizer"
    assert by_name["finalizer"].finalizers == "a,z"
    assert by_name["terminal"].classification == "terminal"
    assert by_name["unbound"].classification == "unbound"
    assert by_name["invalid"].classification == "invalid_timestamp"
    assert by_name["invalid"].overdue_seconds is None


def test_terminating_gpu_count_uses_requests_limits_and_init_peak():
    manifest = _with_gpu(_termination_candidate("mixed-resources", node="node-a"), "2")
    manifest["spec"]["containers"][0]["resources"]["requests"] = {GPU_RESOURCE: "1"}
    manifest["spec"]["containers"].append({"name": "sidecar", "resources": {"requests": {GPU_RESOURCE: "1"}}})
    manifest["spec"]["initContainers"] = [
        {"name": "setup", "resources": {"limits": {GPU_RESOURCE: "4"}}},
        {
            "name": "restartable",
            "restartPolicy": "Always",
            "resources": {"limits": {GPU_RESOURCE: "1"}},
        },
    ]
    routes = {
        "/api/v1/namespaces": [_namespace("iris")],
        "/api/v1/namespaces/iris/pods": [manifest],
    }
    (row,) = make_k8s_source(k8s_api(routes)).termination_candidates()
    assert row.gpu_count == 4


def test_terminating_rejects_an_invalid_gpu_quantity():
    routes = {
        "/api/v1/namespaces": [_namespace("iris")],
        "/api/v1/namespaces/iris/pods": [_with_gpu(_termination_candidate("invalid-gpu", node="node-a"), "many")],
    }
    with pytest.raises(ValueError):
        make_k8s_source(k8s_api(routes)).termination_candidates()


# --- K8sFleet ---------------------------------------------------------------


def _fleet(*handlers_by_name: tuple[str, object]) -> K8sFleet:
    return K8sFleet([make_k8s_source(handler, name=name) for name, handler in handlers_by_name])


def _forbidden(request: httpx.Request) -> httpx.Response:
    return httpx.Response(403, json={})


def test_fleet_stamps_cluster_and_keeps_healthy_clusters_on_partial_failure():
    fleet = _fleet(("cw-a", k8s_api(healthy_k8s_routes())), ("cw-b", _forbidden))
    rows = fleet.control_plane()
    healthy = [row for row in rows if row["cluster"] == "cw-a"]
    assert len(healthy) == 5  # 4 components + 1 webhook
    (error_row,) = [row for row in rows if row["cluster"] == "cw-b"]
    assert error_row["error_class"] == "auth"
    assert "403" in error_row["error"]


def test_alert_routes_return_explicit_zeros_when_healthy():
    fleet = _fleet(("cw-a", k8s_api(healthy_k8s_routes())))
    assert fleet.alert_unreachable() == [{"cluster": "cw-a", "error_class": "none", "value": 0}]
    assert fleet.alert_crashloops() == [
        {"cluster": "cw-a", "scope": "control-plane", "value": 0},
        {"cluster": "cw-a", "scope": "workload", "value": 0},
    ]
    assert fleet.alert_webhook_ready() == [
        {"cluster": "cw-a", "webhook": "kueue-system/kueue-webhook-service", "value": 1}
    ]
    assert fleet.alert_degraded() == [
        {"cluster": "cw-a", "component": "kueue-system/kueue-controller-manager", "value": 0},
        {"cluster": "cw-a", "component": "iris/iris-controller", "value": 0},
        {"cluster": "cw-a", "component": "traefik/traefik", "value": 0},
        {"cluster": "cw-a", "component": "cert-manager/cert-manager", "value": 0},
    ]
    assert fleet.alert_stuck_gpu_pods() == [{"cluster": "cw-a", "node": "", "value": 0}]


def test_alert_routes_keep_one_row_per_cluster_when_unreachable():
    # Zeros everywhere except unreachable: no fabricated health evidence, and only
    # webhook_ready (where zero means empty) also fires alongside unreachable.
    fleet = _fleet(("cw-a", _forbidden))
    assert fleet.alert_unreachable() == [{"cluster": "cw-a", "error_class": "auth", "value": 1}]
    assert {row["value"] for row in fleet.alert_crashloops()} == {0}
    assert fleet.alert_webhook_ready() == [
        {"cluster": "cw-a", "webhook": "kueue-system/kueue-webhook-service", "value": 0}
    ]
    assert {row["value"] for row in fleet.alert_degraded()} == {0}
    assert {row["value"] for row in fleet.alert_stuck_gpu_pods()} == {0}


def test_stuck_gpu_alert_groups_only_node_cleanup_rows_by_node():
    routes_a = healthy_k8s_routes()
    routes_a["/api/v1/namespaces"] = [_namespace("iris")]
    routes_a["/api/v1/namespaces/iris/pods"] = [
        _with_task_attempt(_with_gpu(_termination_candidate("task-b", node="node-a"), "2"), "/u/job/1:2"),
        _with_task_attempt(_with_gpu(_termination_candidate("task-a", node="node-a"), "1"), "/u/job/0:2"),
        _with_gpu(_termination_candidate("terminal", node="node-b", phase="Failed"), "4"),
        _with_gpu(_termination_candidate("finalizer", node="node-c", finalizers=["x"]), "4"),
        _with_gpu(_termination_candidate("unbound"), "4"),
        _termination_candidate("cpu-only", node="node-d"),
    ]
    fleet = _fleet(("cw-a", k8s_api(routes_a)), ("cw-b", k8s_api(healthy_k8s_routes())))
    rows = fleet.alert_stuck_gpu_pods(fleet.termination_candidates())
    assert rows == [
        {
            "cluster": "cw-a",
            "node": "node-a",
            "value": 2,
        },
        {"cluster": "cw-b", "node": "", "value": 0},
    ]


def test_crashloop_alert_counts_by_scope():
    routes = healthy_k8s_routes()
    routes["/api/v1/namespaces"] = [_namespace("iris")]
    routes["/api/v1/namespaces/iris/pods"] = [
        pod("iris", "iris-controller-7f9-x2", waiting="CrashLoopBackOff"),
        pod("iris", "task-a-0", waiting="CrashLoopBackOff"),
        pod("iris", "task-b-0", waiting="ImagePullBackOff"),
    ]
    assert _fleet(("cw-a", k8s_api(routes))).alert_crashloops() == [
        {"cluster": "cw-a", "scope": "control-plane", "value": 1},
        {"cluster": "cw-a", "scope": "workload", "value": 2},
    ]


# --- endpoints --------------------------------------------------------------


def _client(fleet: K8sFleet) -> TestClient:
    return TestClient(
        create_app(bridge_config(), {}, {}, GithubSource(token=None, timeout=5.0), fleet, WandbSource(timeout=5.0))
    )


def test_k8s_routes_serve_fleet_rows():
    client = _client(_fleet(("cw-a", k8s_api(healthy_k8s_routes()))))
    for path in (
        "/k8s/control_plane",
        "/k8s/crashloops",
        "/k8s/pending",
        "/k8s/kueue",
        "/k8s/events",
    ):
        assert client.get(path).status_code == 200
    health = client.get("/k8s/health").json()
    assert health[0]["cluster"] == "cw-a" and health[0]["reachable"] is True


def test_stuck_termination_routes_return_classification_and_alert_projection():
    routes = healthy_k8s_routes()
    routes["/api/v1/namespaces"] = [_namespace("iris")]
    routes["/api/v1/namespaces/iris/pods"] = [
        _with_task_attempt(_with_gpu(_termination_candidate("task-0", node="gpu-node"), "4"), "/user/job/0:1")
    ]
    client = _client(_fleet(("cw-a", k8s_api(routes))))

    (terminating,) = client.get("/k8s/termination_candidates").json()
    assert terminating["cluster"] == "cw-a"
    assert terminating["pod"] == "task-0"
    assert terminating["classification"] == "node_cleanup"
    assert client.get("/k8s/alerts/stuck_gpu_pods").json() == [
        {
            "cluster": "cw-a",
            "node": "gpu-node",
            "value": 1,
        }
    ]


def test_alerts_crashloops_scope_param_filters_rows():
    client = _client(_fleet(("cw-a", k8s_api(healthy_k8s_routes()))))
    rows = client.get("/k8s/alerts/crashloops", params={"scope": "control-plane"}).json()
    assert rows == [{"cluster": "cw-a", "scope": "control-plane", "value": 0}]
