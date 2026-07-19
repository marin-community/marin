# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the k8s source: flattening of canned API-server JSON, pagination,
error classification, and the fleet's always-one-row-per-cluster alert contract."""

import httpx
import pytest
from config import BridgeConfig, K8sClusterTarget
from github_source import GithubSource
from k8s_source import K8sError, K8sErrorClass, K8sFleet, K8sSource
from server import create_app
from starlette.testclient import TestClient

KUEUE_DEPLOY = "/apis/apps/v1/namespaces/kueue-system/deployments/kueue-controller-manager"
IRIS_DEPLOY = "/apis/apps/v1/namespaces/iris/deployments/iris-controller"
TRAEFIK_DEPLOY = "/apis/apps/v1/namespaces/traefik/deployments/traefik"
CERT_DEPLOY = "/apis/apps/v1/namespaces/cert-manager/deployments/cert-manager"
KUEUE_SLICES = "/apis/discovery.k8s.io/v1/namespaces/kueue-system/endpointslices"


def _deployment(namespace: str, name: str, *, ready: int = 1, desired: int = 1) -> dict:
    return {
        "metadata": {"namespace": namespace, "name": name},
        "spec": {"replicas": desired, "selector": {"matchLabels": {"app": name}}},
        "status": {"readyReplicas": ready},
    }


def _pod(
    namespace: str,
    name: str,
    *,
    waiting: str | None = None,
    restarts: int = 0,
    created: str = "2026-07-19T00:00:00Z",
    gates: list | None = None,
    conditions: list | None = None,
) -> dict:
    state = {"waiting": {"reason": waiting}} if waiting else {"running": {}}
    return {
        "metadata": {"namespace": namespace, "name": name, "creationTimestamp": created},
        "spec": {"schedulingGates": gates or []},
        "status": {
            "conditions": conditions or [],
            "containerStatuses": [{"name": "main", "restartCount": restarts, "state": state}],
        },
    }


def _workload(name: str, queue: str, *, conditions: list | None = None, created: str = "2026-07-19T00:00:00Z") -> dict:
    return {
        "metadata": {"namespace": "iris", "name": name, "creationTimestamp": created},
        "spec": {"queueName": queue},
        "status": {"conditions": conditions or []},
    }


def _namespace(name: str) -> dict:
    return {"metadata": {"name": name}}


def _api(routes: dict):
    """A MockTransport handler serving canned bodies by path.

    A list value becomes a one-page LIST response; a callable runs per request;
    anything else is returned as the JSON body. Unknown paths 404. Requests are
    recorded on ``handler.calls``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        handler.calls.append(request)
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={})
        if callable(body):
            return body(request)
        if isinstance(body, list):
            return httpx.Response(200, json={"items": body, "metadata": {}})
        return httpx.Response(200, json=body)

    handler.calls = []
    return handler


def _source(handler, name: str = "cw-a", token: str | None = "secret") -> K8sSource:
    source = K8sSource(K8sClusterTarget(name, "https://api.example"), token=token, timeout=5.0)
    source._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.example", headers=source._client.headers
    )
    return source


def _healthy_routes() -> dict:
    """A cluster where every watched component is up and the webhook has one endpoint."""
    return {
        "/version": {"gitVersion": "v1.32.0"},
        KUEUE_DEPLOY: _deployment("kueue-system", "kueue-controller-manager"),
        IRIS_DEPLOY: _deployment("iris", "iris-controller"),
        TRAEFIK_DEPLOY: _deployment("traefik", "traefik"),
        CERT_DEPLOY: _deployment("cert-manager", "cert-manager"),
        "/api/v1/namespaces/kueue-system/pods": [_pod("kueue-system", "kueue-controller-manager-abc")],
        "/api/v1/namespaces/iris/pods": [_pod("iris", "iris-controller-abc")],
        "/api/v1/namespaces/traefik/pods": [_pod("traefik", "traefik-abc")],
        "/api/v1/namespaces/cert-manager/pods": [_pod("cert-manager", "cert-manager-abc")],
        KUEUE_SLICES: [{"endpoints": [{"conditions": {"ready": True}}]}],
        "/api/v1/namespaces": [],
        "/apis/kueue.x-k8s.io/v1beta2/workloads": [],
        "/api/v1/events": [],
    }


# --- K8sSource --------------------------------------------------------------


def test_control_plane_flattens_components_and_webhooks():
    routes = _healthy_routes()
    routes[KUEUE_DEPLOY] = _deployment("kueue-system", "kueue-controller-manager", ready=0)
    routes["/api/v1/namespaces/kueue-system/pods"] = [
        _pod("kueue-system", "kueue-controller-manager-abc", waiting="CrashLoopBackOff", restarts=7)
    ]
    routes[KUEUE_SLICES] = [
        # nil ready counts as ready per the EndpointSlice contract; False does not.
        {"endpoints": [{"conditions": {"ready": True}}, {"conditions": {}}]},
        {"endpoints": [{"conditions": {"ready": False}}]},
    ]
    rows = _source(_api(routes)).control_plane()

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
    routes = _healthy_routes()
    del routes[IRIS_DEPLOY]
    rows = _source(_api(routes)).control_plane()
    iris = next(row for row in rows if row["component"] == "iris/iris-controller")
    assert iris["ready"] == 0 and iris["desired"] == 1 and iris["waiting_reason"] == "Missing"


def test_list_follows_continue_pagination():
    pages = [
        {"items": [_pod("iris", "task-1", waiting="CrashLoopBackOff")], "metadata": {"continue": "tok"}},
        {"items": [_pod("iris", "task-2", waiting="ImagePullBackOff")], "metadata": {}},
    ]
    seen_continues = []

    def pods(request: httpx.Request) -> httpx.Response:
        seen_continues.append(request.url.params.get("continue"))
        return httpx.Response(200, json=pages[len(seen_continues) - 1])

    routes = {"/api/v1/namespaces": [_namespace("iris")], "/api/v1/namespaces/iris/pods": pods}
    rows = _source(_api(routes)).crashloops()
    assert [row["pod"] for row in rows] == ["task-1", "task-2"]
    assert seen_continues == [None, "tok"]


def test_429_is_retried_once_after_retry_after():
    responses = [httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, json={"gitVersion": "v1"})]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    source = _source(handler)
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
        _source(failure).probe()
    assert excinfo.value.error_class == expected_class


def test_missing_token_is_an_auth_error_without_a_network_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent without a token")

    with pytest.raises(K8sError) as excinfo:
        _source(handler, token=None).probe()
    assert excinfo.value.error_class == K8sErrorClass.AUTH


def test_crashloop_scope_separates_watched_components_from_workloads():
    routes = {
        "/api/v1/namespaces": [_namespace("iris")],
        "/api/v1/namespaces/iris/pods": [
            _pod("iris", "iris-controller-7f9-x2", waiting="CrashLoopBackOff", restarts=3),
            _pod("iris", "some-user-task-0", waiting="ImagePullBackOff"),
            _pod("iris", "healthy-task-0"),
        ],
    }
    rows = _source(_api(routes)).crashloops()
    assert [(row["pod"], row["scope"], row["reason"]) for row in rows] == [
        ("iris-controller-7f9-x2", "control-plane", "CrashLoopBackOff"),
        ("some-user-task-0", "workload", "ImagePullBackOff"),
    ]


def test_provider_namespaces_are_excluded_from_pod_scans():
    # Only the iris pods route exists: a scan reaching cw-* or kube-* would 404
    # and raise, so a passing scan proves the exclusion.
    routes = {
        "/api/v1/namespaces": [_namespace("cw-exporters"), _namespace("kube-system"), _namespace("iris")],
        "/api/v1/namespaces/iris/pods": [_pod("iris", "task-0", waiting="CrashLoopBackOff")],
    }
    assert [row["pod"] for row in _source(_api(routes)).crashloops()] == ["task-0"]


def test_pending_splits_gated_from_pending_and_sorts_oldest_first():
    unschedulable = [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}]
    gated = [{"type": "PodScheduled", "status": "False", "reason": "SchedulingGated"}]
    routes = {
        "/api/v1/namespaces": [_namespace("iris")],
        "/api/v1/namespaces/iris/pods": [
            _pod("iris", "young-gated", created="2026-07-19T12:00:00Z", conditions=gated),
            _pod("iris", "old-stuck", created="2026-07-01T00:00:00Z", conditions=unschedulable),
        ],
    }
    rows = _source(_api(routes)).pending()
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
    rows = _source(_api(routes)).kueue()
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
    rows = _source(_api(routes)).warning_events()
    assert [row["object"] for row in rows] == ["Deployment/kueue-controller-manager", "Pod/task-0"]
    assert rows[1]["count"] == 4
    assert len(rows[0]["message"]) == 200


# --- K8sFleet ---------------------------------------------------------------


def _fleet(*handlers_by_name: tuple[str, object]) -> K8sFleet:
    return K8sFleet([_source(handler, name=name) for name, handler in handlers_by_name])


def _forbidden(request: httpx.Request) -> httpx.Response:
    return httpx.Response(403, json={})


def test_fleet_stamps_cluster_and_keeps_healthy_clusters_on_partial_failure():
    fleet = _fleet(("cw-a", _api(_healthy_routes())), ("cw-b", _forbidden))
    rows = fleet.control_plane()
    healthy = [row for row in rows if row["cluster"] == "cw-a"]
    assert len(healthy) == 5  # 4 components + 1 webhook
    (error_row,) = [row for row in rows if row["cluster"] == "cw-b"]
    assert error_row["error_class"] == "auth"
    assert "403" in error_row["error"]


def test_alert_routes_return_explicit_zeros_when_healthy():
    fleet = _fleet(("cw-a", _api(_healthy_routes())))
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


def test_crashloop_alert_counts_by_scope():
    routes = _healthy_routes()
    routes["/api/v1/namespaces"] = [_namespace("iris")]
    routes["/api/v1/namespaces/iris/pods"] = [
        _pod("iris", "iris-controller-7f9-x2", waiting="CrashLoopBackOff"),
        _pod("iris", "task-a-0", waiting="CrashLoopBackOff"),
        _pod("iris", "task-b-0", waiting="ImagePullBackOff"),
    ]
    assert _fleet(("cw-a", _api(routes))).alert_crashloops() == [
        {"cluster": "cw-a", "scope": "control-plane", "value": 1},
        {"cluster": "cw-a", "scope": "workload", "value": 2},
    ]


# --- endpoints --------------------------------------------------------------


def _client(fleet: K8sFleet) -> TestClient:
    config = BridgeConfig(
        max_rows=1000,
        cache_ttl=20,
        query_timeout_ms=5000,
        iris_cache_ttl=15,
        github_cache_ttl=60,
        k8s_cache_ttl=30,
        http_timeout=5,
        github_token=None,
        cw_read_token=None,
    )
    return TestClient(create_app(config, {}, {}, GithubSource(token=None, timeout=5.0), fleet))


def test_k8s_routes_serve_fleet_rows():
    client = _client(_fleet(("cw-a", _api(_healthy_routes()))))
    for path in ("/k8s/control_plane", "/k8s/crashloops", "/k8s/pending", "/k8s/kueue", "/k8s/events"):
        assert client.get(path).status_code == 200
    health = client.get("/k8s/health").json()
    assert health[0]["cluster"] == "cw-a" and health[0]["reachable"] is True


def test_alerts_crashloops_scope_param_filters_rows():
    client = _client(_fleet(("cw-a", _api(_healthy_routes()))))
    rows = client.get("/k8s/alerts/crashloops", params={"scope": "control-plane"}).json()
    assert rows == [{"cluster": "cw-a", "scope": "control-plane", "value": 0}]
