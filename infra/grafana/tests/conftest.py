# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers: the bridge config and a canned k8s API server."""

import httpx
from config import BridgeConfig, K8sClusterTarget
from k8s_source import K8sSource

KUEUE_DEPLOY = "/apis/apps/v1/namespaces/kueue-system/deployments/kueue-controller-manager"
IRIS_DEPLOY = "/apis/apps/v1/namespaces/iris/deployments/iris-controller"
TRAEFIK_DEPLOY = "/apis/apps/v1/namespaces/traefik/deployments/traefik"
CERT_DEPLOY = "/apis/apps/v1/namespaces/cert-manager/deployments/cert-manager"
FINELOG_DEPLOYMENTS_PATH = "/apis/apps/v1/deployments"
KUEUE_SLICES = "/apis/discovery.k8s.io/v1/namespaces/kueue-system/endpointslices"


def bridge_config(cache_ttl: float = 20.0) -> BridgeConfig:
    return BridgeConfig(
        max_rows=1000,
        cache_ttl=cache_ttl,
        query_timeout_ms=5000,
        iris_cache_ttl=15.0,
        github_cache_ttl=60.0,
        k8s_cache_ttl=30.0,
        http_timeout=5.0,
        github_token=None,
        cw_read_token=None,
    )


def deployment(
    namespace: str,
    name: str,
    *,
    ready: int = 1,
    desired: int = 1,
    containers: tuple[str, ...] = (),
) -> dict:
    return {
        "metadata": {"namespace": namespace, "name": name},
        "spec": {
            "replicas": desired,
            "selector": {"matchLabels": {"app": name}},
            "template": {"spec": {"containers": [{"name": container} for container in containers]}},
        },
        "status": {"readyReplicas": ready},
    }


def pod(
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


def node(
    name: str,
    *,
    rack: str | None = None,
    rack_name: str = "",
    instance_type: str = "",
    gpu_capacity: int = 0,
    ready: bool = True,
) -> dict:
    labels = {}
    if rack is not None:
        labels["node.coreweave.cloud/rack"] = rack
        labels["ds.coreweave.com/physical-topology.rack-name"] = rack_name
    if instance_type:
        labels["node.kubernetes.io/instance-type"] = instance_type
    return {
        "metadata": {"name": name, "labels": labels},
        "status": {
            "capacity": {"nvidia.com/gpu": str(gpu_capacity)},
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


def k8s_api(routes: dict):
    """A MockTransport handler serving canned bodies by path.

    A list value becomes a one-page LIST response; a callable runs per request;
    anything else is returned as the JSON body. Unknown paths 404.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={})
        if callable(body):
            return body(request)
        if isinstance(body, list):
            return httpx.Response(200, json={"items": body, "metadata": {}})
        return httpx.Response(200, json=body)

    return handler


def make_k8s_source(handler, name: str = "cw-a", token: str | None = "secret") -> K8sSource:
    source = K8sSource(K8sClusterTarget(name, "https://api.example"), token=token, timeout=5.0)
    source._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.example", headers=source._client.headers
    )
    return source


def healthy_k8s_routes() -> dict:
    """A cluster where every watched component is up, the webhook has one endpoint, and one GPU rack is full."""
    return {
        "/version": {"gitVersion": "v1.32.0"},
        KUEUE_DEPLOY: deployment("kueue-system", "kueue-controller-manager"),
        IRIS_DEPLOY: deployment("iris", "iris-controller"),
        TRAEFIK_DEPLOY: deployment("traefik", "traefik"),
        CERT_DEPLOY: deployment("cert-manager", "cert-manager"),
        FINELOG_DEPLOYMENTS_PATH: [deployment("iris", "finelog-cw-a", containers=("finelog",))],
        "/api/v1/namespaces/kueue-system/pods": [pod("kueue-system", "kueue-controller-manager-abc")],
        "/api/v1/namespaces/iris/pods": [pod("iris", "iris-controller-abc")],
        "/api/v1/namespaces/traefik/pods": [pod("traefik", "traefik-abc")],
        "/api/v1/namespaces/cert-manager/pods": [pod("cert-manager", "cert-manager-abc")],
        KUEUE_SLICES: [{"endpoints": [{"conditions": {"ready": True}}]}],
        "/api/v1/namespaces": [],
        "/apis/kueue.x-k8s.io/v1beta2/workloads": [],
        "/api/v1/events": [],
        "/api/v1/nodes": [
            node("g1", rack="169", rack_name="dh1-r169-us-east-08a", instance_type="gb200-4x", gpu_capacity=4)
        ],
    }
