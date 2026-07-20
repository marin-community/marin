# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only CoreWeave k8s control-plane state, as flat JSON rows.

One K8sSource per cluster speaks to the public CKS API server with plain httpx
GETs and the CW read-role bearer token — no kubernetes client. K8sFleet fans a
query out across every cluster and stamps a ``cluster`` column, so one response
covers the fleet.

The pod-level scans (crashloops, pending) cover every namespace except the
provider-managed prefixes in PROVIDER_NAMESPACE_PREFIXES: CoreWeave's per-node
daemons are thousands of pods of someone else's infrastructure, while the
namespaces we operate hold about a hundred.

Failure semantics: a cluster that cannot be queried becomes labeled rows inside
the aggregate response — never an empty result — so healthy clusters keep
rendering and alert rules keep one row per cluster. Errors are classified
(auth / network / timeout / http) so a revoked token reads differently from a
dead API server. On the alert routes every numeric falls back to zero for an
unreachable cluster: for the crashloop and degraded rules zero means "no
evidence, no page" (the unreachable rule pages instead), while for
webhook_ready zero means "no ready endpoints", so unreachability also fires the
webhook rule — deliberate, because unknown admission state is exactly the
failure class it watches.
"""

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum

import httpx
from config import (
    PROVIDER_NAMESPACE_PREFIXES,
    WATCHED_COMPONENTS,
    WATCHED_WEBHOOKS,
    K8sClusterTarget,
    WatchedComponent,
    WatchedWebhook,
)

logger = logging.getLogger(__name__)

_LIST_PAGE = 500
_MAX_LIST_PAGES = 10
_MAX_RETRY_AFTER = 3.0
# Warning events returned per cluster, most recent first.
_EVENT_LIMIT = 100
_EVENT_MESSAGE_LIMIT = 200

# Container waiting reasons the crashloop rows report.
BACKOFF_REASONS = ("CrashLoopBackOff", "ImagePullBackOff")

# Crashloop scope labels: pods of a watched component vs everything else. The alert
# rules filter on these values (the crashloops rule pages only on SCOPE_CONTROL_PLANE).
SCOPE_CONTROL_PLANE = "control-plane"
SCOPE_WORKLOAD = "workload"


class K8sErrorClass(StrEnum):
    """Why a cluster query failed, coarse enough to be an alert label."""

    AUTH = "auth"  # 401/403: token missing, revoked, or under-privileged
    NETWORK = "network"  # connect/TLS failure
    TIMEOUT = "timeout"
    HTTP = "http"  # any other non-200


class K8sError(Exception):
    """A per-cluster query failure, carrying its class for error rows."""

    def __init__(self, error_class: K8sErrorClass, message: str) -> None:
        self.error_class = error_class
        super().__init__(message)


def _age_seconds(timestamp: str | None) -> int | None:
    if not timestamp:
        return None
    created = datetime.fromisoformat(timestamp)
    return max(int((datetime.now(UTC) - created).total_seconds()), 0)


def _epoch_ms(timestamp: str | None) -> int | None:
    if not timestamp:
        return None
    return round(datetime.fromisoformat(timestamp).timestamp() * 1000)


def _retry_after(response: httpx.Response) -> float:
    try:
        return min(float(response.headers.get("retry-after", "1")), _MAX_RETRY_AFTER)
    except ValueError:
        return 1.0


class K8sSource:
    """A query handle for one cluster's k8s API server.

    Every method raises K8sError on failure; K8sFleet turns that into labeled
    rows. Rows carry no cluster column — the fleet stamps it.
    """

    def __init__(self, target: K8sClusterTarget, *, token: str | None, timeout: float) -> None:
        self._target = target
        self._token = token
        headers = {"authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(base_url=target.api_server, timeout=timeout, headers=headers)

    @property
    def target(self) -> K8sClusterTarget:
        return self._target

    def _get(self, path: str, params: dict | None = None, *, none_on_404: bool = False) -> dict | None:
        """GET one API path as JSON, retrying a 429 once after its Retry-After."""
        if self._token is None:
            raise K8sError(K8sErrorClass.AUTH, "CW_READ_TOKEN is not set")
        for attempt in (1, 2):
            try:
                response = self._client.get(path, params=params)
            except httpx.TimeoutException as err:
                raise K8sError(K8sErrorClass.TIMEOUT, f"{path}: {err}") from err
            except httpx.TransportError as err:
                raise K8sError(K8sErrorClass.NETWORK, f"{path}: {err}") from err
            if response.status_code == 429 and attempt == 1:
                time.sleep(_retry_after(response))
                continue
            if response.status_code == 404 and none_on_404:
                return None
            if response.status_code in (401, 403):
                raise K8sError(K8sErrorClass.AUTH, f"{path}: HTTP {response.status_code}")
            if response.status_code != 200:
                raise K8sError(K8sErrorClass.HTTP, f"{path}: HTTP {response.status_code}")
            return response.json()
        raise AssertionError("unreachable")

    def _list(self, path: str, params: dict | None = None) -> list[dict]:
        """LIST with limit/continue pagination, concatenating the pages' items."""
        params = dict(params or {})
        params["limit"] = _LIST_PAGE
        items: list[dict] = []
        for _ in range(_MAX_LIST_PAGES):
            page = self._get(path, params)
            assert page is not None
            items.extend(page.get("items", []))
            cont = (page.get("metadata") or {}).get("continue")
            if not cont:
                return items
            params["continue"] = cont
        logger.warning("%s: %s pagination stopped after %d pages", self._target.name, path, _MAX_LIST_PAGES)
        return items

    def probe(self) -> int:
        """Round-trip /version (cheap, still authenticated) and return the latency in ms."""
        started = time.monotonic()
        self._get("/version")
        return round((time.monotonic() - started) * 1000)

    def _deployment(self, component: WatchedComponent) -> dict | None:
        path = f"/apis/apps/v1/namespaces/{component.namespace}/deployments/{component.deployment}"
        return self._get(path, none_on_404=True)

    def _deployment_pods(self, component: WatchedComponent, deployment: dict) -> list[dict]:
        match_labels = ((deployment.get("spec") or {}).get("selector") or {}).get("matchLabels") or {}
        if not match_labels:
            return []
        selector = ",".join(f"{key}={value}" for key, value in sorted(match_labels.items()))
        return self._list(f"/api/v1/namespaces/{component.namespace}/pods", {"labelSelector": selector})

    def component_status(self, component: WatchedComponent) -> dict:
        """Ready/desired replicas plus restart and waiting state from the pods behind one Deployment.

        A missing Deployment reports desired=1, ready=0 with waiting_reason=Missing, so
        a deleted control-plane component reads as degraded rather than healthy.
        """
        deployment = self._deployment(component)
        if deployment is None:
            return {
                "kind": "component",
                "component": component.key,
                "ready": 0,
                "desired": 1,
                "restarts": 0,
                "waiting_reason": "Missing",
            }
        desired = (deployment.get("spec") or {}).get("replicas")
        desired = 1 if desired is None else desired
        ready = (deployment.get("status") or {}).get("readyReplicas") or 0
        restarts = 0
        waiting_reason = ""
        for pod in self._deployment_pods(component, deployment):
            for status in _container_statuses(pod):
                restarts += status.get("restartCount") or 0
                reason = ((status.get("state") or {}).get("waiting") or {}).get("reason")
                if reason and not waiting_reason:
                    waiting_reason = reason
        return {
            "kind": "component",
            "component": component.key,
            "ready": ready,
            "desired": desired,
            "restarts": restarts,
            "waiting_reason": waiting_reason,
        }

    def webhook_ready_endpoints(self, webhook: WatchedWebhook) -> int:
        """Count ready endpoints across the service's EndpointSlices (nil ready counts as ready)."""
        slices = self._list(
            f"/apis/discovery.k8s.io/v1/namespaces/{webhook.namespace}/endpointslices",
            {"labelSelector": f"kubernetes.io/service-name={webhook.service}"},
        )
        ready = 0
        for slice_ in slices:
            for endpoint in slice_.get("endpoints") or []:
                if (endpoint.get("conditions") or {}).get("ready") is not False:
                    ready += 1
        return ready

    def control_plane(self) -> list[dict]:
        """One row per watched component, then one per watched webhook service."""
        rows = [self.component_status(component) for component in WATCHED_COMPONENTS]
        for webhook in WATCHED_WEBHOOKS:
            rows.append(
                {
                    "kind": "webhook",
                    "component": webhook.key,
                    "ready_endpoints": self.webhook_ready_endpoints(webhook),
                }
            )
        return rows

    def _scanned_namespaces(self) -> list[str]:
        """Namespaces the pod scans cover: everything not provider-managed by prefix."""
        namespaces = self._list("/api/v1/namespaces")
        names = [(ns.get("metadata") or {}).get("name") or "" for ns in namespaces]
        return [name for name in names if name and not name.startswith(PROVIDER_NAMESPACE_PREFIXES)]

    def _scan_pods(self, field_selector: str) -> list[dict]:
        pods: list[dict] = []
        for namespace in self._scanned_namespaces():
            pods.extend(self._list(f"/api/v1/namespaces/{namespace}/pods", {"fieldSelector": field_selector}))
        return pods

    def crashloops(self) -> list[dict]:
        """One row per container sitting in a backoff waiting state, across the scanned namespaces."""
        pods = self._scan_pods("status.phase!=Succeeded,status.phase!=Failed")
        rows = []
        for pod in pods:
            for status in _container_statuses(pod):
                reason = ((status.get("state") or {}).get("waiting") or {}).get("reason")
                if reason not in BACKOFF_REASONS:
                    continue
                metadata = pod.get("metadata") or {}
                rows.append(
                    {
                        "namespace": metadata.get("namespace"),
                        "pod": metadata.get("name"),
                        "container": status.get("name"),
                        "reason": reason,
                        "restarts": status.get("restartCount") or 0,
                        "scope": _pod_scope(metadata),
                    }
                )
        return rows

    def pending(self) -> list[dict]:
        """One row per Pending pod in the scanned namespaces, split into scheduling_gated vs pending, oldest first."""
        pods = self._scan_pods("status.phase=Pending")
        rows = []
        for pod in pods:
            metadata = pod.get("metadata") or {}
            scheduled = _pod_condition(pod, "PodScheduled")
            gated = bool((pod.get("spec") or {}).get("schedulingGates")) or scheduled.get("reason") == "SchedulingGated"
            rows.append(
                {
                    "namespace": metadata.get("namespace"),
                    "pod": metadata.get("name"),
                    "state": "scheduling_gated" if gated else "pending",
                    "reason": scheduled.get("reason") or "",
                    "age_seconds": _age_seconds(metadata.get("creationTimestamp")),
                }
            )
        rows.sort(key=lambda row: row["age_seconds"] or 0, reverse=True)
        return rows

    def kueue(self) -> list[dict]:
        """Unadmitted, unfinished Kueue Workloads aggregated per local queue."""
        workloads = self._list("/apis/kueue.x-k8s.io/v1beta2/workloads")
        queues: dict[str, dict] = {}
        for workload in workloads:
            conditions = {c.get("type"): c.get("status") for c in (workload.get("status") or {}).get("conditions") or []}
            if conditions.get("Admitted") == "True" or conditions.get("Finished") == "True":
                continue
            queue = (workload.get("spec") or {}).get("queueName") or "unknown"
            age = _age_seconds((workload.get("metadata") or {}).get("creationTimestamp")) or 0
            bucket = queues.setdefault(queue, {"queue": queue, "unadmitted": 0, "oldest_age_seconds": 0})
            bucket["unadmitted"] += 1
            bucket["oldest_age_seconds"] = max(bucket["oldest_age_seconds"], age)
        return [queues[queue] for queue in sorted(queues)]

    def warning_events(self) -> list[dict]:
        """The most recent Warning events, newest first."""
        events = self._list("/api/v1/events", {"fieldSelector": "type=Warning"})
        rows = []
        for event in events:
            involved = event.get("involvedObject") or {}
            last_seen = (
                event.get("lastTimestamp")
                or event.get("eventTime")
                or (event.get("metadata") or {}).get("creationTimestamp")
            )
            rows.append(
                {
                    "namespace": involved.get("namespace"),
                    "object": f"{involved.get('kind', '?')}/{involved.get('name', '?')}",
                    "reason": event.get("reason"),
                    "message": (event.get("message") or "")[:_EVENT_MESSAGE_LIMIT],
                    "count": event.get("count") or 1,
                    "last_seen": _epoch_ms(last_seen),
                }
            )
        rows.sort(key=lambda row: row["last_seen"] or 0, reverse=True)
        return rows[:_EVENT_LIMIT]


def _container_statuses(pod: dict) -> list[dict]:
    status = pod.get("status") or {}
    return list(status.get("initContainerStatuses") or []) + list(status.get("containerStatuses") or [])


def _pod_condition(pod: dict, condition_type: str) -> dict:
    for condition in (pod.get("status") or {}).get("conditions") or []:
        if condition.get("type") == condition_type:
            return condition
    return {}


def _pod_scope(metadata: dict) -> str:
    """SCOPE_CONTROL_PLANE if the pod belongs to a watched Deployment, else SCOPE_WORKLOAD."""
    namespace, name = metadata.get("namespace"), metadata.get("name") or ""
    for component in WATCHED_COMPONENTS:
        if namespace == component.namespace and name.startswith(f"{component.deployment}-"):
            return SCOPE_CONTROL_PLANE
    return SCOPE_WORKLOAD


class K8sFleet:
    """Fans one query out across every cluster, one thread each, stamping ``cluster``."""

    def __init__(self, sources: Sequence[K8sSource]) -> None:
        self._sources = tuple(sources)
        self._executor = ThreadPoolExecutor(max_workers=max(len(self._sources), 1), thread_name_prefix="k8s")

    def _fan_out(
        self,
        fn: Callable[[K8sSource], list[dict]],
        on_error: Callable[[K8sError], list[dict]],
    ) -> list[dict]:
        futures = [(source, self._executor.submit(fn, source)) for source in self._sources]
        rows: list[dict] = []
        for source, future in futures:
            try:
                cluster_rows = future.result()
            except K8sError as err:
                logger.warning("k8s query failed for %s: %s", source.target.name, err)
                cluster_rows = on_error(err)
            rows.extend({"cluster": source.target.name, **row} for row in cluster_rows)
        return rows

    @staticmethod
    def _error_row(err: K8sError) -> list[dict]:
        return [{"error_class": str(err.error_class), "error": str(err)}]

    def control_plane(self) -> list[dict]:
        return self._fan_out(lambda s: s.control_plane(), self._error_row)

    def crashloops(self) -> list[dict]:
        return self._fan_out(lambda s: s.crashloops(), self._error_row)

    def pending(self) -> list[dict]:
        return self._fan_out(lambda s: s.pending(), self._error_row)

    def kueue(self) -> list[dict]:
        return self._fan_out(lambda s: s.kueue(), self._error_row)

    def warning_events(self) -> list[dict]:
        return self._fan_out(lambda s: s.warning_events(), self._error_row)

    def health(self) -> list[dict]:
        """One row per cluster: reachable, error class, and API server latency."""

        def probe(source: K8sSource) -> list[dict]:
            return [{"reachable": True, "error_class": "", "error": "", "latency_ms": source.probe()}]

        def on_error(err: K8sError) -> list[dict]:
            return [{"reachable": False, "error_class": str(err.error_class), "error": str(err), "latency_ms": None}]

        return self._fan_out(probe, on_error)

    def alert_unreachable(self) -> list[dict]:
        """Per cluster: value=1 with its error class when the API server cannot be read."""

        def probe(source: K8sSource) -> list[dict]:
            source.probe()
            return [{"error_class": "none", "value": 0}]

        return self._fan_out(probe, lambda err: [{"error_class": str(err.error_class), "value": 1}])

    def alert_crashloops(self) -> list[dict]:
        """Per cluster: backoff container counts for both scopes, zero rows included."""

        def counts(source: K8sSource) -> list[dict]:
            by_scope = {SCOPE_CONTROL_PLANE: 0, SCOPE_WORKLOAD: 0}
            for row in source.crashloops():
                by_scope[row["scope"]] += 1
            return [{"scope": scope, "value": count} for scope, count in by_scope.items()]

        return self._fan_out(
            counts, lambda err: [{"scope": s, "value": 0} for s in (SCOPE_CONTROL_PLANE, SCOPE_WORKLOAD)]
        )

    def alert_webhook_ready(self) -> list[dict]:
        """Per cluster and watched webhook: the ready-endpoint count (0 when unreachable)."""

        def counts(source: K8sSource) -> list[dict]:
            return [{"webhook": w.key, "value": source.webhook_ready_endpoints(w)} for w in WATCHED_WEBHOOKS]

        return self._fan_out(counts, lambda err: [{"webhook": w.key, "value": 0} for w in WATCHED_WEBHOOKS])

    def alert_degraded(self) -> list[dict]:
        """Per cluster and watched component: desired minus ready replicas."""

        def gaps(source: K8sSource) -> list[dict]:
            rows = []
            for component in WATCHED_COMPONENTS:
                status = source.component_status(component)
                rows.append({"component": component.key, "value": max(status["desired"] - status["ready"], 0)})
            return rows

        return self._fan_out(gaps, lambda err: [{"component": c.key, "value": 0} for c in WATCHED_COMPONENTS])
