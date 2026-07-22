# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only CoreWeave k8s control-plane state, as flat JSON rows.

One K8sSource per cluster speaks to the public CKS API server with plain httpx
GETs and the CW read-role bearer token — no kubernetes client. K8sFleet fans a
query out across every cluster and stamps a ``cluster`` column, so one response
covers the fleet.

The pod-level scans (crashloops, pending, termination candidates) cover every namespace except the
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
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar

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

# A pod is a stuck-termination candidate only after its API deletion deadline has
# been expired for this long. Grafana holds the alert for another five minutes.
STUCK_TERMINATION_OVERDUE_SECONDS = 120

_GPU_RESOURCE = "nvidia.com/gpu"
_IRIS_TASK_ID_ENV = "IRIS_TASK_ID"
_TERMINAL_POD_PHASES = frozenset(("Succeeded", "Failed"))

# CoreWeave physical-topology labels a GPU node carries: which rack it lives in,
# the rack's full CoreWeave-assigned name, and its instance type.
_RACK_LABEL = "node.coreweave.cloud/rack"
_RACK_NAME_LABEL = "ds.coreweave.com/physical-topology.rack-name"
_INSTANCE_TYPE_LABEL = "node.kubernetes.io/instance-type"

# gpu_racks' tray/rack concept — many nodes sharing one liquid-cooled rack, with a
# fleet-wide expected tray count — is specific to GB200 NVL72. Other instance types
# (e.g. gd-8xh100ib-i128, a standalone 8-GPU H100 server) get their own CoreWeave
# rack label too, but mostly 1 node per rack: cw-us-east-02a has 29 racks, 26 of them
# with exactly one node. Grouping those in made every one read as "1 of 18 trays" and
# fired the below-16 alert on hardware the threshold was never about.
_GB200_INSTANCE_TYPE_SUBSTRING = "gb200"

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


class TerminationClass(StrEnum):
    """Operator action implied by an overdue terminating pod."""

    INVALID_TIMESTAMP = "invalid_timestamp"
    FINALIZER = "finalizer"
    TERMINAL = "terminal"
    UNBOUND = "unbound"
    NODE_CLEANUP = "node_cleanup"


@dataclass(frozen=True)
class TerminatingPod:
    """A pod termination record that requires operator visibility."""

    cluster: str
    namespace: str
    pod: str
    node: str
    phase: str
    deletion_timestamp: str
    deletion_grace_seconds: int | None
    overdue_seconds: int | None
    gpu_count: int
    task_attempt: str
    task_label: str
    job_label: str
    priority_class: str
    finalizers: str
    classification: TerminationClass


@dataclass(frozen=True)
class TerminatingPodError:
    """A cluster query failure returned beside healthy clusters' pod records."""

    cluster: str
    error_class: str
    error: str


TerminatingPodResult = TerminatingPod | TerminatingPodError
_Row = TypeVar("_Row")


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


def _deletion_overdue_seconds(timestamp: str) -> int | None:
    """Return age past a deletion deadline, or None for an invalid deadline."""
    try:
        return _age_seconds(timestamp)
    except (TypeError, ValueError):
        return None


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

    def _scan_pods(self, field_selector: str | None) -> list[dict]:
        pods: list[dict] = []
        params = {"fieldSelector": field_selector} if field_selector else None
        for namespace in self._scanned_namespaces():
            pods.extend(self._list(f"/api/v1/namespaces/{namespace}/pods", params))
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

    def termination_candidates(self) -> list[TerminatingPod]:
        """Return overdue terminating pods and invalid deletion timestamps.

        Raises:
            ValueError: A candidate has a malformed GPU resource quantity.
        """
        rows = []
        for pod in self._scan_pods(None):
            metadata = pod.get("metadata") or {}
            deletion_timestamp = metadata.get("deletionTimestamp")
            if not deletion_timestamp:
                continue
            overdue_seconds = _deletion_overdue_seconds(deletion_timestamp)
            if overdue_seconds is not None and overdue_seconds < STUCK_TERMINATION_OVERDUE_SECONDS:
                continue
            spec = pod.get("spec") or {}
            status = pod.get("status") or {}
            labels = metadata.get("labels") or {}
            finalizers = sorted(metadata.get("finalizers") or [])
            rows.append(
                TerminatingPod(
                    cluster=self._target.name,
                    namespace=metadata.get("namespace") or "",
                    pod=metadata.get("name") or "",
                    node=spec.get("nodeName") or "",
                    phase=status.get("phase") or "",
                    deletion_timestamp=deletion_timestamp,
                    deletion_grace_seconds=metadata.get("deletionGracePeriodSeconds"),
                    overdue_seconds=overdue_seconds,
                    gpu_count=_pod_gpu_count(pod),
                    task_attempt=_iris_task_attempt(pod),
                    task_label=labels.get("iris.task_id") or "",
                    job_label=labels.get("iris.job_id") or "",
                    priority_class=spec.get("priorityClassName") or "",
                    finalizers=",".join(finalizers),
                    classification=_termination_class(pod, overdue_seconds),
                )
            )
        rows.sort(key=lambda row: row.overdue_seconds if row.overdue_seconds is not None else -1, reverse=True)
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

    def gpu_racks(self) -> list[dict]:
        """One row per physical rack of GB200 nodes: trays registered vs. Ready.

        Scoped to GB200 NVL72 instance types (see _GB200_INSTANCE_TYPE_SUBSTRING):
        other GPU instance types carry a CoreWeave rack label too, but with a
        different physical-rack topology the 16/18-tray expectation doesn't apply to.

        Raises:
            ValueError: A node has a malformed nvidia.com/gpu capacity quantity.
        """
        racks: dict[str, dict] = {}
        for node in self._list("/api/v1/nodes"):
            if _node_gpu_capacity(node) <= 0:
                continue
            labels = (node.get("metadata") or {}).get("labels") or {}
            instance_type = labels.get(_INSTANCE_TYPE_LABEL, "")
            if _GB200_INSTANCE_TYPE_SUBSTRING not in instance_type:
                continue
            rack = labels.get(_RACK_LABEL)
            if rack is None:
                continue
            bucket = racks.setdefault(
                rack,
                {
                    "rack": rack,
                    "rack_name": labels.get(_RACK_NAME_LABEL, ""),
                    "instance_type": instance_type,
                    "trays_total": 0,
                    "trays_ready": 0,
                },
            )
            bucket["trays_total"] += 1
            if _node_ready(node):
                bucket["trays_ready"] += 1
        return [racks[rack] for rack in sorted(racks, key=int)]


def _node_gpu_capacity(node: dict) -> int:
    raw = ((node.get("status") or {}).get("capacity") or {}).get(_GPU_RESOURCE, 0)
    try:
        return int(raw)
    except (TypeError, ValueError) as err:
        raise ValueError(f"invalid {_GPU_RESOURCE} capacity quantity: {raw!r}") from err


def _node_ready(node: dict) -> bool:
    conditions = (node.get("status") or {}).get("conditions") or []
    return any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)


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


def _resource_gpu_count(resources: dict | None) -> int:
    resources = resources or {}
    values = []
    for field in ("requests", "limits"):
        raw = (resources.get(field) or {}).get(_GPU_RESOURCE, 0)
        try:
            values.append(int(raw))
        except (TypeError, ValueError) as err:
            raise ValueError(f"invalid {_GPU_RESOURCE} {field} quantity: {raw!r}") from err
    return max(values, default=0)


def _pod_gpu_count(pod: dict) -> int:
    """Return the effective GPU request Kubernetes uses to schedule the pod."""
    spec = pod.get("spec") or {}
    app_total = sum(_resource_gpu_count(c.get("resources")) for c in spec.get("containers") or [])
    restartable_total = 0
    init_peak = 0
    for container in spec.get("initContainers") or []:
        count = _resource_gpu_count(container.get("resources"))
        if container.get("restartPolicy") == "Always":
            restartable_total += count
        else:
            init_peak = max(init_peak, restartable_total + count)
    return max(app_total + restartable_total, init_peak)


def _iris_task_attempt(pod: dict) -> str:
    for container in (pod.get("spec") or {}).get("containers") or []:
        if container.get("name") != "task":
            continue
        for item in container.get("env") or []:
            if item.get("name") == _IRIS_TASK_ID_ENV:
                return item.get("value") or ""
    return ""


def _termination_class(pod: dict, overdue_seconds: int | None) -> TerminationClass:
    metadata = pod.get("metadata") or {}
    if overdue_seconds is None:
        return TerminationClass.INVALID_TIMESTAMP
    if metadata.get("finalizers"):
        return TerminationClass.FINALIZER
    status = pod.get("status") or {}
    if status.get("phase") in _TERMINAL_POD_PHASES:
        return TerminationClass.TERMINAL
    if not (pod.get("spec") or {}).get("nodeName"):
        return TerminationClass.UNBOUND
    return TerminationClass.NODE_CLEANUP


class K8sFleet:
    """Fans one query out across every cluster, one thread each, stamping ``cluster``."""

    def __init__(self, sources: Sequence[K8sSource]) -> None:
        self._sources = tuple(sources)
        self._executor = ThreadPoolExecutor(max_workers=max(len(self._sources), 1), thread_name_prefix="k8s")

    def _collect(
        self,
        fn: Callable[[K8sSource], list[_Row]],
        on_error: Callable[[K8sSource, K8sError], list[_Row]],
    ) -> list[_Row]:
        futures = [(source, self._executor.submit(fn, source)) for source in self._sources]
        rows: list[_Row] = []
        for source, future in futures:
            try:
                rows.extend(future.result())
            except K8sError as err:
                logger.warning("k8s query failed for %s: %s", source.target.name, err)
                rows.extend(on_error(source, err))
        return rows

    def _fan_out(
        self,
        fn: Callable[[K8sSource], list[dict]],
        on_error: Callable[[K8sError], list[dict]],
    ) -> list[dict]:
        def stamped(source: K8sSource) -> list[dict]:
            return [{"cluster": source.target.name, **row} for row in fn(source)]

        def stamped_error(source: K8sSource, err: K8sError) -> list[dict]:
            return [{"cluster": source.target.name, **row} for row in on_error(err)]

        return self._collect(stamped, stamped_error)

    @staticmethod
    def _error_row(err: K8sError) -> list[dict]:
        return [{"error_class": str(err.error_class), "error": str(err)}]

    def control_plane(self) -> list[dict]:
        return self._fan_out(lambda s: s.control_plane(), self._error_row)

    def crashloops(self) -> list[dict]:
        return self._fan_out(lambda s: s.crashloops(), self._error_row)

    def pending(self) -> list[dict]:
        return self._fan_out(lambda s: s.pending(), self._error_row)

    def termination_candidates(self) -> list[TerminatingPodResult]:
        return self._collect(
            lambda source: source.termination_candidates(),
            lambda source, err: [TerminatingPodError(source.target.name, str(err.error_class), str(err))],
        )

    def kueue(self) -> list[dict]:
        return self._fan_out(lambda s: s.kueue(), self._error_row)

    def warning_events(self) -> list[dict]:
        return self._fan_out(lambda s: s.warning_events(), self._error_row)

    def gpu_racks(self) -> list[dict]:
        return self._fan_out(lambda s: s.gpu_racks(), self._error_row)

    def alert_gpu_rack_trays(self) -> list[dict]:
        """Per rack: trays_ready as ``value``, for the below-minimum-trays alert.

        Unlike the other alert routes, an unreachable cluster contributes no rows
        here rather than an explicit safe value: we don't know its rack set to fill
        in placeholders for, and a fabricated value below the threshold would
        double-page alongside K8sClusterUnreachable, which already covers that
        failure mode. noDataState=Alerting still pages if the whole fleet drops out.
        """

        def counts(source: K8sSource) -> list[dict]:
            return [
                {"rack": row["rack"], "rack_name": row["rack_name"], "value": row["trays_ready"]}
                for row in source.gpu_racks()
            ]

        return self._fan_out(counts, lambda err: [])

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

    def alert_stuck_gpu_pods(self, candidates: Sequence[TerminatingPodResult] | None = None) -> list[dict]:
        """Return node-grouped counts plus zero rows where no qualifying evidence exists."""
        rows = list(candidates) if candidates is not None else self.termination_candidates()
        by_node: dict[tuple[str, str], int] = {}
        for row in rows:
            if not isinstance(row, TerminatingPod):
                continue
            if row.classification != TerminationClass.NODE_CLEANUP or row.gpu_count <= 0:
                continue
            key = (row.cluster, row.node)
            by_node[key] = by_node.get(key, 0) + 1

        alert_rows = []
        affected_clusters = set()
        for (cluster, node), count in sorted(by_node.items()):
            affected_clusters.add(cluster)
            alert_rows.append(
                {
                    "cluster": cluster,
                    "node": node,
                    "value": count,
                }
            )
        for source in self._sources:
            if source.target.name not in affected_clusters:
                alert_rows.append(
                    {
                        "cluster": source.target.name,
                        "node": "",
                        "value": 0,
                    }
                )
        return alert_rows
