# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rigging.provenance import Provenance
from rigging.timing import Timestamp

from iris.backends.protocol import BackendCapability, ProviderError, TaskBackend
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.pagination import (
    _decode_page_token,
    _encode_page_token,
    _page_size,
    _query_fingerprint,
)
from iris.cluster.controller.persistence import reads
from iris.cluster.controller.persistence.database import Tx
from iris.cluster.controller.persistence.json_codec import (
    decode_attribute_value,
)
from iris.cluster.controller.resource_identity import (
    _authority_cluster,
    _execution_cluster,
    _opaque_uid,
)
from iris.cluster.controller.source_status import (
    _available_source,
    _unavailable_backend_source,
)
from iris.cluster.controller.task_state import AttemptRecord, TaskDetailRow
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.types import DEFAULT_BACKEND_ID
from iris.resources.attempt import AttemptSummary
from iris.resources.errors import (
    ActionPolicyRejected,
    ResourceNotFound,
)
from iris.resources.identity import (
    AttemptIdentity,
    NodeIdentity,
    NodeLocator,
    ResourceKey,
    ResourceKind,
    SliceIdentity,
)
from iris.resources.names import (
    JobName,
    WorkerId,
)
from iris.resources.node import (
    NodeAttribute,
    NodeAttributeKind,
    NodeCapacity,
    NodeDetail,
    NodeHealth,
    NodeQuery,
    NodeSummary,
)
from iris.resources.source import (
    MAX_PROVIDER_SNAPSHOT_ITEMS,
    Page,
    ResourceSourceStatus,
)
from iris.resources.state import TaskState

_MAX_NODE_PAGE = 500
_MAX_NODE_RECENT_ATTEMPTS = 50
_NODE_WORKER_SCAN_BATCH = _MAX_NODE_PAGE + 1


@dataclass(frozen=True, slots=True)
class NodeDetails:
    address: str | None
    attributes: tuple[NodeAttribute, ...]


@dataclass(frozen=True, slots=True)
class _ProviderNodeCandidate:
    summary: NodeSummary
    details: NodeDetails


@dataclass(frozen=True, slots=True)
class _WorkerNodeMetadata:
    capacity: NodeCapacity
    slice_id: str | None
    attributes: tuple[NodeAttribute, ...]
    region: str | None


@dataclass(frozen=True, slots=True)
class _ProviderNodeObservation:
    provider_node_id: str
    incarnation: str
    ready: bool
    schedulable: bool
    capacity: NodeCapacity
    running_task_count: int
    region: str | None
    instance_type: str | None


def _provider_node_observations(backend: TaskBackend) -> tuple[_ProviderNodeObservation, ...]:
    status = backend.status()
    if status.kubernetes is None:
        return ()
    return tuple(
        _ProviderNodeObservation(
            provider_node_id=node.name,
            incarnation=node.created,
            ready=node.ready,
            schedulable=node.schedulable,
            capacity=NodeCapacity(
                cpu_millicores=node.cpu_millicores,
                memory_bytes=node.memory_bytes,
                disk_bytes=node.disk_bytes,
                accelerator_kind="gpu" if node.gpu_count else "",
                accelerator_variant=node.gpu_model,
                accelerator_count=node.gpu_count,
            ),
            running_task_count=node.running_pods,
            region=node.region or None,
            instance_type=node.instance_type or None,
        )
        for node in status.kubernetes.nodes
    )


def _worker_node_metadata(
    worker: reads.WorkerRecord,
    attributes: Mapping[str, str | int | float],
) -> _WorkerNodeMetadata:
    accelerator_count = 0
    if worker.device_type == "gpu":
        accelerator_count = worker.total_gpu_count
    elif worker.device_type == "tpu":
        accelerator_count = worker.total_tpu_count
    visible_attributes = dict(attributes)
    if worker.md_provenance_json and worker.md_provenance_json != "{}":
        provenance = Provenance.from_json(worker.md_provenance_json)
        visible_attributes["provenance.tree_hash"] = provenance.tree_hash
        visible_attributes["provenance.base_commit"] = provenance.base_commit
        visible_attributes["provenance.dirty"] = int(provenance.dirty)
        if provenance.branch:
            visible_attributes["provenance.branch"] = provenance.branch
        if provenance.built_by:
            visible_attributes["provenance.built_by"] = provenance.built_by
    region = visible_attributes.get("region")
    return _WorkerNodeMetadata(
        capacity=NodeCapacity(
            cpu_millicores=worker.total_cpu_millicores,
            memory_bytes=worker.total_memory_bytes,
            disk_bytes=worker.md_disk_bytes,
            accelerator_kind=worker.device_type,
            accelerator_variant=worker.device_variant,
            accelerator_count=accelerator_count,
        ),
        slice_id=worker.slice_id or None,
        attributes=tuple(_node_attribute(key, value) for key, value in sorted(visible_attributes.items())),
        region=region if isinstance(region, str) and region else None,
    )


def _node_attribute(key: str, value: str | int | float) -> NodeAttribute:
    if isinstance(value, str):
        return NodeAttribute(key, NodeAttributeKind.STRING, string_value=value)
    if isinstance(value, int):
        return NodeAttribute(key, NodeAttributeKind.INTEGER, integer_value=value)
    return NodeAttribute(key, NodeAttributeKind.FLOAT, float_value=value)


@dataclass(frozen=True, slots=True)
class _WorkerNodeCandidate:
    backend_id: str
    worker: reads.WorkerRecord
    liveness: WorkerLiveness


_NodeCandidate = _ProviderNodeCandidate | _WorkerNodeCandidate


@dataclass(frozen=True, slots=True)
class _ProviderNodeSnapshot:
    candidates: tuple[_ProviderNodeCandidate, ...]
    source_statuses: tuple[ResourceSourceStatus, ...]


class NodeResources:
    """Node resource operations."""

    def __init__(self, dependencies: ResourceDependencies) -> None:
        self._dependencies = dependencies

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]:
        """List one bounded page of canonical Nodes."""
        page, _details = self.list_nodes_with_details(query)
        return page

    def list_nodes_with_details(
        self,
        query: NodeQuery = NodeQuery(),
    ) -> tuple[Page[NodeSummary], Mapping[tuple[str, str], NodeDetails]]:
        """List Nodes and their row-local details without loading attempt history."""
        page_size = _page_size(query.page_size, _MAX_NODE_PAGE)
        fingerprint = _query_fingerprint(
            "nodes",
            {
                "backend_id": query.backend_id,
                "contains": query.contains,
                "health": sorted(value.value for value in query.health),
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        last_key = (
            (
                str(position["backend_id"]),
                str(position["node_id"]),
                str(position["node_uid"]),
            )
            if position is not None
            else None
        )
        provider_snapshot = self._provider_node_snapshot()
        candidates: list[_NodeCandidate] = [
            candidate
            for candidate in provider_snapshot.candidates
            if (query.backend_id is None or candidate.summary.identity.backend_id == query.backend_id)
            and (
                query.contains is None
                or query.contains.casefold() in candidate.summary.identity.key.resource_id.casefold()
            )
            and (not query.health or candidate.summary.health in query.health)
            and (last_key is None or _node_candidate_key(candidate) > last_key)
        ]
        with self._dependencies.db.read_snapshot() as tx:
            candidates.extend(self._worker_node_candidates(tx, query, last_key, page_size + 1))
            candidates.sort(key=_node_candidate_key)
            selected = candidates[:page_size]
            worker_nodes, worker_details = self._materialize_worker_nodes(
                tx,
                [candidate for candidate in selected if isinstance(candidate, _WorkerNodeCandidate)],
            )

        nodes_by_key = {_node_summary_key(node): node for node in worker_nodes}
        details: dict[tuple[str, str], NodeDetails] = dict(worker_details)
        items: list[NodeSummary] = []
        for candidate in selected:
            if isinstance(candidate, _ProviderNodeCandidate):
                node = candidate.summary
                details[(node.identity.backend_id, node.identity.node_uid)] = candidate.details
            else:
                node = nodes_by_key[_node_candidate_key(candidate)]
            items.append(node)

        next_token = None
        if len(candidates) > page_size:
            last = items[-1]
            next_token = _encode_page_token(
                fingerprint,
                {
                    "backend_id": last.identity.backend_id,
                    "node_id": last.identity.key.resource_id,
                    "node_uid": last.identity.node_uid,
                },
            )
        return (
            Page(
                items=tuple(items),
                next_page_token=next_token,
                source_statuses=provider_snapshot.source_statuses,
            ),
            details,
        )

    def describe_node(self, locator: NodeLocator) -> NodeDetail:
        provider_snapshot = self._provider_node_snapshot()
        matches: list[tuple[NodeSummary, NodeDetails]] = [
            (candidate.summary, candidate.details)
            for candidate in provider_snapshot.candidates
            if candidate.summary.identity.key == locator.key
            and candidate.summary.identity.backend_id == locator.backend_id
            and (locator.node_uid is None or candidate.summary.identity.node_uid == locator.node_uid)
        ]
        backend = self._dependencies.backends.get(locator.backend_id)
        if backend is not None and BackendCapability.WORKER_DAEMON in backend.capabilities:
            worker_id = WorkerId(locator.key.resource_id)
            with self._dependencies.db.read_snapshot() as tx:
                worker = reads.get_worker_detail(tx, worker_id)
                if (
                    worker is not None
                    and self._dependencies.runtime.backend_id_for_scale_group(str(worker.scale_group or ""))
                    == locator.backend_id
                    and (locator.node_uid is None or locator.node_uid == worker_id)
                ):
                    nodes, details = self._materialize_worker_nodes(
                        tx,
                        [
                            _WorkerNodeCandidate(
                                locator.backend_id, worker, self._dependencies.runtime.liveness_for_worker(worker_id)
                            )
                        ],
                    )
                    node = nodes[0]
                    matches.append((node, details[(locator.backend_id, node.identity.node_uid)]))
        if not matches:
            raise ResourceNotFound(locator.key.resource_id)
        if len(matches) != 1:
            raise ActionPolicyRejected(f"Node locator {locator.key.resource_id!r} is ambiguous")
        node, details = matches[0]
        return NodeDetail(
            summary=node,
            address=details.address,
            attributes=details.attributes,
            recent_attempts=self._recent_attempts_for_node(node),
            bootstrap_logs=self._bootstrap_logs(node),
            source_statuses=provider_snapshot.source_statuses,
        )

    def _bootstrap_logs(self, node: NodeSummary) -> str | None:
        backend = self._dependencies.backends[node.identity.backend_id]
        if BackendCapability.WORKER_DAEMON not in backend.capabilities or backend.autoscaler is None:
            return None
        return backend.autoscaler.get_init_log(node.identity.key.resource_id, tail=200) or None

    def _recent_attempts_for_node(self, node: NodeSummary) -> tuple[AttemptSummary, ...]:
        backend = self._dependencies.backends[node.identity.backend_id]
        with self._dependencies.db.read_snapshot() as tx:
            if BackendCapability.WORKER_DAEMON in backend.capabilities:
                attempts = reads.recent_attempts_for_worker(
                    tx,
                    WorkerId(node.identity.key.resource_id),
                    limit=_MAX_NODE_RECENT_ATTEMPTS,
                )
            elif BackendCapability.CLUSTER_VIEW in backend.capabilities:
                attempts = reads.recent_attempts_for_provider_node(
                    tx,
                    node.identity.backend_id,
                    node.identity.key.resource_id,
                    limit=_MAX_NODE_RECENT_ATTEMPTS,
                )
            else:
                return ()
            if not attempts:
                return ()
            task_ids = {attempt.task_id for attempt in attempts}
            tasks = reads.bulk_get_task_detail(tx, task_ids)
            jobs = self._job_rows(tx, {task.job_id for task in tasks.values()})
        return tuple(
            self._attempt_summary(tasks[attempt.task_id], attempt, jobs[tasks[attempt.task_id].job_id])
            for attempt in attempts
        )

    def _provider_node_snapshot(self) -> _ProviderNodeSnapshot:
        candidates: list[_ProviderNodeCandidate] = []
        statuses: list[ResourceSourceStatus] = []
        for backend_id, backend in sorted(self._dependencies.backends.items()):
            try:
                observations = _provider_node_observations(backend)
            except (ConnectionError, ProviderError) as exc:
                statuses.append(_unavailable_backend_source(backend_id, exc))
            else:
                if len(observations) > MAX_PROVIDER_SNAPSHOT_ITEMS:
                    statuses.append(
                        _unavailable_backend_source(
                            backend_id,
                            ValueError(f"provider returned more than {MAX_PROVIDER_SNAPSHOT_ITEMS} nodes"),
                        )
                    )
                    continue
                observed_at = Timestamp.now()
                for node in observations:
                    identity = NodeIdentity(
                        ResourceKey(self._dependencies.cluster_id, ResourceKind.NODE, node.provider_node_id),
                        backend_id,
                        _opaque_uid(f"kubernetes:{backend_id}:{node.provider_node_id}:{node.incarnation}"),
                    )
                    summary = NodeSummary(
                        identity=identity,
                        health=NodeHealth.READY if node.ready else NodeHealth.UNAVAILABLE,
                        schedulable=node.schedulable,
                        capacity=node.capacity,
                        scaling_group_id=None,
                        slice=None,
                        running_task_count=node.running_task_count,
                        observed_at=observed_at,
                        region=node.region,
                    )
                    attributes = tuple(
                        attribute
                        for attribute in (
                            _string_node_attribute("instance_type", node.instance_type or ""),
                            _string_node_attribute("region", node.region or ""),
                        )
                        if attribute is not None
                    )
                    candidates.append(_ProviderNodeCandidate(summary, NodeDetails(None, attributes)))
                statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
        return _ProviderNodeSnapshot(tuple(candidates), tuple(statuses))

    def _worker_node_candidates(
        self,
        tx: Tx,
        query: NodeQuery,
        last_key: tuple[str, str, str] | None,
        limit: int,
    ) -> list[_WorkerNodeCandidate]:
        configured_scale_groups = {
            scale_group: backend_id
            for backend_id, config in self._dependencies.backend_configs.items()
            for scale_group in config.scale_groups
        }
        candidates: list[_WorkerNodeCandidate] = []
        for backend_id, backend in sorted(self._dependencies.backends.items()):
            if BackendCapability.WORKER_DAEMON not in backend.capabilities:
                continue
            if query.backend_id is not None and query.backend_id != backend_id:
                continue
            if last_key is not None and backend_id < last_key[0]:
                continue
            after_worker_id = WorkerId(last_key[1]) if last_key is not None and backend_id == last_key[0] else None
            include_after = after_worker_id is not None
            while len(candidates) < limit:
                if backend_id == DEFAULT_BACKEND_ID:
                    rows = reads.worker_detail_page_outside_scale_groups(
                        tx,
                        [
                            scale_group
                            for scale_group, owner in configured_scale_groups.items()
                            if owner != DEFAULT_BACKEND_ID
                        ],
                        after_worker_id=after_worker_id,
                        include_after=include_after,
                        limit=_NODE_WORKER_SCAN_BATCH,
                    )
                else:
                    rows = reads.worker_detail_page_in_scale_groups(
                        tx,
                        [scale_group for scale_group, owner in configured_scale_groups.items() if owner == backend_id],
                        after_worker_id=after_worker_id,
                        include_after=include_after,
                        limit=_NODE_WORKER_SCAN_BATCH,
                    )
                if not rows:
                    break
                for worker in rows:
                    worker_id = WorkerId(worker.worker_id)
                    candidate = _WorkerNodeCandidate(
                        backend_id,
                        worker,
                        self._dependencies.runtime.liveness_for_worker(worker_id),
                    )
                    if last_key is not None and _node_candidate_key(candidate) <= last_key:
                        continue
                    if query.contains is not None and query.contains.casefold() not in worker_id.casefold():
                        continue
                    health = NodeHealth.READY if candidate.liveness.healthy else NodeHealth.DEGRADED
                    if query.health and health not in query.health:
                        continue
                    candidates.append(candidate)
                    if len(candidates) == limit:
                        break
                if len(candidates) == limit or len(rows) < _NODE_WORKER_SCAN_BATCH:
                    break
                after_worker_id = WorkerId(rows[-1].worker_id)
                include_after = False
        return candidates

    def _materialize_worker_nodes(
        self,
        tx: Tx,
        candidates: Sequence[_WorkerNodeCandidate],
    ) -> tuple[tuple[NodeSummary, ...], Mapping[tuple[str, str], NodeDetails]]:
        if not candidates:
            return (), {}
        worker_ids = [WorkerId(candidate.worker.worker_id) for candidate in candidates]
        attributes_by_worker: dict[WorkerId, dict[str, str | int | float]] = {}
        for row in reads.worker_attribute_rows(tx, worker_ids):
            key, value = decode_attribute_value(row)
            attributes_by_worker.setdefault(WorkerId(row.worker_id), {})[key] = value
        running = reads.running_tasks_by_worker(tx, set(worker_ids))
        nodes: list[NodeSummary] = []
        details: dict[tuple[str, str], NodeDetails] = {}
        for candidate in candidates:
            worker = candidate.worker
            worker_id = WorkerId(worker.worker_id)
            stored_attributes = attributes_by_worker.get(worker_id, {})
            metadata = _worker_node_metadata(worker, stored_attributes)
            identity = NodeIdentity(
                ResourceKey(self._dependencies.cluster_id, ResourceKind.NODE, worker_id),
                candidate.backend_id,
                worker_id,
            )
            slice_identity = None
            if metadata.slice_id:
                slice_identity = SliceIdentity(
                    ResourceKey(self._dependencies.cluster_id, ResourceKind.SLICE, metadata.slice_id),
                    candidate.backend_id,
                    _opaque_uid(f"rpc:{candidate.backend_id}:{metadata.slice_id}"),
                )
            nodes.append(
                NodeSummary(
                    identity=identity,
                    health=NodeHealth.READY if candidate.liveness.healthy else NodeHealth.DEGRADED,
                    schedulable=candidate.liveness.healthy,
                    capacity=metadata.capacity,
                    scaling_group_id=str(worker.scale_group or "") or None,
                    slice=slice_identity,
                    running_task_count=len(running.get(worker_id, set())),
                    observed_at=Timestamp.from_ms(candidate.liveness.last_heartbeat_ms),
                    region=metadata.region,
                )
            )
            details[(candidate.backend_id, identity.node_uid)] = NodeDetails(
                str(worker.address or "") or None,
                metadata.attributes,
            )
        return tuple(nodes), details

    def _attempt_summary(
        self,
        task: TaskDetailRow,
        attempt: AttemptRecord,
        job: reads.JobCoordinates,
    ) -> AttemptSummary:
        authority = _authority_cluster(self._dependencies.cluster_id, job)
        task_key = ResourceKey(authority, ResourceKind.TASK, task.task_id.to_wire())
        backend_id = str(attempt.backend_id or task.backend_id or "")
        execution = _execution_cluster(self._dependencies.cluster_id, str(task.cluster)) if backend_id else ""
        node_id = str(attempt.node_name or attempt.worker_id or "")
        node = None
        if node_id and backend_id:
            node = self._current_node_identity(execution, backend_id, node_id)
        return AttemptSummary(
            identity=AttemptIdentity(task_key, int(attempt.attempt_id), str(attempt.attempt_uid)),
            state=TaskState(attempt.state),
            execution_cluster_id=execution,
            backend_id=backend_id,
            node=node,
            created_at=attempt.created_at_ms,
            started_at=attempt.started_at_ms,
            finished_at=attempt.finished_at_ms,
            exit_code=attempt.exit_code,
            error_message=str(attempt.error or ""),
            terminal_reason=str(attempt.terminal_reason or ""),
        )

    def _current_node_identity(self, execution: str, backend_id: str, node_id: str) -> NodeIdentity | None:
        if execution != self._dependencies.cluster_id or not backend_id or not node_id:
            return None
        backend = self._dependencies.backends.get(backend_id)
        if backend is None or BackendCapability.WORKER_DAEMON not in backend.capabilities:
            return None
        return NodeIdentity(ResourceKey(execution, ResourceKind.NODE, node_id), backend_id, node_id)

    def _job_rows(self, tx: Tx, job_ids: set[JobName]) -> dict[JobName, reads.JobCoordinates]:
        return reads.job_coordinates(tx, job_ids)


def _node_summary_key(node: NodeSummary) -> tuple[str, str, str]:
    return (
        node.identity.backend_id,
        node.identity.key.resource_id,
        node.identity.node_uid,
    )


def _node_candidate_key(candidate: _NodeCandidate) -> tuple[str, str, str]:
    if isinstance(candidate, _ProviderNodeCandidate):
        return _node_summary_key(candidate.summary)
    worker_id = str(candidate.worker.worker_id)
    return (candidate.backend_id, worker_id, worker_id)


def _string_node_attribute(key: str, value: str) -> NodeAttribute | None:
    if not value:
        return None
    return NodeAttribute(key=key, kind=NodeAttributeKind.STRING, string_value=value)
