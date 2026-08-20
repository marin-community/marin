# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource-level capacity composition over persistence and backend snapshots."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from rigging.timing import Timestamp

from iris.backends.protocol import BackendCapability, DeviceCapacity, ProviderError
from iris.backends.status import (
    BackendStatus,
    KubernetesStatus,
    RoutingStatus,
    ScaleGroupStatus,
    WorkerFleetStatus,
)
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.persistence import reads
from iris.cluster.controller.slice import slice_detail_from_status
from iris.cluster.controller.source_status import (
    _available_source,
    _unavailable_backend_source,
    peer_source_statuses,
)
from iris.cluster.federation.availability import AVAILABILITY_METRIC_VERSION
from iris.cluster.federation.protocol import (
    FederationPeerObservation,
    FederationResourceAvailability,
)
from iris.resources.capacity import (
    CapacityAction,
    CapacityBackend,
    CapacityDemandEntry,
    CapacityGroupRouting,
    CapacityKubernetesNode,
    CapacityKubernetesPod,
    CapacityKubernetesPool,
    CapacityKubernetesStatus,
    CapacityPeer,
    CapacityPeerBackend,
    CapacityRouting,
    CapacityScalingGroup,
    CapacityStatus,
    CapacityUnmetDemand,
    ResourceAvailability,
    RunningTaskPlacement,
    UnroutableJob,
)
from iris.resources.state import TaskState

_UNROUTABLE_SAMPLE_SIZE = 20


class CapacityResources:
    """Read one coherent controller-owned capacity and routing view."""

    def __init__(self, dependencies: ResourceDependencies) -> None:
        self._dependencies = dependencies

    def status(self) -> CapacityStatus:
        with self._dependencies.db.read_snapshot() as snapshot:
            pending = reads.task_counts_by_backend(snapshot, TaskState.PENDING)
            running = reads.task_counts_by_backend(snapshot, TaskState.RUNNING)
            workers_by_group = reads.worker_counts_by_scale_group(snapshot)
            placements = _running_placements(reads.running_task_band_rows(snapshot))

        worker_counts: dict[str, int] = {backend_id: 0 for backend_id in self._dependencies.backends}
        for scale_group, count in workers_by_group.items():
            backend_id = self._dependencies.runtime.backend_id_for_scale_group(scale_group)
            worker_counts[backend_id] = worker_counts.get(backend_id, 0) + count

        observed_at = Timestamp.now()
        backends: list[CapacityBackend] = []
        statuses = []
        for backend_id, backend in sorted(self._dependencies.backends.items()):
            try:
                authored = backend.status()
                availability = backend.resource_capacity()
            except (ConnectionError, ProviderError) as exc:
                statuses.append(_unavailable_backend_source(backend_id, exc))
                authored = _empty_backend_status(backend.capabilities)
                availability = None
            else:
                statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))

            worker = authored.worker
            autoscaler = worker.autoscaler if worker is not None else None
            authored_groups = tuple(
                _canonical_group_backend(group, backend_id) for group in (autoscaler.groups if autoscaler else ())
            )
            capacity_health: dict[str, int] = {}
            for group in authored_groups:
                state = group.availability_status or "unknown"
                capacity_health[state] = capacity_health.get(state, 0) + 1
            last_evaluation = autoscaler.last_evaluation if autoscaler else None
            backends.append(
                CapacityBackend(
                    backend_id=backend_id,
                    name=backend.name,
                    kind=_backend_kind(backend.capabilities),
                    capabilities=tuple(sorted(capability.value for capability in backend.capabilities)),
                    advertised_attributes={
                        key: tuple(sorted(values)) for key, values in sorted(backend.advertised_attributes().items())
                    },
                    worker_count=worker_counts.get(backend_id, 0),
                    pending_task_count=pending.get(backend_id, 0),
                    running_task_count=running.get(backend_id, 0),
                    has_autoscaler=backend.autoscaler is not None,
                    capacity_health=capacity_health,
                    availability=(_local_availability(availability, observed_at) if availability is not None else None),
                    scaling_groups=tuple(
                        _scaling_group(
                            group,
                            cluster_id=self._dependencies.cluster_id,
                            observed_at=last_evaluation or observed_at,
                        )
                        for group in authored_groups
                    ),
                    recent_actions=tuple(
                        CapacityAction(
                            action.timestamp,
                            action.action_type,
                            action.scale_group,
                            action.slice_id,
                            action.reason,
                            action.status,
                        )
                        for action in (autoscaler.recent_actions if autoscaler else ())
                    ),
                    routing=_routing(autoscaler.last_routing_decision) if autoscaler else None,
                    last_evaluation=last_evaluation,
                    healthy_worker_count=worker.healthy_worker_count if worker is not None else 0,
                    kubernetes=_kubernetes(authored.kubernetes),
                )
            )

        peer_observations = self._dependencies.runtime.federation.peer_observations()
        statuses.extend(
            peer_source_statuses(
                self._dependencies,
                {peer.peer_id for peer in peer_observations},
                observations={peer.peer_id: peer for peer in peer_observations},
            )
        )
        unroutable = self._dependencies.runtime.last_unroutable_jobs
        return CapacityStatus(
            backends=tuple(backends),
            peers=tuple(_peer(peer) for peer in peer_observations),
            running_placements=placements,
            unroutable_job_count=len(unroutable),
            unroutable_jobs=tuple(
                UnroutableJob(job_id, reason)
                for job_id, reason in list(sorted(unroutable.items()))[:_UNROUTABLE_SAMPLE_SIZE]
            ),
            source_statuses=tuple(statuses),
        )


def _backend_kind(capabilities: frozenset[BackendCapability]) -> str:
    if BackendCapability.CLUSTER_VIEW in capabilities:
        return "kubernetes"
    if BackendCapability.WORKER_DAEMON in capabilities:
        return "worker-daemon"
    return "unknown"


def _empty_backend_status(capabilities: frozenset[BackendCapability]) -> BackendStatus:
    if BackendCapability.CLUSTER_VIEW in capabilities:
        return BackendStatus(kubernetes=KubernetesStatus())
    return BackendStatus(worker=WorkerFleetStatus())


def _canonical_group_backend(group: ScaleGroupStatus, backend_id: str) -> ScaleGroupStatus:
    if group.backend_id and group.backend_id != backend_id:
        raise ValueError(
            f"backend {backend_id!r} authored scaling group {group.name!r} for backend {group.backend_id!r}"
        )
    return group if group.backend_id else replace(group, backend_id=backend_id)


def _local_availability(capacity: Mapping[str, DeviceCapacity], observed_at: Timestamp) -> ResourceAvailability:
    held_by_band: dict[int, dict[str, int]] = {}
    for token, device in capacity.items():
        for band, amount in device.held_by_band.items():
            held_by_band.setdefault(int(band), {})[token] = amount
    return ResourceAvailability(
        version=AVAILABILITY_METRIC_VERSION,
        observed_at=observed_at,
        amounts={token: value.free for token, value in capacity.items()},
        total_amounts={token: value.total for token, value in capacity.items()},
        held_by_band=held_by_band,
    )


def _peer_availability(value: FederationResourceAvailability | None) -> ResourceAvailability | None:
    if value is None:
        return None
    return ResourceAvailability(
        version=value.version,
        observed_at=Timestamp.from_ms(value.observation_epoch_ms),
        amounts=value.amounts,
        total_amounts=value.total_amounts,
        held_by_band=value.held_by_band,
    )


def _peer(value: FederationPeerObservation) -> CapacityPeer:
    return CapacityPeer(
        peer_id=value.peer_id,
        controller_address=value.controller_address,
        reachable=value.reachable,
        last_contact_ms=value.last_contact_ms,
        active_federated_jobs=value.active_federated_jobs,
        backends=tuple(
            CapacityPeerBackend(
                backend_id=backend.backend_id,
                name=backend.name,
                kind=backend.kind,
                capabilities=backend.capabilities,
                advertised_attributes=backend.advertised_attributes,
                scale_groups=backend.scale_groups,
                worker_count=backend.worker_count,
                pending_task_count=backend.pending_task_count,
                running_task_count=backend.running_task_count,
                has_autoscaler=backend.has_autoscaler,
                capacity_health=backend.capacity_health,
                availability=_peer_availability(backend.availability),
            )
            for backend in value.backends
        ),
    )


def _running_placements(rows: Sequence[reads.RunningTaskBandRecord]) -> tuple[RunningTaskPlacement, ...]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        job_id = (row.task_id.parent or row.task_id).to_wire()
        key = (str(row.backend_id or ""), str(row.worker_id), job_id, row.task_id.user)
        counts[key] = counts.get(key, 0) + 1
    return tuple(
        RunningTaskPlacement(backend_id, worker_id, job_id, user_id, count)
        for (backend_id, worker_id, job_id, user_id), count in sorted(counts.items())
    )


def _scaling_group(
    value: ScaleGroupStatus,
    *,
    cluster_id: str,
    observed_at: Timestamp,
) -> CapacityScalingGroup:
    return CapacityScalingGroup(
        name=value.name,
        backend_id=value.backend_id,
        device_type=value.device_type,
        device_variant=value.device_variant,
        quota_pool=value.quota_pool,
        allocation_tier=value.allocation_tier,
        region=value.region,
        current_demand=value.current_demand,
        peak_demand=value.peak_demand,
        backoff_until=value.backoff_until,
        consecutive_failures=value.consecutive_failures,
        last_scale_up=value.last_scale_up,
        last_scale_down=value.last_scale_down,
        slices=tuple(
            slice_detail_from_status(
                cluster_id=cluster_id,
                backend_id=value.backend_id,
                scaling_group_id=value.name,
                value=slice_status,
                observed_at=observed_at,
            )
            for slice_status in value.slices
        ),
        slice_state_counts=value.slice_state_counts,
        availability_status=value.availability_status,
        availability_reason=value.availability_reason,
        blocked_until=value.blocked_until,
        scale_up_cooldown_until=value.scale_up_cooldown_until,
        idle_threshold_ms=value.idle_threshold_ms,
    )


def _routing(value: RoutingStatus | None) -> CapacityRouting | None:
    if value is None:
        return None
    return CapacityRouting(
        unmet=tuple(
            CapacityUnmetDemand(
                CapacityDemandEntry(
                    task_ids=item.entry.task_ids,
                    coschedule_group_id=item.entry.coschedule_group_id,
                    device_type=item.entry.device_type,
                    device_variant=item.entry.device_variant,
                    preemptible=item.entry.preemptible,
                ),
                item.reason,
            )
            for item in value.unmet_entries
        ),
        groups=tuple(
            CapacityGroupRouting(
                scaling_group_id=item.group,
                priority=item.priority,
                assigned=item.assigned,
                launch=item.launch,
                decision=item.decision,
                reason=item.reason,
            )
            for item in value.group_statuses
        ),
    )


def _kubernetes(value: KubernetesStatus | None) -> CapacityKubernetesStatus | None:
    if value is None:
        return None
    return CapacityKubernetesStatus(
        namespace=value.namespace,
        total_nodes=value.total_nodes,
        schedulable_nodes=value.schedulable_nodes,
        allocatable_cpu=value.allocatable_cpu,
        allocatable_memory=value.allocatable_memory,
        pods=tuple(
            CapacityKubernetesPod(
                item.pod_name,
                item.task_id,
                item.phase,
                item.reason,
                item.message,
                item.last_transition,
                item.node_name,
            )
            for item in value.pod_statuses
        ),
        provider_version=value.provider_version,
        pools=tuple(
            CapacityKubernetesPool(
                item.name,
                item.instance_type,
                item.scale_group,
                item.target_nodes,
                item.current_nodes,
                item.queued_nodes,
                item.in_progress_nodes,
                item.autoscaling,
                item.min_nodes,
                item.max_nodes,
                item.capacity,
                item.quota,
            )
            for item in value.node_pools
        ),
        nodes=tuple(
            CapacityKubernetesNode(
                item.name,
                item.ready,
                item.schedulable,
                item.status_summary,
                item.instance_type,
                item.region,
                item.gpu_count,
                item.gpu_model,
                item.cpu_millicores,
                item.memory_bytes,
                item.disk_bytes,
                item.running_pods,
                item.created,
            )
            for item in value.nodes
        ),
    )
