# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encode native backend status records for the controller Connect API."""

from iris.backends.status import (
    AutoscalerStatus,
    BackendStatus,
    DemandEntryStatus,
    KubernetesStatus,
    RoutingStatus,
)
from iris.rpc import controller_pb2, vm_pb2
from iris.time_proto import timestamp_to_proto


def backend_status_to_proto(status: BackendStatus) -> controller_pb2.Controller.BackendStatus:
    result = controller_pb2.Controller.BackendStatus()
    if status.kubernetes is not None:
        result.kubernetes.CopyFrom(kubernetes_status_to_proto(status.kubernetes))
    else:
        assert status.worker is not None
        result.worker.CopyFrom(
            controller_pb2.Controller.WorkerFleetDetail(
                autoscaler=autoscaler_status_to_proto(status.worker.autoscaler),
                healthy_worker_count=status.worker.healthy_worker_count,
                total_worker_count=status.worker.total_worker_count,
            )
        )
    return result


def kubernetes_status_to_proto(status: KubernetesStatus) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
    return controller_pb2.Controller.GetKubernetesClusterStatusResponse(
        namespace=status.namespace,
        total_nodes=status.total_nodes,
        schedulable_nodes=status.schedulable_nodes,
        allocatable_cpu=status.allocatable_cpu,
        allocatable_memory=status.allocatable_memory,
        pod_statuses=[
            controller_pb2.Controller.KubernetesPodStatus(
                pod_name=item.pod_name,
                task_id=item.task_id,
                phase=item.phase,
                reason=item.reason,
                message=item.message,
                last_transition=(timestamp_to_proto(item.last_transition) if item.last_transition is not None else None),
                node_name=item.node_name,
            )
            for item in status.pod_statuses
        ],
        provider_version=status.provider_version,
        node_pools=[
            controller_pb2.Controller.NodePoolStatus(
                name=item.name,
                instance_type=item.instance_type,
                scale_group=item.scale_group,
                target_nodes=item.target_nodes,
                current_nodes=item.current_nodes,
                queued_nodes=item.queued_nodes,
                in_progress_nodes=item.in_progress_nodes,
                autoscaling=item.autoscaling,
                min_nodes=item.min_nodes,
                max_nodes=item.max_nodes,
                capacity=item.capacity,
                quota=item.quota,
            )
            for item in status.node_pools
        ],
        nodes=[
            controller_pb2.Controller.NodeStatus(
                name=item.name,
                ready=item.ready,
                schedulable=item.schedulable,
                status_summary=item.status_summary,
                instance_type=item.instance_type,
                region=item.region,
                gpu_count=item.gpu_count,
                gpu_model=item.gpu_model,
                cpu_millicores=item.cpu_millicores,
                memory_bytes=item.memory_bytes,
                disk_bytes=item.disk_bytes,
                running_pods=item.running_pods,
                created=item.created,
            )
            for item in status.nodes
        ],
    )


def autoscaler_status_to_proto(status: AutoscalerStatus) -> vm_pb2.AutoscalerStatus:
    result = vm_pb2.AutoscalerStatus(
        groups=[
            vm_pb2.ScaleGroupStatus(
                name=group.name,
                backend_id=group.backend_id,
                device_type=group.device_type,
                device_variant=group.device_variant,
                quota_pool=group.quota_pool,
                allocation_tier=group.allocation_tier,
                region=group.region,
                current_demand=group.current_demand,
                peak_demand=group.peak_demand,
                backoff_until=(timestamp_to_proto(group.backoff_until) if group.backoff_until is not None else None),
                consecutive_failures=group.consecutive_failures,
                last_scale_up=(timestamp_to_proto(group.last_scale_up) if group.last_scale_up is not None else None),
                last_scale_down=(
                    timestamp_to_proto(group.last_scale_down) if group.last_scale_down is not None else None
                ),
                slices=[
                    vm_pb2.SliceInfo(
                        slice_id=item.slice_id,
                        scale_group=item.scale_group,
                        created_at=(timestamp_to_proto(item.created_at) if item.created_at is not None else None),
                        vms=[
                            vm_pb2.VmInfo(
                                vm_id=vm.vm_id,
                                slice_id=vm.slice_id,
                                scale_group=vm.scale_group,
                                state=int(vm.state),
                                address=vm.address,
                                zone=vm.zone,
                                created_at=(timestamp_to_proto(vm.created_at) if vm.created_at is not None else None),
                                state_changed_at=(
                                    timestamp_to_proto(vm.state_changed_at) if vm.state_changed_at is not None else None
                                ),
                                worker_id=vm.worker_id,
                                worker_healthy=vm.worker_healthy,
                                usability=vm.usability,
                                init_phase=vm.init_phase,
                                init_log_tail=vm.init_log_tail,
                                init_error=vm.init_error,
                                running_task_count=vm.running_task_count,
                                labels=vm.labels,
                            )
                            for vm in item.vms
                        ],
                        error_message=item.error_message,
                        last_active=(timestamp_to_proto(item.last_active) if item.last_active is not None else None),
                        idle=item.idle,
                        state=item.state,
                        degraded_slot_count=item.degraded_slot_count,
                        capacity_status=item.capacity_status,
                    )
                    for item in group.slices
                ],
                slice_state_counts=group.slice_state_counts,
                availability_status=group.availability_status,
                availability_reason=group.availability_reason,
                blocked_until=(timestamp_to_proto(group.blocked_until) if group.blocked_until is not None else None),
                scale_up_cooldown_until=(
                    timestamp_to_proto(group.scale_up_cooldown_until)
                    if group.scale_up_cooldown_until is not None
                    else None
                ),
                idle_threshold_ms=group.idle_threshold_ms,
            )
            for group in status.groups
        ],
        current_demand=status.current_demand,
        last_evaluation=(timestamp_to_proto(status.last_evaluation) if status.last_evaluation is not None else None),
        recent_actions=[
            vm_pb2.AutoscalerAction(
                timestamp=(timestamp_to_proto(action.timestamp) if action.timestamp is not None else None),
                action_type=action.action_type,
                scale_group=action.scale_group,
                slice_id=action.slice_id,
                reason=action.reason,
                status=action.status,
            )
            for action in status.recent_actions
        ],
    )
    if status.last_routing_decision is not None:
        result.last_routing_decision.CopyFrom(_routing_status_to_proto(status.last_routing_decision))
    return result


def _demand_entry_to_proto(entry: DemandEntryStatus) -> vm_pb2.DemandEntryStatus:
    return vm_pb2.DemandEntryStatus(
        task_ids=entry.task_ids,
        coschedule_group_id=entry.coschedule_group_id,
        device_type=entry.device_type,
        device_variant=entry.device_variant,
        preemptible=entry.preemptible,
        resources=vm_pb2.ResourceSpec(
            cpu_millicores=entry.resources.cpu_millicores,
            memory_bytes=entry.resources.memory_bytes,
            disk_bytes=entry.resources.disk_bytes,
            gpu_count=entry.resources.gpu_count,
            tpu_count=entry.resources.tpu_count,
        ),
    )


def _routing_status_to_proto(status: RoutingStatus) -> vm_pb2.RoutingDecision:
    return vm_pb2.RoutingDecision(
        group_to_launch=status.group_to_launch,
        group_reasons=status.group_reasons,
        routed_entries={
            group: vm_pb2.DemandEntryStatusList(entries=[_demand_entry_to_proto(entry) for entry in entries])
            for group, entries in status.routed_entries.items()
        },
        unmet_entries=[
            vm_pb2.UnmetDemand(entry=_demand_entry_to_proto(item.entry), reason=item.reason)
            for item in status.unmet_entries
        ],
        group_statuses=[
            vm_pb2.GroupRoutingStatus(
                group=item.group,
                priority=item.priority,
                assigned=item.assigned,
                launch=item.launch,
                decision=item.decision,
                reason=item.reason,
            )
            for item in status.group_statuses
        ],
    )
