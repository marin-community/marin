# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalized physical-node telemetry shared by node-agent collectors."""

from dataclasses import dataclass

from rigging import telemetry


@dataclass(frozen=True)
class DeviceMetric:
    """One accelerator measurement with stable device identity."""

    name: str
    value: float
    unit: str
    temporality: str
    gpu_index: str
    gpu_uuid: str
    pci_bus_id: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class NodeTarget:
    """Stable identity and exporter address for one physical node."""

    name: str
    node_uid: str
    internal_ip: str
    device_type: str = ""
    device_variant: str = ""


@dataclass
class NodeMetrics:
    """Host and accelerator readings from one bounded collection pass."""

    cpu_pct: float | None = None
    mem_used_bytes: int | None = None
    mem_total_bytes: int | None = None
    disk_used_bytes: int | None = None
    disk_total_bytes: int | None = None
    net_recv_bytes: int | None = None
    net_sent_bytes: int | None = None
    gpu_count: int | None = None
    gpu_model: str = ""
    hbm_used_bytes: int | None = None
    hbm_total_bytes: int | None = None
    gpu_util_pct: float | None = None
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None
    gpu_power_limit_w: float | None = None
    host_available: bool = False
    host_source_kind: str = "node_exporter"
    host_source_replica_uid: str = ""
    dcgm_available: bool = False
    dcgm_exporter_uid: str = ""
    device_metrics: tuple[DeviceMetric, ...] = ()


def publish_node_telemetry(target: NodeTarget, metrics: NodeMetrics, timestamp_seconds: float) -> None:
    """Publish one normalized physical-node snapshot."""
    if not target.node_uid:
        return
    node_identity = {"node_name": target.name, "node_uid": target.node_uid}
    host_current = {
        **node_identity,
        **telemetry.snapshot_attributes(metrics.host_source_kind, telemetry.CURRENT_SNAPSHOT),
    }
    telemetry.gauge("hardware_source_available").set(float(metrics.host_available), attributes=host_current)
    if metrics.host_available:
        telemetry.gauge("progress_time_seconds", unit="s").set(
            timestamp_seconds,
            attributes={**host_current, "progress_kind": "hardware_source"},
        )
        current_metrics = (
            ("node_cpu_utilization_percent", metrics.cpu_pct, "%"),
            ("node_memory_used_bytes", metrics.mem_used_bytes, "By"),
            ("node_memory_total_bytes", metrics.mem_total_bytes, "By"),
            ("node_disk_used_bytes", metrics.disk_used_bytes, "By"),
            ("node_disk_total_bytes", metrics.disk_total_bytes, "By"),
        )
        for name, value, unit in current_metrics:
            if value is not None:
                telemetry.gauge(name, unit=unit).set(value, attributes=host_current)
        host_cumulative = {
            **node_identity,
            "source_replica_uid": metrics.host_source_replica_uid,
            **telemetry.snapshot_attributes(metrics.host_source_kind, telemetry.CUMULATIVE_SNAPSHOT),
        }
        telemetry.gauge("node_network_receive_bytes", unit="By").set(
            metrics.net_recv_bytes or 0, attributes=host_cumulative
        )
        telemetry.gauge("node_network_transmit_bytes", unit="By").set(
            metrics.net_sent_bytes or 0, attributes=host_cumulative
        )

    expects_dcgm = target.device_type == "gpu" or bool(target.device_variant) or metrics.gpu_count is not None
    if not expects_dcgm:
        return
    dcgm_current = {**node_identity, **telemetry.snapshot_attributes("dcgm", telemetry.CURRENT_SNAPSHOT)}
    if metrics.dcgm_exporter_uid:
        dcgm_current["dcgm_exporter_uid"] = metrics.dcgm_exporter_uid
    telemetry.gauge("hardware_source_available").set(float(metrics.dcgm_available), attributes=dcgm_current)
    if metrics.dcgm_available:
        telemetry.gauge("progress_time_seconds", unit="s").set(
            timestamp_seconds,
            attributes={**dcgm_current, "progress_kind": "hardware_source"},
        )
    for metric in metrics.device_metrics:
        attributes = {
            "dcgm_exporter_uid": metrics.dcgm_exporter_uid,
            "gpu_index": metric.gpu_index,
            "gpu_uuid": metric.gpu_uuid,
            **node_identity,
            "pci_bus_id": metric.pci_bus_id,
            "source_replica_uid": metrics.dcgm_exporter_uid,
            **telemetry.snapshot_attributes("dcgm", metric.temporality),
            **metric.attributes,
        }
        telemetry.gauge(metric.name, unit=metric.unit).set(metric.value, attributes=attributes)
