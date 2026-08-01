# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA SMI inventory and slow-health adapter."""

import csv
import io
from dataclasses import dataclass

from rigging.probe_types import (
    MAX_RESULT_EVENTS,
    MAX_SUBPROCESS_OUTPUT_BYTES,
    CommandRunner,
    CommandStatus,
    MetricKind,
    ProbeCollection,
    ProbeEvent,
    ProbeMetric,
    ProbeOutcome,
    ProbeReason,
)

_QUERY_FIELDS = (
    "uuid",
    "pci.bus_id",
    "name",
    "driver_version",
    "memory.total",
    "compute_mode",
    "mig.mode.current",
    "power.limit",
    "vbios_version",
    "ecc.errors.uncorrected.volatile.total",
    "retired_pages.single_bit_ecc.count",
    "retired_pages.double_bit_ecc.count",
    "retired_pages.pending",
    "remapped_rows.correctable",
    "remapped_rows.uncorrectable",
    "remapped_rows.pending",
    "remapped_rows.failure",
)
_NOT_AVAILABLE = {"", "n/a", "[n/a]", "not supported", "unknown error"}


@dataclass(frozen=True)
class NvidiaSmiProbe:
    """Configured NVIDIA SMI inventory probe."""

    interval: float
    timeout: float
    name: str = "nvidia_smi"

    def collect(self, runner: CommandRunner) -> ProbeCollection:
        result = runner.run(
            (
                "nvidia-smi",
                f"--query-gpu={','.join(_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ),
            timeout=self.timeout,
            max_output_bytes=MAX_SUBPROCESS_OUTPUT_BYTES,
        )
        terminal = _terminal_collection(result.status)
        if terminal is not None:
            return terminal
        if result.returncode != 0:
            return ProbeCollection(ProbeOutcome.UNAVAILABLE, ProbeReason.NONZERO_EXIT)

        try:
            rows = list(csv.reader(io.StringIO(result.stdout.decode("utf-8"))))
            rows = [row for row in rows if any(field.strip() for field in row)]
            if not rows:
                return ProbeCollection(ProbeOutcome.NOT_APPLICABLE, ProbeReason.NO_DEVICES)
            if len(rows) > 256:
                return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.RESULT_LIMIT)
            metrics: list[ProbeMetric] = []
            events: list[ProbeEvent] = []
            for row in rows:
                row_metrics, row_events = _row_result(row)
                metrics.extend(row_metrics)
                events.extend(row_events[: MAX_RESULT_EVENTS - len(events)])
            return ProbeCollection.succeeded(metrics=tuple(metrics), events=tuple(events))
        except (UnicodeDecodeError, ValueError):
            return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.PARSE_ERROR)


def _terminal_collection(status: CommandStatus) -> ProbeCollection | None:
    if status is CommandStatus.NOT_FOUND:
        return ProbeCollection(ProbeOutcome.UNAVAILABLE, ProbeReason.TOOL_MISSING)
    if status is CommandStatus.TIMED_OUT:
        return ProbeCollection(ProbeOutcome.TIMED_OUT, ProbeReason.SUBPROCESS_TIMEOUT)
    if status is CommandStatus.OUTPUT_LIMIT:
        return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.OUTPUT_LIMIT)
    if status is CommandStatus.FAILED:
        return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.INTERNAL_ERROR)
    return None


def _row_result(row: list[str]) -> tuple[list[ProbeMetric], list[ProbeEvent]]:
    if len(row) != len(_QUERY_FIELDS):
        raise ValueError("unexpected nvidia-smi column count")
    values = {field: value.strip() for field, value in zip(_QUERY_FIELDS, row, strict=True)}
    gpu_uuid = _required(values["uuid"])
    pci_bus_id = _required(values["pci.bus_id"])
    identity = {"gpu_uuid": gpu_uuid, "pci_bus_id": pci_bus_id}
    inventory = {
        **identity,
        "device_kind": "gpu",
        "gpu_model": _required(values["name"]),
        "driver_version": _required(values["driver_version"]),
    }
    _add_optional(inventory, "compute_mode", values["compute_mode"])
    _add_optional(inventory, "mig_mode", values["mig.mode.current"])
    _add_optional(inventory, "vbios_version", values["vbios_version"])

    metrics = [ProbeMetric("hardware_inventory", MetricKind.GAUGE, 1.0, attributes=inventory)]
    _append_metric(metrics, "gpu_memory_total_bytes", values["memory.total"], identity, scale=1024**2, unit="By")
    _append_metric(metrics, "gpu_power_limit_watts", values["power.limit"], identity, unit="W")
    cumulative = {
        **identity,
        "source_kind": "nvidia_smi",
        "source_temporality": "cumulative_snapshot",
    }
    health_values = {
        "ecc_uncorrected": _append_metric(
            metrics,
            "gpu_ecc_uncorrected_errors",
            values["ecc.errors.uncorrected.volatile.total"],
            cumulative,
            unit="{error}",
        ),
        "retired_pages_single_bit": _append_metric(
            metrics,
            "gpu_retired_pages",
            values["retired_pages.single_bit_ecc.count"],
            {**cumulative, "error_kind": "single_bit_ecc"},
            unit="{page}",
        ),
        "retired_pages_double_bit": _append_metric(
            metrics,
            "gpu_retired_pages",
            values["retired_pages.double_bit_ecc.count"],
            {**cumulative, "error_kind": "double_bit_ecc"},
            unit="{page}",
        ),
        "retired_pages_pending": _append_metric(
            metrics,
            "gpu_retired_pages_pending",
            values["retired_pages.pending"],
            identity,
        ),
        "row_remapped_correctable": _append_metric(
            metrics,
            "gpu_row_remapped_rows",
            values["remapped_rows.correctable"],
            {**cumulative, "error_kind": "correctable"},
            unit="{row}",
        ),
        "row_remapped_uncorrectable": _append_metric(
            metrics,
            "gpu_row_remapped_rows",
            values["remapped_rows.uncorrectable"],
            {**cumulative, "error_kind": "uncorrectable"},
            unit="{row}",
        ),
        "row_remap_pending": _append_metric(
            metrics,
            "gpu_row_remap_pending",
            values["remapped_rows.pending"],
            identity,
        ),
        "row_remap_failure": _append_metric(
            metrics,
            "gpu_row_remap_failures",
            values["remapped_rows.failure"],
            cumulative,
            unit="{failure}",
        ),
    }
    anomalies = {key: value for key, value in health_values.items() if value is not None and value > 0}
    events = [ProbeEvent("gpu_health_anomaly", anomalies, attributes=identity)] if anomalies else []
    return metrics, events


def _append_metric(
    metrics: list[ProbeMetric],
    name: str,
    raw_value: str,
    attributes: dict[str, str],
    *,
    scale: float = 1.0,
    unit: str = "",
) -> float | None:
    value = _number(raw_value)
    if value is None:
        return None
    scaled = value * scale
    metrics.append(ProbeMetric(name, MetricKind.GAUGE, scaled, unit=unit, attributes=attributes))
    return scaled


def _number(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in _NOT_AVAILABLE:
        return None
    if normalized in {"true", "yes"}:
        return 1.0
    if normalized in {"false", "no"}:
        return 0.0
    return float(value)


def _required(value: str) -> str:
    if value.strip().lower() in _NOT_AVAILABLE:
        raise ValueError("required nvidia-smi identity field is unavailable")
    return value


def _add_optional(attributes: dict[str, str], name: str, value: str) -> None:
    if value.strip().lower() not in _NOT_AVAILABLE:
        attributes[name] = value
