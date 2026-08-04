# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect stable NVIDIA inventory and slow-health metrics."""

import csv
import io
import logging
from typing import NamedTuple

from rigging import telemetry
from rigging.telemetry.probes.runner import BoundedCommandRunner, CommandOutput, PeriodicProbe
from rigging.timing import Deadline

TIMEOUT = 5.0
_MAX_DEVICES = 256
_BASELINE_FIELDS = (
    "uuid",
    "pci.bus_id",
    "name",
    "driver_version",
    "memory.total",
)
_EXTENDED_FIELDS = (
    *_BASELINE_FIELDS,
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
_NOT_AVAILABLE = frozenset({"", "n/a", "[n/a]", "not supported", "[not supported]", "unknown error"})

logger = logging.getLogger(__name__)


def start() -> PeriodicProbe:
    """Collect NVIDIA inventory and slow-health evidence until shutdown."""
    return PeriodicProbe("nvidia_smi", collect)


class _Metric(NamedTuple):
    name: str
    value: float
    unit: str
    attributes: dict[str, str]


def collect(runner: BoundedCommandRunner) -> None:
    deadline = Deadline.from_seconds(TIMEOUT)
    result = _query(runner, _EXTENDED_FIELDS, deadline)
    if result is None:
        return
    fields = _EXTENDED_FIELDS
    if result.returncode != 0:
        result = _query(runner, _BASELINE_FIELDS, deadline)
        fields = _BASELINE_FIELDS
    if result is None or result.returncode != 0:
        return

    try:
        rows = [row for row in csv.reader(io.StringIO(result.stdout.decode())) if any(value.strip() for value in row)]
        if len(rows) > _MAX_DEVICES:
            raise ValueError("nvidia-smi device limit exceeded")
        metrics = [metric for row in rows for metric in _row_metrics(row, fields)]
    except (UnicodeDecodeError, ValueError) as error:
        logger.warning("could not parse nvidia-smi telemetry: %s", error)
        return
    for metric in metrics:
        telemetry.gauge(metric.name, unit=metric.unit).set(metric.value, attributes=metric.attributes)


def _query(
    runner: BoundedCommandRunner,
    fields: tuple[str, ...],
    deadline: Deadline,
) -> CommandOutput | None:
    return runner.run_result(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        deadline.remaining_seconds(),
    ).output


def _row_metrics(row: list[str], fields: tuple[str, ...]) -> list[_Metric]:
    if len(row) != len(fields):
        raise ValueError("unexpected nvidia-smi column count")
    values = {field: value.strip() for field, value in zip(fields, row, strict=True)}
    identity = {
        "gpu_uuid": _required(values["uuid"]),
        "pci_bus_id": _required(values["pci.bus_id"]),
    }
    current = {**identity, **telemetry.snapshot_attributes("nvidia_smi", telemetry.CURRENT_SNAPSHOT)}
    inventory = {
        **current,
        "device_kind": "gpu",
        "gpu_model": _required(values["name"]),
        "driver_version": _required(values["driver_version"]),
    }
    if fields == _EXTENDED_FIELDS:
        _add_optional(inventory, "compute_mode", values["compute_mode"])
        _add_optional(inventory, "mig_mode", values["mig.mode.current"])
        _add_optional(inventory, "vbios_version", values["vbios_version"])

    metrics: list[_Metric] = [_Metric("hardware_inventory", 1.0, "", inventory)]
    _append(metrics, "gpu_memory_total_bytes", values["memory.total"], current, scale=1024**2, unit="By")
    if fields == _BASELINE_FIELDS:
        return metrics

    _append(metrics, "gpu_power_limit_watts", values["power.limit"], current, unit="W")
    cumulative = {**identity, **telemetry.snapshot_attributes("nvidia_smi", telemetry.CUMULATIVE_SNAPSHOT)}
    _append(
        metrics,
        "gpu_ecc_uncorrected_errors",
        values["ecc.errors.uncorrected.volatile.total"],
        cumulative,
        unit="{error}",
    )
    _append(
        metrics,
        "gpu_retired_pages",
        values["retired_pages.single_bit_ecc.count"],
        {**cumulative, "error_kind": "single_bit_ecc"},
        unit="{page}",
    )
    _append(
        metrics,
        "gpu_retired_pages",
        values["retired_pages.double_bit_ecc.count"],
        {**cumulative, "error_kind": "double_bit_ecc"},
        unit="{page}",
    )
    _append(metrics, "gpu_retired_pages_pending", values["retired_pages.pending"], current)
    _append(
        metrics,
        "gpu_row_remapped_rows",
        values["remapped_rows.correctable"],
        {**cumulative, "error_kind": "correctable"},
        unit="{row}",
    )
    _append(
        metrics,
        "gpu_row_remapped_rows",
        values["remapped_rows.uncorrectable"],
        {**cumulative, "error_kind": "uncorrectable"},
        unit="{row}",
    )
    _append(metrics, "gpu_row_remap_pending", values["remapped_rows.pending"], current)
    _append(metrics, "gpu_row_remap_failures", values["remapped_rows.failure"], current, unit="{failure}")
    return metrics


def _append(
    metrics: list[_Metric],
    name: str,
    raw_value: str,
    attributes: dict[str, str],
    *,
    scale: float = 1.0,
    unit: str = "",
) -> None:
    value = _available_number(raw_value)
    if value is not None:
        metrics.append(_Metric(name, value * scale, unit, attributes))


def _available_number(value: str) -> float | None:
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
