# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect sparse NVIDIA inventory and health summaries."""

import csv
import io
import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rigging import telemetry
from rigging.telemetry.metrics import MetricSnapshot, MetricSnapshotPublisher
from rigging.telemetry.probes.runner import BoundedCommandRunner, CommandResult, CommandStatus, PeriodicProbe
from rigging.timing import Deadline, RateLimiter

TIMEOUT = 5.0
_INVENTORY_INTERVAL = 60 * 60.0
_MAX_DEVICES = 256
_MAX_METRICS = 4_096
_MAX_FIELD_CHARS = 256
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


class NvidiaOutcome(StrEnum):
    """Terminal outcome of one NVIDIA telemetry query."""

    SUCCESS = "success"
    SUCCESS_BASELINE = "success_baseline"
    START_FAILED = "start_failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    OUTPUT_LIMIT = "output_limit"
    OUTPUT_FAILED = "output_failed"
    INVALID_TIMEOUT = "invalid_timeout"
    NONZERO_EXIT = "nonzero_exit"
    INVALID_PAYLOAD = "invalid_payload"


_COMMAND_FAILURES = {
    CommandStatus.START_FAILED: NvidiaOutcome.START_FAILED,
    CommandStatus.DEADLINE_EXCEEDED: NvidiaOutcome.DEADLINE_EXCEEDED,
    CommandStatus.OUTPUT_LIMIT: NvidiaOutcome.OUTPUT_LIMIT,
    CommandStatus.OUTPUT_FAILED: NvidiaOutcome.OUTPUT_FAILED,
    CommandStatus.INVALID_TIMEOUT: NvidiaOutcome.INVALID_TIMEOUT,
}


class NvidiaDevice(BaseModel):
    """Validated values parsed from one nvidia-smi CSV row."""

    model_config = ConfigDict(frozen=True)

    uuid: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    pci_bus_id: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    model: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    driver_version: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    memory_total_mib: float = Field(ge=0, allow_inf_nan=False)
    compute_mode: str | None = Field(default=None, max_length=_MAX_FIELD_CHARS)
    mig_mode: str | None = Field(default=None, max_length=_MAX_FIELD_CHARS)
    power_limit_watts: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vbios_version: str | None = Field(default=None, max_length=_MAX_FIELD_CHARS)
    ecc_uncorrected: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    retired_single_bit: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    retired_double_bit: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    retired_pending: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    remapped_correctable: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    remapped_uncorrectable: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    remapped_pending: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    remapped_failure: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @property
    def abnormal(self) -> bool:
        error_counts = (
            self.ecc_uncorrected,
            self.retired_single_bit,
            self.retired_double_bit,
            self.retired_pending,
            self.remapped_correctable,
            self.remapped_uncorrectable,
            self.remapped_pending,
            self.remapped_failure,
        )
        return any((value or 0) > 0 for value in error_counts)


@dataclass(frozen=True)
class NvidiaSample:
    """Validated devices and query outcome for one NVIDIA sample."""

    outcome: NvidiaOutcome
    devices: tuple[NvidiaDevice, ...] = ()


class _NvidiaCollector:
    def __init__(self) -> None:
        self._inventory_limiter = RateLimiter(_INVENTORY_INTERVAL)
        self._publisher = MetricSnapshotPublisher(max_records=_MAX_METRICS)

    def collect(self, runner: BoundedCommandRunner) -> None:
        inventory_due = self._inventory_limiter.should_run()
        sample = _sample(runner)
        if sample is None:
            if inventory_due:
                self._inventory_limiter.reset()
            return
        _publish_summary(sample)
        snapshots = [snapshot for device in sample.devices for snapshot in _error_snapshots(device)]
        if inventory_due:
            snapshots.extend(snapshot for device in sample.devices for snapshot in _inventory_snapshots(device))
        self._publisher.publish(snapshots)
        if inventory_due and not sample.devices:
            self._inventory_limiter.reset()


def start() -> PeriodicProbe:
    """Collect NVIDIA summary health and sparse detail until shutdown."""
    collector = _NvidiaCollector()
    return PeriodicProbe("nvidia_smi", collector.collect)


def collect(runner: BoundedCommandRunner) -> None:
    """Collect one sample, including slow-changing inventory."""
    sample = _sample(runner)
    if sample is None:
        return
    _publish_summary(sample)
    snapshots = [snapshot for device in sample.devices for snapshot in _error_snapshots(device)]
    snapshots.extend(snapshot for device in sample.devices for snapshot in _inventory_snapshots(device))
    MetricSnapshotPublisher(max_records=_MAX_METRICS).publish(snapshots)


def _sample(runner: BoundedCommandRunner) -> NvidiaSample | None:
    deadline = Deadline.from_seconds(TIMEOUT)
    result = _query(runner, _EXTENDED_FIELDS, deadline)
    if result.status is CommandStatus.CANCELLED:
        return None
    fields = _EXTENDED_FIELDS
    outcome = NvidiaOutcome.SUCCESS
    if result.output is not None and result.output.returncode != 0:
        result = _query(runner, _BASELINE_FIELDS, deadline)
        fields = _BASELINE_FIELDS
        outcome = NvidiaOutcome.SUCCESS_BASELINE
    if result.status is CommandStatus.CANCELLED:
        return None
    if result.output is None:
        return NvidiaSample(_COMMAND_FAILURES[result.status])
    if result.output.returncode != 0:
        return NvidiaSample(NvidiaOutcome.NONZERO_EXIT)

    try:
        parsed_rows = csv.reader(io.StringIO(result.output.stdout.decode()))
        rows = [row for row in parsed_rows if any(value.strip() for value in row)]
        if not rows:
            raise ValueError("nvidia-smi returned no devices")
        if len(rows) > _MAX_DEVICES:
            raise ValueError("nvidia-smi device limit exceeded")
        devices = tuple(_parse_device(row, fields) for row in rows)
    except (UnicodeDecodeError, ValueError) as error:
        logger.warning("could not parse nvidia-smi telemetry: %s", error)
        return NvidiaSample(NvidiaOutcome.INVALID_PAYLOAD)

    return NvidiaSample(outcome, devices)


def _publish_summary(sample: NvidiaSample) -> None:
    _record_health(sample.outcome)
    if not sample.devices:
        return
    current = telemetry.snapshot_attributes("nvidia_smi", telemetry.CURRENT_SNAPSHOT)
    extended = sample.outcome is NvidiaOutcome.SUCCESS
    healthy = sum(not device.abnormal for device in sample.devices) if extended else 0
    abnormal = sum(device.abnormal for device in sample.devices) if extended else 0
    states = {"total": len(sample.devices), "healthy": healthy, "abnormal": abnormal}
    if not extended:
        states["unknown"] = len(sample.devices)
    for state, count in states.items():
        telemetry.gauge("gpu_devices", unit="{device}").set(float(count), attributes={"device_state": state, **current})


def _query(
    runner: BoundedCommandRunner,
    fields: tuple[str, ...],
    deadline: Deadline,
) -> CommandResult:
    return runner.run_result(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        deadline.remaining_seconds(),
    )


def _parse_device(row: list[str], fields: tuple[str, ...]) -> NvidiaDevice:
    if len(row) != len(fields):
        raise ValueError("unexpected nvidia-smi column count")
    values = {field: value.strip() for field, value in zip(fields, row, strict=True)}
    parsed: dict[str, object] = {
        "uuid": _required(values["uuid"]),
        "pci_bus_id": _required(values["pci.bus_id"]),
        "model": _required(values["name"]),
        "driver_version": _required(values["driver_version"]),
        "memory_total_mib": _required_number(values["memory.total"]),
    }
    if fields == _EXTENDED_FIELDS:
        parsed.update(
            compute_mode=_available_text(values["compute_mode"]),
            mig_mode=_available_text(values["mig.mode.current"]),
            power_limit_watts=_available_number(values["power.limit"]),
            vbios_version=_available_text(values["vbios_version"]),
            ecc_uncorrected=_available_number(values["ecc.errors.uncorrected.volatile.total"]),
            retired_single_bit=_available_number(values["retired_pages.single_bit_ecc.count"]),
            retired_double_bit=_available_number(values["retired_pages.double_bit_ecc.count"]),
            retired_pending=_available_number(values["retired_pages.pending"]),
            remapped_correctable=_available_number(values["remapped_rows.correctable"]),
            remapped_uncorrectable=_available_number(values["remapped_rows.uncorrectable"]),
            remapped_pending=_available_number(values["remapped_rows.pending"]),
            remapped_failure=_available_number(values["remapped_rows.failure"]),
        )
    return NvidiaDevice.model_validate(parsed)


def _inventory_snapshots(device: NvidiaDevice) -> list[MetricSnapshot]:
    identity = {"gpu_uuid": device.uuid, "pci_bus_id": device.pci_bus_id}
    inventory = {
        **identity,
        "device_kind": "gpu",
        "gpu_model": device.model,
        "driver_version": device.driver_version,
    }
    for name, value in (
        ("compute_mode", device.compute_mode),
        ("mig_mode", device.mig_mode),
        ("vbios_version", device.vbios_version),
    ):
        if value is not None:
            inventory[name] = value
    snapshots = [
        MetricSnapshot(
            name="hardware_inventory",
            value=1.0,
            unit="",
            attributes=inventory,
            source_kind="nvidia_smi",
            source_temporality=telemetry.CURRENT_SNAPSHOT,
        ),
        MetricSnapshot(
            name="gpu_memory_total_bytes",
            value=device.memory_total_mib * 1024**2,
            unit="By",
            attributes=identity,
            source_kind="nvidia_smi",
            source_temporality=telemetry.CURRENT_SNAPSHOT,
        ),
    ]
    if device.power_limit_watts is not None:
        snapshots.append(
            MetricSnapshot(
                name="gpu_power_limit_watts",
                value=device.power_limit_watts,
                unit="W",
                attributes=identity,
                source_kind="nvidia_smi",
                source_temporality=telemetry.CURRENT_SNAPSHOT,
            )
        )
    return snapshots


def _error_snapshots(device: NvidiaDevice) -> list[MetricSnapshot]:
    snapshots = []
    if device.ecc_uncorrected:
        snapshots.append(
            MetricSnapshot(
                name="gpu_ecc_uncorrected_errors",
                value=device.ecc_uncorrected,
                unit="{error}",
                attributes={"gpu_uuid": device.uuid, "pci_bus_id": device.pci_bus_id},
                source_kind="nvidia_smi",
                source_temporality=telemetry.CUMULATIVE_SNAPSHOT,
            )
        )
    if device.retired_single_bit:
        snapshots.append(
            MetricSnapshot(
                name="gpu_retired_pages",
                value=device.retired_single_bit,
                unit="{page}",
                attributes={
                    "gpu_uuid": device.uuid,
                    "pci_bus_id": device.pci_bus_id,
                    "error_kind": "single_bit_ecc",
                },
                source_kind="nvidia_smi",
                source_temporality=telemetry.CUMULATIVE_SNAPSHOT,
            )
        )
    if device.retired_double_bit:
        snapshots.append(
            MetricSnapshot(
                name="gpu_retired_pages",
                value=device.retired_double_bit,
                unit="{page}",
                attributes={
                    "gpu_uuid": device.uuid,
                    "pci_bus_id": device.pci_bus_id,
                    "error_kind": "double_bit_ecc",
                },
                source_kind="nvidia_smi",
                source_temporality=telemetry.CUMULATIVE_SNAPSHOT,
            )
        )
    if device.retired_pending:
        snapshots.append(
            MetricSnapshot(
                name="gpu_retired_pages_pending",
                value=device.retired_pending,
                unit="",
                attributes={"gpu_uuid": device.uuid, "pci_bus_id": device.pci_bus_id},
                source_kind="nvidia_smi",
                source_temporality=telemetry.CURRENT_SNAPSHOT,
            )
        )
    if device.remapped_correctable:
        snapshots.append(
            MetricSnapshot(
                name="gpu_row_remapped_rows",
                value=device.remapped_correctable,
                unit="{row}",
                attributes={
                    "gpu_uuid": device.uuid,
                    "pci_bus_id": device.pci_bus_id,
                    "error_kind": "correctable",
                },
                source_kind="nvidia_smi",
                source_temporality=telemetry.CUMULATIVE_SNAPSHOT,
            )
        )
    if device.remapped_uncorrectable:
        snapshots.append(
            MetricSnapshot(
                name="gpu_row_remapped_rows",
                value=device.remapped_uncorrectable,
                unit="{row}",
                attributes={
                    "gpu_uuid": device.uuid,
                    "pci_bus_id": device.pci_bus_id,
                    "error_kind": "uncorrectable",
                },
                source_kind="nvidia_smi",
                source_temporality=telemetry.CUMULATIVE_SNAPSHOT,
            )
        )
    if device.remapped_pending:
        snapshots.append(
            MetricSnapshot(
                name="gpu_row_remap_pending",
                value=device.remapped_pending,
                unit="",
                attributes={"gpu_uuid": device.uuid, "pci_bus_id": device.pci_bus_id},
                source_kind="nvidia_smi",
                source_temporality=telemetry.CURRENT_SNAPSHOT,
            )
        )
    if device.remapped_failure:
        snapshots.append(
            MetricSnapshot(
                name="gpu_row_remap_failures",
                value=device.remapped_failure,
                unit="{failure}",
                attributes={"gpu_uuid": device.uuid, "pci_bus_id": device.pci_bus_id},
                source_kind="nvidia_smi",
                source_temporality=telemetry.CURRENT_SNAPSHOT,
            )
        )
    return snapshots


def _record_health(outcome: NvidiaOutcome) -> None:
    current = {
        "outcome": outcome.value,
        **telemetry.snapshot_attributes("nvidia_smi", telemetry.CURRENT_SNAPSHOT),
    }
    available = outcome in {NvidiaOutcome.SUCCESS, NvidiaOutcome.SUCCESS_BASELINE}
    telemetry.gauge("nvidia_health_available").set(float(available), attributes=current)
    if not available:
        telemetry.counter("nvidia_health_failures", unit="{failure}").add(
            1,
            attributes={
                "failure_kind": outcome.value,
                **telemetry.snapshot_attributes("nvidia_smi", telemetry.CUMULATIVE_SNAPSHOT),
            },
        )


def _available_number(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in _NOT_AVAILABLE:
        return None
    if normalized in {"true", "yes"}:
        return 1.0
    if normalized in {"false", "no"}:
        return 0.0
    return float(value)


def _required_number(value: str) -> float:
    number = _available_number(value)
    if number is None:
        raise ValueError("required nvidia-smi numeric field is unavailable")
    return number


def _required(value: str) -> str:
    available = _available_text(value)
    if available is None:
        raise ValueError("required nvidia-smi identity field is unavailable")
    return available


def _available_text(value: str) -> str | None:
    return None if value.strip().lower() in _NOT_AVAILABLE else value
