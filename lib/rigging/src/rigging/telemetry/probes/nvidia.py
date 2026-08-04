# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect sparse NVIDIA inventory and health summaries."""

import csv
import io
import logging

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
_ERROR_FIELDS = (
    "ecc_uncorrected",
    "retired_single_bit",
    "retired_double_bit",
    "retired_pending",
    "remapped_correctable",
    "remapped_uncorrectable",
    "remapped_pending",
    "remapped_failure",
)

logger = logging.getLogger(__name__)


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
        return any((getattr(self, field) or 0) > 0 for field in _ERROR_FIELDS)


class _NvidiaCollector:
    def __init__(self) -> None:
        self._inventory_limiter = RateLimiter(_INVENTORY_INTERVAL)
        self._publisher = MetricSnapshotPublisher(max_records=_MAX_METRICS)

    def collect(self, runner: BoundedCommandRunner) -> None:
        include_inventory = self._inventory_limiter.should_run()
        collected = _collect(runner, include_inventory=include_inventory, publisher=self._publisher)
        if include_inventory and not collected:
            self._inventory_limiter.reset()


def start() -> PeriodicProbe:
    """Collect NVIDIA summary health and sparse detail until shutdown."""
    collector = _NvidiaCollector()
    return PeriodicProbe("nvidia_smi", collector.collect)


def collect(runner: BoundedCommandRunner) -> None:
    """Collect one sample, including slow-changing inventory."""
    _collect(runner, include_inventory=True, publisher=MetricSnapshotPublisher(max_records=_MAX_METRICS))


def _collect(
    runner: BoundedCommandRunner,
    *,
    include_inventory: bool,
    publisher: MetricSnapshotPublisher,
) -> bool:
    deadline = Deadline.from_seconds(TIMEOUT)
    result = _query(runner, _EXTENDED_FIELDS, deadline)
    if result.status is CommandStatus.CANCELLED:
        return False
    fields = _EXTENDED_FIELDS
    outcome = "success"
    if result.output is not None and result.output.returncode != 0:
        result = _query(runner, _BASELINE_FIELDS, deadline)
        fields = _BASELINE_FIELDS
        outcome = "success_baseline"
    if result.status is CommandStatus.CANCELLED:
        return False
    if result.output is None:
        _record_health(result.status.value)
        return False
    if result.output.returncode != 0:
        _record_health("nonzero_exit")
        return False

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
        _record_health("invalid_payload")
        return False

    _record_health(outcome)
    current = telemetry.snapshot_attributes("nvidia_smi", telemetry.CURRENT_SNAPSHOT)
    healthy = sum(not device.abnormal for device in devices) if fields == _EXTENDED_FIELDS else 0
    abnormal = sum(device.abnormal for device in devices) if fields == _EXTENDED_FIELDS else 0
    states = {"total": len(devices), "healthy": healthy, "abnormal": abnormal}
    if fields == _BASELINE_FIELDS:
        states["unknown"] = len(devices)
    for state, count in states.items():
        telemetry.gauge("gpu_devices", unit="{device}").set(float(count), attributes={"device_state": state, **current})

    metrics = [metric for device in devices for metric in _device_metrics(device, include_inventory=include_inventory)]
    publisher.publish(metrics)
    return True


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


def _device_metrics(device: NvidiaDevice, *, include_inventory: bool) -> list[MetricSnapshot]:
    identity = {"gpu_uuid": device.uuid, "pci_bus_id": device.pci_bus_id}
    metrics: list[MetricSnapshot] = []
    if include_inventory:
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
        metrics.extend(
            (
                _snapshot("hardware_inventory", 1.0, "", inventory),
                _snapshot("gpu_memory_total_bytes", device.memory_total_mib * 1024**2, "By", identity),
            )
        )
        if device.power_limit_watts is not None:
            metrics.append(_snapshot("gpu_power_limit_watts", device.power_limit_watts, "W", identity))

    _append_positive(
        metrics,
        "gpu_ecc_uncorrected_errors",
        device.ecc_uncorrected,
        identity,
        unit="{error}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
    )
    _append_positive(
        metrics,
        "gpu_retired_pages",
        device.retired_single_bit,
        {**identity, "error_kind": "single_bit_ecc"},
        unit="{page}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
    )
    _append_positive(
        metrics,
        "gpu_retired_pages",
        device.retired_double_bit,
        {**identity, "error_kind": "double_bit_ecc"},
        unit="{page}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
    )
    _append_positive(metrics, "gpu_retired_pages_pending", device.retired_pending, identity)
    _append_positive(
        metrics,
        "gpu_row_remapped_rows",
        device.remapped_correctable,
        {**identity, "error_kind": "correctable"},
        unit="{row}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
    )
    _append_positive(
        metrics,
        "gpu_row_remapped_rows",
        device.remapped_uncorrectable,
        {**identity, "error_kind": "uncorrectable"},
        unit="{row}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
    )
    _append_positive(metrics, "gpu_row_remap_pending", device.remapped_pending, identity)
    _append_positive(metrics, "gpu_row_remap_failures", device.remapped_failure, identity, unit="{failure}")
    return metrics


def _append_positive(
    metrics: list[MetricSnapshot],
    name: str,
    value: float | None,
    attributes: dict[str, str],
    *,
    unit: str = "",
    temporality: str = telemetry.CURRENT_SNAPSHOT,
) -> None:
    if value is not None and value > 0:
        metrics.append(_snapshot(name, value, unit, attributes, temporality=temporality))


def _snapshot(
    name: str,
    value: float,
    unit: str,
    attributes: dict[str, str],
    *,
    temporality: str = telemetry.CURRENT_SNAPSHOT,
) -> MetricSnapshot:
    return MetricSnapshot(
        name=name,
        value=value,
        unit=unit,
        attributes=attributes,
        source_kind="nvidia_smi",
        source_temporality=temporality,
    )


def _record_health(outcome: str) -> None:
    current = {
        "outcome": outcome,
        **telemetry.snapshot_attributes("nvidia_smi", telemetry.CURRENT_SNAPSHOT),
    }
    available = outcome in {"success", "success_baseline"}
    telemetry.gauge("nvidia_health_available").set(float(available), attributes=current)
    if not available:
        telemetry.counter("nvidia_health_failures", unit="{failure}").add(
            1,
            attributes={
                "failure_kind": outcome,
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
