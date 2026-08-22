# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect bounded NCCL RAS summaries and anomalous-rank evidence."""

import logging
import math
import sys
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic

from pydantic import ValidationError

from rigging import telemetry
from rigging.telemetry.metrics import MetricSnapshot, MetricSnapshotPublisher
from rigging.telemetry.probes import nccl_client, nccl_ras
from rigging.telemetry.probes.runner import BoundedCommandRunner, CommandStatus, PeriodicProbe

TIMEOUT = 8.0
_DEFAULT_INTERVAL = 10 * 60.0
_MAX_CLIENT_RESULT_BYTES = 32 * 1024 * 1024
_METRIC_RECORDS_NAME = "ras_metric_records"
_CLIENT_COMMAND = (sys.executable, "-m", nccl_client.__name__)
_CLIENT_FAILURES = {
    nccl_client.TIMEOUT_EXIT_CODE: nccl_ras.NcclRasResult.CLIENT_TIMEOUT,
    nccl_client.UNAVAILABLE_EXIT_CODE: nccl_ras.NcclRasResult.UNAVAILABLE,
    nccl_client.INVALID_CONFIG_EXIT_CODE: nccl_ras.NcclRasResult.INVALID_CLIENT_CONFIG,
    nccl_client.OUTPUT_LIMIT_EXIT_CODE: nccl_ras.NcclRasResult.CLIENT_RESPONSE_LIMIT,
    nccl_client.INVALID_PAYLOAD_EXIT_CODE: nccl_ras.NcclRasResult.INVALID_PAYLOAD,
}
_RUNNER_FAILURES = {
    CommandStatus.CANCELLED: nccl_ras.NcclRasResult.CANCELLED,
    CommandStatus.START_FAILED: nccl_ras.NcclRasResult.START_FAILED,
    CommandStatus.DEADLINE_EXCEEDED: nccl_ras.NcclRasResult.DEADLINE_EXCEEDED,
    CommandStatus.OUTPUT_LIMIT: nccl_ras.NcclRasResult.RUNNER_OUTPUT_LIMIT,
    CommandStatus.OUTPUT_FAILED: nccl_ras.NcclRasResult.OUTPUT_FAILED,
    CommandStatus.INVALID_TIMEOUT: nccl_ras.NcclRasResult.INVALID_TIMEOUT,
}

logger = logging.getLogger(__name__)


class RasTrigger(StrEnum):
    """Reason one RAS collection was requested."""

    PERIODIC = "periodic"
    STALL = "stall"


@dataclass(frozen=True)
class NcclRasCollection:
    """Validated client output and parent-observed collection duration."""

    client_output: nccl_ras.NcclRasClientOutput
    duration_seconds: float


class NcclRasSession:
    """Own periodic collection and provide one serialized stall capture path."""

    def __init__(self, *, interval: float) -> None:
        self._periodic = PeriodicProbe(
            "nccl_ras",
            self._collect_periodic,
            interval=interval,
            max_output_bytes=_MAX_CLIENT_RESULT_BYTES,
        )

    def _collect_periodic(self, runner: BoundedCommandRunner) -> None:
        result = collect_ras(runner, detail=nccl_ras.RasDetail.PERIODIC, timeout=TIMEOUT)
        self._publish(result, RasTrigger.PERIODIC)

    def capture_stall(self, timeout: float) -> NcclRasCollection:
        """Stop periodic work, run one fresh detailed query, and publish its summary."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("stall capture timeout must be positive and finite")
        self._periodic.shutdown(min(2.0, timeout))
        runner = BoundedCommandRunner(max_output_bytes=_MAX_CLIENT_RESULT_BYTES)
        result = collect_ras(runner, detail=nccl_ras.RasDetail.STALL, timeout=timeout)
        self._publish(result, RasTrigger.STALL)
        return result

    def _publish(self, collection: NcclRasCollection, trigger: RasTrigger) -> None:
        result = collection.client_output
        if result.result is nccl_ras.NcclRasResult.CANCELLED:
            return
        _record_poll(
            result.result,
            collection.duration_seconds,
            trigger,
            observed_bytes=result.observed_bytes,
            limit_bytes=result.limit_bytes,
        )
        if result.result in {
            nccl_ras.NcclRasResult.CLIENT_TIMEOUT,
            nccl_ras.NcclRasResult.DEADLINE_EXCEEDED,
        }:
            _record_timeout(trigger)
        if result.report is None:
            return
        snapshots = ras_snapshots(result.report, trigger=trigger)
        published = MetricSnapshotPublisher(max_records=len(snapshots)).publish(snapshots)
        attributes = {
            "trigger": trigger.value,
            **telemetry.snapshot_attributes("nccl_ras", telemetry.CURRENT_SNAPSHOT),
        }
        telemetry.gauge(_METRIC_RECORDS_NAME, unit="{record}").set(
            float(len(snapshots)), attributes={**attributes, "record_state": "input"}
        )
        telemetry.gauge(_METRIC_RECORDS_NAME, unit="{record}").set(
            float(published.enqueued_records), attributes={**attributes, "record_state": "enqueued"}
        )
        telemetry.gauge(_METRIC_RECORDS_NAME, unit="{record}").set(
            float(published.telemetry_lost_records), attributes={**attributes, "record_state": "telemetry_lost"}
        )

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop periodic collection and reap its active client."""
        self._periodic.shutdown(timeout)


def start(*, interval: float = _DEFAULT_INTERVAL) -> NcclRasSession:
    """Collect NCCL RAS summaries until shutdown."""
    return NcclRasSession(interval=interval)


def collect_ras(
    runner: BoundedCommandRunner,
    *,
    detail: nccl_ras.RasDetail,
    timeout: float,
) -> NcclRasCollection:
    """Run the bounded client and return its validated compact result."""
    client_timeout = max(1, math.ceil(timeout * 0.75))
    started = monotonic()
    command = runner.run_result(
        (*_CLIENT_COMMAND, "--timeout", str(client_timeout), "--detail", detail.value),
        timeout,
    )
    duration = max(0.0, monotonic() - started)
    if command.output is None:
        client_output = nccl_ras.NcclRasClientOutput.failure(
            _RUNNER_FAILURES[command.status],
            f"NCCL RAS client runner ended with {command.status.value}",
            observed_bytes=command.observed_output_bytes,
            limit_bytes=runner.max_output_bytes if command.status is CommandStatus.OUTPUT_LIMIT else None,
        )
    elif command.output.returncode != 0:
        fallback_result = _CLIENT_FAILURES.get(command.output.returncode, nccl_ras.NcclRasResult.NONZERO_EXIT)
        try:
            client_output = nccl_ras.NcclRasClientOutput.from_bytes(command.output.stdout)
        except (ValidationError, ValueError) as error:
            logger.warning(
                "could not parse NCCL RAS client failure for exit code %d: %s",
                command.output.returncode,
                error,
            )
            client_output = nccl_ras.NcclRasClientOutput.failure(
                fallback_result, f"client exited {command.output.returncode}"
            )
        if client_output.result is nccl_ras.NcclRasResult.SUCCESS:
            client_output = nccl_ras.NcclRasClientOutput.failure(
                nccl_ras.NcclRasResult.INVALID_CLIENT_OUTPUT,
                f"client returned success payload with exit code {command.output.returncode}",
            )
    else:
        try:
            client_output = nccl_ras.NcclRasClientOutput.from_bytes(command.output.stdout)
        except (ValidationError, ValueError) as error:
            logger.warning("could not parse NCCL RAS client result: %s", error)
            client_output = nccl_ras.NcclRasClientOutput.failure(
                nccl_ras.NcclRasResult.INVALID_CLIENT_OUTPUT, str(error)
            )
    return NcclRasCollection(client_output=client_output, duration_seconds=duration)


def ras_snapshots(report: nccl_ras.NcclRasReport, *, trigger: RasTrigger) -> tuple[MetricSnapshot, ...]:
    """Translate one compact report into bounded, independently publishable snapshots."""
    snapshots: list[MetricSnapshot] = [
        MetricSnapshot(
            name="communicators",
            value=float(report.input_communicators),
            unit="{communicator}",
            attributes={"trigger": trigger.value, "detail": report.detail.value},
            source_kind="nccl_ras",
            source_temporality=telemetry.CURRENT_SNAPSHOT,
        ),
        MetricSnapshot(
            name="runtime_inventory",
            value=1.0,
            unit="",
            attributes={
                "runtime_kind": "nccl",
                "nccl_version": report.nccl_version,
                "cuda_runtime_version": report.cuda_runtime_version,
                "cuda_driver_version": report.cuda_driver_version,
                "trigger": trigger.value,
                "detail": report.detail.value,
            },
            source_kind="nccl_ras",
            source_temporality=telemetry.CURRENT_SNAPSHOT,
        ),
        MetricSnapshot(
            name="ras_collection_duration_seconds",
            value=report.collection_time_seconds,
            unit="s",
            attributes={"trigger": trigger.value, "detail": report.detail.value},
            source_kind="nccl_ras",
            source_temporality=telemetry.CURRENT_SNAPSHOT,
        ),
        MetricSnapshot(
            name="ras_collection_timeouts",
            value=float(report.ras_timeouts),
            unit="{timeout}",
            attributes={"trigger": trigger.value, "detail": report.detail.value},
            source_kind="nccl_ras",
            source_temporality=telemetry.CURRENT_SNAPSHOT,
        ),
    ]
    reduction_counts = {
        "communicators_input": report.input_communicators,
        "communicators_emitted": report.emitted_communicators,
        "communicators_invalid": report.invalid_communicators,
        "communicators_omitted": report.omitted_communicators,
    }
    snapshots.extend(
        MetricSnapshot(
            name="ras_reduction_records",
            value=float(count),
            unit="{record}",
            attributes={"record_state": state, "trigger": trigger.value, "detail": report.detail.value},
            source_kind="nccl_ras",
            source_temporality=telemetry.CURRENT_SNAPSHOT,
        )
        for state, count in reduction_counts.items()
    )
    for communicator in report.communicators:
        snapshots.append(
            MetricSnapshot(
                name="communicator_state",
                value=1.0,
                unit="",
                attributes={
                    "communicator_hash": communicator.communicator_hash,
                    "secondary_hash": communicator.secondary_hash,
                    "lifecycle_state": communicator.lifecycle_state,
                    "collective_mismatch": str(communicator.collective_mismatch).lower(),
                    "trigger": trigger.value,
                    "detail": report.detail.value,
                },
                source_kind="nccl_ras",
                source_temporality=telemetry.CURRENT_SNAPSHOT,
            )
        )
        rank_counts = {
            "total": communicator.size,
            "reported": communicator.reported_ranks,
            "missing": communicator.missing_ranks,
            "unresponsive": communicator.unresponsive_ranks,
            "considered_dead": communicator.considered_dead_ranks,
        }
        snapshots.extend(
            MetricSnapshot(
                name="communicator_ranks",
                value=float(count),
                unit="{rank}",
                attributes={
                    "communicator_hash": communicator.communicator_hash,
                    "secondary_hash": communicator.secondary_hash,
                    "rank_state": state,
                    "trigger": trigger.value,
                    "detail": report.detail.value,
                },
                source_kind="nccl_ras",
                source_temporality=telemetry.CURRENT_SNAPSHOT,
            )
            for state, count in rank_counts.items()
        )
    for rank in report.rank_observations:
        snapshots.append(
            MetricSnapshot(
                name="communicator_rank_status",
                value=float(rank.rank_state == "reported"),
                unit="",
                attributes={
                    "communicator_hash": rank.communicator_hash,
                    "secondary_hash": rank.secondary_hash,
                    "rank": str(rank.rank),
                    "rank_host": rank.host,
                    "process_id": str(rank.pid),
                    "cuda_device": str(rank.cuda_device),
                    "nvml_device": str(rank.nvml_device),
                    "rank_state": rank.rank_state,
                    "reasons": ",".join(rank.reasons),
                    "init_state": str(rank.init_state),
                    "async_error": str(rank.async_error),
                    "finalize_called": str(rank.finalize_called).lower(),
                    "destroy_flag": str(rank.destroy_flag).lower(),
                    "abort_flag": str(rank.abort_flag).lower(),
                    "unresponsive": str(rank.unresponsive).lower(),
                    "considered_dead": str(rank.considered_dead).lower(),
                    "trigger": trigger.value,
                    "detail": report.detail.value,
                },
                source_kind="nccl_ras",
                source_temporality=telemetry.CURRENT_SNAPSHOT,
            )
        )
    for progress in report.progress:
        snapshots.extend(
            (
                MetricSnapshot(
                    name="collective_operations",
                    value=float(progress.minimum),
                    unit="{operation}",
                    attributes={
                        "communicator_hash": progress.communicator_hash,
                        "secondary_hash": progress.secondary_hash,
                        "collective": progress.collective,
                        "rank_statistic": "minimum",
                        "trigger": trigger.value,
                        "detail": report.detail.value,
                    },
                    source_kind="nccl_ras",
                    source_temporality=telemetry.CUMULATIVE_SNAPSHOT,
                ),
                MetricSnapshot(
                    name="collective_operations",
                    value=float(progress.maximum),
                    unit="{operation}",
                    attributes={
                        "communicator_hash": progress.communicator_hash,
                        "secondary_hash": progress.secondary_hash,
                        "collective": progress.collective,
                        "rank_statistic": "maximum",
                        "trigger": trigger.value,
                        "detail": report.detail.value,
                    },
                    source_kind="nccl_ras",
                    source_temporality=telemetry.CUMULATIVE_SNAPSHOT,
                ),
            )
        )
    return tuple(snapshots)


def _record_timeout(trigger: RasTrigger) -> None:
    telemetry.counter("ras_poll_timeouts", unit="{timeout}").add(
        1,
        attributes={
            "trigger": trigger.value,
            **telemetry.snapshot_attributes("nccl_ras", telemetry.CUMULATIVE_SNAPSHOT),
        },
    )


def _record_poll(
    outcome: nccl_ras.NcclRasResult,
    duration: float,
    trigger: RasTrigger,
    *,
    observed_bytes: int | None = None,
    limit_bytes: int | None = None,
) -> None:
    available = outcome is nccl_ras.NcclRasResult.SUCCESS
    current = {
        "outcome": outcome.value,
        "trigger": trigger.value,
        **telemetry.snapshot_attributes("nccl_ras", telemetry.CURRENT_SNAPSHOT),
    }
    telemetry.gauge("ras_available").set(float(available), attributes=current)
    telemetry.histogram("ras_poll_duration_seconds", unit="s").record(duration, attributes=current)
    if not available:
        limit_attributes = {}
        if observed_bytes is not None:
            limit_attributes["observed_bytes"] = str(observed_bytes)
        if limit_bytes is not None:
            limit_attributes["limit_bytes"] = str(limit_bytes)
        telemetry.counter("ras_poll_failures", unit="{failure}").add(
            1,
            attributes={
                "failure_kind": outcome.value,
                "trigger": trigger.value,
                **limit_attributes,
                **telemetry.snapshot_attributes("nccl_ras", telemetry.CUMULATIVE_SNAPSHOT),
            },
        )
