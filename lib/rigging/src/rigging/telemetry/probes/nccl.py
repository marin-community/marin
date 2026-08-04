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
from rigging.telemetry.probes.runner import MAX_OUTPUT_BYTES, BoundedCommandRunner, CommandStatus, PeriodicProbe

TIMEOUT = 8.0
_DEFAULT_INTERVAL = 10 * 60.0
_MAX_METRICS = 4_096
_SUCCESS_OUTCOME = "success"
_METRIC_RECORDS_NAME = "ras_metric_records"
_CLIENT_COMMAND = (sys.executable, "-m", nccl_client.__name__)
_CLIENT_FAILURES = {
    nccl_client.TIMEOUT_EXIT_CODE: "client_timeout",
    nccl_client.UNAVAILABLE_EXIT_CODE: "unavailable",
    nccl_client.INVALID_CONFIG_EXIT_CODE: "invalid_client_config",
    nccl_client.OUTPUT_LIMIT_EXIT_CODE: "client_response_limit",
    nccl_client.INVALID_PAYLOAD_EXIT_CODE: "invalid_payload",
    nccl_client.REDUCED_OUTPUT_LIMIT_EXIT_CODE: "reduced_output_limit",
}

logger = logging.getLogger(__name__)


class RasTrigger(StrEnum):
    """Reason one RAS collection was requested."""

    PERIODIC = "periodic"
    STALL = "stall"


@dataclass(frozen=True)
class RasCollectionResult:
    """Result of one client subprocess invocation."""

    outcome: str
    duration: float
    report: nccl_ras.NcclRasReport | None = None
    observed_bytes: int | None = None
    limit_bytes: int | None = None


class NcclRasSession:
    """Own periodic collection and provide one serialized stall capture path."""

    def __init__(self, *, interval: float, max_publish_records: int) -> None:
        self._publisher = MetricSnapshotPublisher(max_records=max_publish_records)
        self._periodic = PeriodicProbe("nccl_ras", self._collect_periodic, interval=interval)

    def _collect_periodic(self, runner: BoundedCommandRunner) -> None:
        result = collect_ras(runner, detail=nccl_ras.RasDetail.PERIODIC, timeout=TIMEOUT)
        self._publish(result, RasTrigger.PERIODIC)

    def capture_stall(self, timeout: float) -> RasCollectionResult:
        """Stop periodic work, run one fresh detailed query, and publish its summary."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("stall capture timeout must be positive and finite")
        self._periodic.shutdown(min(2.0, timeout))
        result = collect_ras(BoundedCommandRunner(), detail=nccl_ras.RasDetail.STALL, timeout=timeout)
        self._publish(result, RasTrigger.STALL)
        return result

    def _publish(self, result: RasCollectionResult, trigger: RasTrigger) -> None:
        if result.outcome == CommandStatus.CANCELLED.value:
            return
        _record_poll(
            result.outcome,
            result.duration,
            trigger,
            observed_bytes=result.observed_bytes,
            limit_bytes=result.limit_bytes,
        )
        if result.outcome in {"client_timeout", CommandStatus.DEADLINE_EXCEEDED.value}:
            _record_timeout(trigger)
        if result.report is None:
            return
        snapshots = ras_snapshots(result, trigger=trigger)
        published = self._publisher.publish(snapshots)
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
            float(published.sample_limit_dropped_records),
            attributes={**attributes, "record_state": "sample_limit_dropped"},
        )
        telemetry.gauge(_METRIC_RECORDS_NAME, unit="{record}").set(
            float(published.telemetry_lost_records), attributes={**attributes, "record_state": "telemetry_lost"}
        )

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop periodic collection and reap its active client."""
        self._periodic.shutdown(timeout)


def start(*, interval: float = _DEFAULT_INTERVAL, max_publish_records: int = _MAX_METRICS) -> NcclRasSession:
    """Collect NCCL RAS summaries until shutdown."""
    return NcclRasSession(interval=interval, max_publish_records=max_publish_records)


def collect_ras(
    runner: BoundedCommandRunner,
    *,
    detail: nccl_ras.RasDetail,
    timeout: float,
) -> RasCollectionResult:
    """Run the bounded client and return its validated compact result."""
    client_timeout = max(1, math.ceil(timeout * 0.75))
    started = monotonic()
    command = runner.run_result(
        (*_CLIENT_COMMAND, "--timeout", str(client_timeout), "--detail", detail.value),
        timeout,
    )
    duration = max(0.0, monotonic() - started)
    if command.output is None:
        outcome = "runner_output_limit" if command.status is CommandStatus.OUTPUT_LIMIT else command.status.value
        limit_bytes = MAX_OUTPUT_BYTES if command.status is CommandStatus.OUTPUT_LIMIT else None
        return RasCollectionResult(
            outcome,
            duration,
            observed_bytes=command.observed_output_bytes,
            limit_bytes=limit_bytes,
        )
    if command.output.returncode != 0:
        outcome = _CLIENT_FAILURES.get(command.output.returncode, "nonzero_exit")
        try:
            failure = nccl_ras.parse_client_result(command.output.stdout)
            if isinstance(failure, nccl_ras.NcclRasFailure):
                outcome = failure.failure_kind
                return RasCollectionResult(
                    outcome,
                    duration,
                    observed_bytes=failure.observed_bytes,
                    limit_bytes=failure.limit_bytes,
                )
        except (ValidationError, ValueError):
            pass
        return RasCollectionResult(outcome, duration)

    try:
        client_result = nccl_ras.parse_client_result(command.output.stdout)
    except (ValidationError, ValueError) as error:
        logger.warning("could not parse NCCL RAS client result: %s", error)
        return RasCollectionResult("invalid_client_output", duration)
    if isinstance(client_result, nccl_ras.NcclRasFailure):
        return RasCollectionResult(client_result.failure_kind, duration)
    return RasCollectionResult(_SUCCESS_OUTCOME, duration, client_result.report)


def ras_snapshots(result: RasCollectionResult, *, trigger: RasTrigger) -> tuple[MetricSnapshot, ...]:
    """Translate one compact report into bounded, independently publishable snapshots."""
    report = result.report
    if report is None:
        return ()
    current = {"trigger": trigger.value, "detail": report.detail.value}
    snapshots = [
        _snapshot("communicators", report.input_communicators, "{communicator}", current),
        _snapshot(
            "runtime_inventory",
            1,
            "",
            {
                "runtime_kind": "nccl",
                "nccl_version": report.nccl_version,
                "cuda_runtime_version": report.cuda_runtime_version,
                "cuda_driver_version": report.cuda_driver_version,
                **current,
            },
        ),
        _snapshot("ras_collection_duration_seconds", report.collection_time_seconds, "s", current),
        _snapshot("ras_collection_timeouts", report.ras_timeouts, "{timeout}", current),
    ]
    reduction_counts = {
        "communicators_input": report.input_communicators,
        "communicators_emitted": report.emitted_communicators,
        "communicators_invalid": report.invalid_communicators,
        "communicators_omitted": report.omitted_communicators,
        "progress_input": report.input_progress_summaries,
        "progress_omitted": report.omitted_progress_summaries,
        "rank_observations_input": report.input_rank_observations,
        "rank_observations_omitted": report.omitted_rank_observations,
    }
    snapshots.extend(
        _snapshot("ras_reduction_records", count, "{record}", {**current, "record_state": state})
        for state, count in reduction_counts.items()
    )
    for communicator in report.communicators:
        identity = {
            "communicator_hash": communicator.communicator_hash,
            "secondary_hash": communicator.secondary_hash,
            **current,
        }
        snapshots.append(
            _snapshot(
                "communicator_state",
                1,
                "",
                {
                    **identity,
                    "lifecycle_state": communicator.lifecycle_state,
                    "collective_mismatch": str(communicator.collective_mismatch).lower(),
                },
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
            _snapshot("communicator_ranks", count, "{rank}", {**identity, "rank_state": state})
            for state, count in rank_counts.items()
        )
    for rank in report.rank_observations:
        snapshots.append(
            _snapshot(
                "communicator_rank_status",
                float(rank.rank_state == "reported"),
                "",
                {
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
                    **current,
                },
            )
        )
    for progress in report.progress:
        identity = {
            "communicator_hash": progress.communicator_hash,
            "secondary_hash": progress.secondary_hash,
            "collective": progress.collective,
            **current,
        }
        snapshots.extend(
            (
                _snapshot(
                    "collective_operations",
                    progress.minimum,
                    "{operation}",
                    {**identity, "rank_statistic": "minimum"},
                    temporality=telemetry.CUMULATIVE_SNAPSHOT,
                ),
                _snapshot(
                    "collective_operations",
                    progress.maximum,
                    "{operation}",
                    {**identity, "rank_statistic": "maximum"},
                    temporality=telemetry.CUMULATIVE_SNAPSHOT,
                ),
            )
        )
    return tuple(snapshots)


def _snapshot(
    name: str,
    value: int | float,
    unit: str,
    attributes: dict[str, str],
    *,
    temporality: str = telemetry.CURRENT_SNAPSHOT,
) -> MetricSnapshot:
    return MetricSnapshot(
        name=name,
        value=float(value),
        unit=unit,
        attributes=attributes,
        source_kind="nccl_ras",
        source_temporality=temporality,
    )


def _record_timeout(trigger: RasTrigger) -> None:
    telemetry.counter("ras_poll_timeouts", unit="{timeout}").add(
        1,
        attributes={
            "trigger": trigger.value,
            **telemetry.snapshot_attributes("nccl_ras", telemetry.CUMULATIVE_SNAPSHOT),
        },
    )


def _record_poll(
    outcome: str,
    duration: float,
    trigger: RasTrigger,
    *,
    observed_bytes: int | None = None,
    limit_bytes: int | None = None,
) -> None:
    available = outcome == _SUCCESS_OUTCOME
    current = {
        "outcome": outcome,
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
                "failure_kind": outcome,
                "trigger": trigger.value,
                **limit_attributes,
                **telemetry.snapshot_attributes("nccl_ras", telemetry.CUMULATIVE_SNAPSHOT),
            },
        )
