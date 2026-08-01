# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe-only adapter for NVIDIA's documented NCCL RAS JSON summary."""

import json
import math
from dataclasses import dataclass
from typing import Any

from rigging import telemetry
from rigging.probe_types import (
    MAX_FIELD_BYTES,
    MAX_RESULT_EVENTS,
    MAX_RESULT_METRICS,
    MAX_SUBPROCESS_OUTPUT_BYTES,
    CommandResult,
    CommandRunner,
    CommandStatus,
    MetricKind,
    ProbeCollection,
    ProbeEvent,
    ProbeMetric,
    ProbeOutcome,
    ProbeReason,
    command_failure_collection,
)
from rigging.timing import Deadline

_MAX_JSON_NODES = 20_000
_MAX_JSON_DEPTH = 32
_MAX_COMMUNICATORS = 256
_MAX_RANKS_PER_COMMUNICATOR = 2_048
_MAX_COLLECTIVES_PER_RANK = 64
_MAX_EXACT_COUNT = 2**53 - 1


@dataclass(frozen=True)
class NcclRasProbe:
    """Configured NCCL RAS summary probe."""

    interval: float
    timeout: float
    name: str = "nccl_ras"

    def collect(self, runner: CommandRunner) -> ProbeCollection:
        deadline = Deadline.from_seconds(self.timeout)
        json_result = _run_query(runner, deadline, json_output=True)
        timeout_metric = ProbeMetric(
            "ras_query_timeouts",
            MetricKind.COUNTER,
            1.0,
            unit="{timeout}",
            attributes={"source_kind": "nccl_ras"},
        )
        terminal = command_failure_collection(json_result.status, timeout_metrics=(timeout_metric,))
        if terminal is not None:
            return terminal

        if json_result.returncode == 0:
            try:
                return _summary_collection(_parse_json_output(json_result.stdout))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.PARSE_ERROR)

        text_result = _run_query(runner, deadline, json_output=False)
        terminal = command_failure_collection(text_result.status, timeout_metrics=(timeout_metric,))
        if terminal is not None:
            return terminal
        if text_result.returncode != 0:
            return ProbeCollection(ProbeOutcome.UNAVAILABLE, ProbeReason.NONZERO_EXIT)
        return ProbeCollection(ProbeOutcome.UNAVAILABLE, ProbeReason.UNSUPPORTED_FORMAT)


@dataclass(frozen=True)
class _RankSummary:
    rank: int
    init_state: int
    async_error: int
    finalize_called: bool
    destroy_flag: bool
    abort_flag: bool
    collective_counts: dict[str, int]


@dataclass(frozen=True)
class _MissingRankSummary:
    rank: int
    unresponsive: bool
    considered_dead: bool


@dataclass(frozen=True)
class _CommunicatorSummary:
    communicator_hash: str
    secondary_hash: str
    size: int
    ranks: tuple[_RankSummary, ...]
    missing_ranks: tuple[_MissingRankSummary, ...]


@dataclass(frozen=True)
class _RasSummary:
    nccl_version: str
    cuda_runtime_version: str
    cuda_driver_version: str
    communicators_count: int
    communicators: tuple[_CommunicatorSummary, ...]
    collection_time: float
    timeouts_count: int


def _run_query(runner: CommandRunner, deadline: Deadline, *, json_output: bool) -> CommandResult:
    remaining = deadline.remaining_seconds()
    if remaining <= 0:
        return CommandResult(CommandStatus.TIMED_OUT)
    client_timeout = max(1, math.ceil(remaining * 0.75))
    command = ("ncclras", "-v", "-t", str(client_timeout))
    if json_output:
        command = (*command, "-f", "json")
    return runner.run(command, timeout=remaining, max_output_bytes=MAX_SUBPROCESS_OUTPUT_BYTES)


def _parse_json_output(output: bytes) -> _RasSummary:
    payload = json.loads(output)
    _validate_json_tree(payload)
    root = _object(payload, "root")
    communicators_count = _count(root, "communicators_count")
    communicator_values = _list(root, "communicators")
    if communicators_count != len(communicator_values):
        raise ValueError("communicators_count does not match communicators")
    if communicators_count > _MAX_COMMUNICATORS:
        raise ValueError("too many NCCL communicators")

    communicators = tuple(_communicator(_object(value, "communicator")) for value in communicator_values)
    ras = _object_field(root, "ras")
    return _RasSummary(
        nccl_version=_text(root, "nccl_version"),
        cuda_runtime_version=_version(root, "cuda_runtime_version"),
        cuda_driver_version=_version(root, "cuda_driver_version"),
        communicators_count=communicators_count,
        communicators=communicators,
        collection_time=_nonnegative_number(ras, "collection_time_sec"),
        timeouts_count=_count(ras, "timeouts_count"),
    )


def _communicator(value: dict[str, Any]) -> _CommunicatorSummary:
    size = _count(value, "size")
    ranks_count = _count(value, "ranks_count")
    missing_ranks_count = _count(value, "missing_ranks_count")
    rank_values = _list(value, "ranks")
    missing_rank_values = _list(value, "missing_ranks")
    if ranks_count != len(rank_values) or missing_ranks_count != len(missing_rank_values):
        raise ValueError("NCCL rank counts do not match rank lists")
    if size != ranks_count + missing_ranks_count:
        raise ValueError("NCCL communicator size does not match rank counts")
    if size > _MAX_RANKS_PER_COMMUNICATOR:
        raise ValueError("too many NCCL communicator ranks")

    ranks = tuple(_rank(_object(item, "rank")) for item in rank_values)
    missing_ranks = tuple(_missing_rank(_object(item, "missing rank")) for item in missing_rank_values)
    rank_numbers = [rank.rank for rank in ranks] + [rank.rank for rank in missing_ranks]
    if len(rank_numbers) != len(set(rank_numbers)):
        raise ValueError("duplicate NCCL communicator rank")
    return _CommunicatorSummary(
        communicator_hash=_text(value, "hash"),
        secondary_hash=_text(value, "secondary_hash"),
        size=size,
        ranks=ranks,
        missing_ranks=missing_ranks,
    )


def _rank(value: dict[str, Any]) -> _RankSummary:
    status = _object_field(value, "status")
    counts = _object_field(value, "collective_counts")
    if len(counts) > _MAX_COLLECTIVES_PER_RANK:
        raise ValueError("too many NCCL collective kinds")
    collective_counts = {_bounded_text(name): _nonnegative_integer(count) for name, count in counts.items()}
    return _RankSummary(
        rank=_count(value, "rank"),
        init_state=_count(status, "init_state"),
        async_error=_count(status, "async_error"),
        finalize_called=_boolean(status, "finalize_called"),
        destroy_flag=_boolean(status, "destroy_flag"),
        abort_flag=_boolean(status, "abort_flag"),
        collective_counts=collective_counts,
    )


def _missing_rank(value: dict[str, Any]) -> _MissingRankSummary:
    status = _object_field(value, "status")
    return _MissingRankSummary(
        rank=_count(value, "rank"),
        unresponsive=_boolean(status, "unresponsive"),
        considered_dead=_boolean(status, "considered_dead"),
    )


def _summary_collection(summary: _RasSummary) -> ProbeCollection:
    current = telemetry.snapshot_attributes("nccl_ras", telemetry.CURRENT_SNAPSHOT)
    metrics = _runtime_metrics(summary, current)
    events: list[ProbeEvent] = []
    for communicator in summary.communicators:
        if len(metrics) + _communicator_metric_count(communicator) > MAX_RESULT_METRICS:
            return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.RESULT_LIMIT)
        metrics.extend(_communicator_metrics(communicator, current))
        remaining_events = MAX_RESULT_EVENTS - len(events)
        events.extend(_communicator_events(communicator)[:remaining_events])
    return ProbeCollection.succeeded(metrics=tuple(metrics), events=tuple(events))


def _runtime_metrics(summary: _RasSummary, current: dict[str, str]) -> list[ProbeMetric]:
    return [
        ProbeMetric(
            "communicators",
            MetricKind.GAUGE,
            float(summary.communicators_count),
            unit="{communicator}",
            attributes=current,
        ),
        ProbeMetric(
            "runtime_inventory",
            MetricKind.GAUGE,
            1.0,
            attributes={
                "runtime_kind": "nccl",
                "nccl_version": summary.nccl_version,
                "cuda_runtime_version": summary.cuda_runtime_version,
                "cuda_driver_version": summary.cuda_driver_version,
                **current,
            },
        ),
        ProbeMetric(
            "ras_collection_duration_seconds",
            MetricKind.GAUGE,
            summary.collection_time,
            unit="s",
            attributes=current,
        ),
        ProbeMetric(
            "ras_collection_timeouts",
            MetricKind.GAUGE,
            float(summary.timeouts_count),
            unit="{timeout}",
            attributes=current,
        ),
    ]


def _communicator_metric_count(communicator: _CommunicatorSummary) -> int:
    rank_metrics = sum(1 + len(rank.collective_counts) for rank in communicator.ranks)
    return 6 + rank_metrics + len(communicator.missing_ranks)


def _communicator_metrics(communicator: _CommunicatorSummary, current: dict[str, str]) -> list[ProbeMetric]:
    identity = _communicator_identity(communicator)
    metrics = [
        ProbeMetric(
            "communicator_state",
            MetricKind.GAUGE,
            1.0,
            attributes={**identity, "lifecycle_state": _lifecycle_state(communicator), **current},
        )
    ]
    rank_counts = {
        "total": communicator.size,
        "reported": len(communicator.ranks),
        "missing": len(communicator.missing_ranks),
        "unresponsive": sum(rank.unresponsive for rank in communicator.missing_ranks),
        "considered_dead": sum(rank.considered_dead for rank in communicator.missing_ranks),
    }
    metrics.extend(
        ProbeMetric(
            "communicator_ranks",
            MetricKind.GAUGE,
            float(count),
            unit="{rank}",
            attributes={**identity, "rank_state": rank_state, **current},
        )
        for rank_state, count in rank_counts.items()
    )
    for rank in communicator.ranks:
        metrics.extend(_reported_rank_metrics(rank, identity, current))
    metrics.extend(_missing_rank_metric(rank, identity, current) for rank in communicator.missing_ranks)
    return metrics


def _reported_rank_metrics(
    rank: _RankSummary,
    identity: dict[str, str],
    current: dict[str, str],
) -> list[ProbeMetric]:
    rank_identity = {**identity, "rank": str(rank.rank)}
    metrics = [
        ProbeMetric(
            "communicator_rank_status",
            MetricKind.GAUGE,
            1.0,
            attributes={
                **rank_identity,
                "rank_state": "reported",
                "init_state": str(rank.init_state),
                "async_error": str(rank.async_error),
                "finalize_called": str(rank.finalize_called).lower(),
                "destroy_flag": str(rank.destroy_flag).lower(),
                "abort_flag": str(rank.abort_flag).lower(),
                **current,
            },
        )
    ]
    cumulative = telemetry.snapshot_attributes("nccl_ras", telemetry.CUMULATIVE_SNAPSHOT)
    metrics.extend(
        ProbeMetric(
            "collective_operations",
            MetricKind.GAUGE,
            float(count),
            unit="{operation}",
            attributes={**rank_identity, "collective": collective, **cumulative},
        )
        for collective, count in rank.collective_counts.items()
    )
    return metrics


def _missing_rank_metric(
    rank: _MissingRankSummary,
    identity: dict[str, str],
    current: dict[str, str],
) -> ProbeMetric:
    return ProbeMetric(
        "communicator_rank_status",
        MetricKind.GAUGE,
        0.0,
        attributes={
            **identity,
            "rank": str(rank.rank),
            "rank_state": "missing",
            "unresponsive": str(rank.unresponsive).lower(),
            "considered_dead": str(rank.considered_dead).lower(),
            **current,
        },
    )


def _communicator_events(communicator: _CommunicatorSummary) -> list[ProbeEvent]:
    identity = _communicator_identity(communicator)
    body = {
        "state": _lifecycle_state(communicator),
        "missing_ranks": len(communicator.missing_ranks),
        "unresponsive_ranks": sum(rank.unresponsive for rank in communicator.missing_ranks),
        "considered_dead_ranks": sum(rank.considered_dead for rank in communicator.missing_ranks),
    }
    return [
        ProbeEvent(
            "nccl_ras_anomaly",
            {**body, "mismatch": anomaly == "collective_mismatch"},
            attributes={**identity, "anomaly": anomaly},
        )
        for anomaly in _anomalies(communicator)
    ]


def _communicator_identity(communicator: _CommunicatorSummary) -> dict[str, str]:
    return {
        "communicator_hash": communicator.communicator_hash,
        "secondary_hash": communicator.secondary_hash,
    }


def _lifecycle_state(communicator: _CommunicatorSummary) -> str:
    if any(rank.abort_flag or rank.async_error != 0 for rank in communicator.ranks):
        return "error"
    if communicator.missing_ranks:
        return "incomplete"
    if any(rank.destroy_flag or rank.finalize_called for rank in communicator.ranks):
        return "finalizing"
    if any(rank.init_state == 7 for rank in communicator.ranks):
        return "initializing"
    if all(rank.init_state == 0 for rank in communicator.ranks):
        return "running"
    return "error"


def _anomalies(communicator: _CommunicatorSummary) -> list[str]:
    anomalies: list[str] = []
    if communicator.missing_ranks:
        anomalies.append("incomplete_communicator")
    if any(rank.unresponsive for rank in communicator.missing_ranks):
        anomalies.append("unresponsive_rank")
    if any(rank.considered_dead for rank in communicator.missing_ranks):
        anomalies.append("dead_peer")
    if _collective_mismatch(communicator.ranks):
        anomalies.append("collective_mismatch")
    if any(rank.abort_flag or rank.async_error != 0 or rank.init_state not in {0, 7} for rank in communicator.ranks):
        anomalies.append("communicator_state")
    return anomalies


def _collective_mismatch(ranks: tuple[_RankSummary, ...]) -> bool:
    if len(ranks) < 2:
        return False
    collectives = set().union(*(rank.collective_counts for rank in ranks))
    return any(len({rank.collective_counts.get(collective, 0) for rank in ranks}) > 1 for collective in collectives)


def _validate_json_tree(value: Any) -> None:
    remaining = _MAX_JSON_NODES
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0 or depth > _MAX_JSON_DEPTH:
            raise ValueError("NCCL RAS JSON exceeds parser limits")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"NCCL RAS {context} must be an object")
    return value


def _object_field(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field not in value:
        raise ValueError(f"NCCL RAS {field} is required")
    return _object(value[field], field)


def _list(value: dict[str, Any], field: str) -> list[Any]:
    child = value.get(field)
    if not isinstance(child, list):
        raise ValueError(f"NCCL RAS {field} must be a list")
    return child


def _text(value: dict[str, Any], field: str) -> str:
    child = value.get(field)
    if not isinstance(child, str):
        raise ValueError(f"NCCL RAS {field} must be a string")
    return _bounded_text(child)


def _version(value: dict[str, Any], field: str) -> str:
    child = value.get(field)
    if isinstance(child, bool) or not isinstance(child, str | int):
        raise ValueError(f"NCCL RAS {field} must be a string or integer")
    return _bounded_text(str(child))


def _count(value: dict[str, Any], field: str) -> int:
    if field not in value:
        raise ValueError(f"NCCL RAS {field} is required")
    return _nonnegative_integer(value[field])


def _nonnegative_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_EXACT_COUNT:
        raise ValueError("NCCL RAS count must be a bounded nonnegative integer")
    return value


def _nonnegative_number(value: dict[str, Any], field: str) -> float:
    child = value.get(field)
    if isinstance(child, bool) or not isinstance(child, int | float):
        raise ValueError(f"NCCL RAS {field} must be numeric")
    number = float(child)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"NCCL RAS {field} must be finite and nonnegative")
    return number


def _boolean(value: dict[str, Any], field: str) -> bool:
    child = value.get(field)
    if not isinstance(child, bool):
        raise ValueError(f"NCCL RAS {field} must be boolean")
    return child


def _bounded_text(value: str) -> str:
    if not value or len(value.encode()) > MAX_FIELD_BYTES:
        raise ValueError("NCCL RAS field exceeds string limit")
    return value
