# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe-only NCCL RAS summary adapter."""

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from rigging import telemetry
from rigging.probe_types import (
    MAX_FIELD_BYTES,
    MAX_RESULT_EVENTS,
    MAX_RESULT_METRICS,
    MAX_SUBPROCESS_OUTPUT_BYTES,
    CommandRunner,
    MetricKind,
    ProbeCollection,
    ProbeEvent,
    ProbeMetric,
    ProbeOutcome,
    ProbeReason,
    command_failure_collection,
)

_MAX_JSON_NODES = 20_000
_MAX_JSON_DEPTH = 32
_MAX_COMMUNICATORS = 256
_MAX_OPERATIONS_PER_COMMUNICATOR = 64
_MAX_TEXT_LINES = 4_096
_MAX_TEXT_LINE_BYTES = 1_024
_HEALTHY_STATES = frozenset({"active", "complete", "healthy", "operational", "ready", "running"})
_NO_ASYNC_ERROR = frozenset({"", "0", "ncclsuccess", "none", "success"})
_LEGACY_OPERATIONS = frozenset(
    {
        "AllGather",
        "AllReduce",
        "AllToAll",
        "Broadcast",
        "Gather",
        "Recv",
        "Reduce",
        "ReduceScatter",
        "Scatter",
        "Send",
    }
)
_VERSION_PATTERN = re.compile(r"NCCL RAS\s*\(v(?P<version>[^)]+)\)", re.IGNORECASE)
_RUNTIME_VERSION_PATTERN = re.compile(r"CUDA runtime version\s*:\s*(?P<version>\S+)", re.IGNORECASE)
_DRIVER_VERSION_PATTERN = re.compile(r"CUDA driver version\s*:\s*(?P<version>\S+)", re.IGNORECASE)
_COMMUNICATOR_PATTERN = re.compile(
    r"^Communicator\b(?:\s+(?:hash|id))?\s*[:=]?\s*(?P<hash>0x[0-9a-f]+|[0-9a-f]{8,})"
    r"(?:.*?secondary(?:_hash|\s+hash)\s*[:=]\s*(?P<secondary>\S+))?",
    re.IGNORECASE,
)
_OPERATION_PATTERN = re.compile(r"^(?P<operation>[A-Za-z][A-Za-z0-9]+)\s*[:=]\s*(?P<count>\d+)\s*$")


@dataclass(frozen=True)
class NcclRasProbe:
    """Configured NCCL RAS summary probe."""

    interval: float
    timeout: float
    name: str = "nccl_ras"

    def collect(self, runner: CommandRunner) -> ProbeCollection:
        client_timeout = max(1, math.floor(self.timeout * 0.75))
        base_command = ("ncclras", "-v", "-t", str(client_timeout))
        json_result = runner.run(
            (*base_command, "-f", "json"),
            timeout=self.timeout,
            max_output_bytes=MAX_SUBPROCESS_OUTPUT_BYTES,
        )
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
            collection = _collection_from_json_or_text(json_result.stdout)
            if collection is not None:
                return collection

        text_result = runner.run(
            base_command,
            timeout=self.timeout,
            max_output_bytes=MAX_SUBPROCESS_OUTPUT_BYTES,
        )
        terminal = command_failure_collection(text_result.status, timeout_metrics=(timeout_metric,))
        if terminal is not None:
            return terminal
        if text_result.returncode != 0:
            return ProbeCollection(ProbeOutcome.UNAVAILABLE, ProbeReason.NONZERO_EXIT)
        try:
            return _summary_collection(_parse_text(text_result.stdout))
        except (UnicodeDecodeError, ValueError):
            return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.PARSE_ERROR)


@dataclass
class _CommunicatorSummary:
    communicator_hash: str = ""
    secondary_hash: str = ""
    state: str = ""
    initialization: str = ""
    async_error: str = ""
    mismatch: bool = False
    ranks: dict[str, int] = field(default_factory=dict)
    operations: dict[str, int] = field(default_factory=dict)


@dataclass
class _RasSummary:
    nccl_version: str = ""
    cuda_runtime_version: str = ""
    cuda_driver_version: str = ""
    communicators: list[_CommunicatorSummary] = field(default_factory=list)


def _collection_from_json_or_text(output: bytes) -> ProbeCollection | None:
    try:
        payload = json.loads(output)
        return _summary_collection(_parse_json(payload))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        try:
            return _summary_collection(_parse_text(output))
        except (UnicodeDecodeError, ValueError):
            return None


def _parse_json(payload: Any) -> _RasSummary:
    _validate_json_tree(payload)
    if not isinstance(payload, dict):
        raise ValueError("NCCL RAS JSON root must be an object")

    summary = _RasSummary(
        nccl_version=_version_value(payload, "nccl", "nccl_version", "ncclversion"),
        cuda_runtime_version=_version_value(
            payload,
            "cuda_runtime",
            "cuda_runtime_version",
            "cudaruntimeversion",
        ),
        cuda_driver_version=_version_value(
            payload,
            "cuda_driver",
            "cuda_driver_version",
            "cudadriverversion",
        ),
    )
    entries = _communicator_entries(payload)
    if len(entries) > _MAX_COMMUNICATORS:
        raise ValueError("too many NCCL communicators")
    summary.communicators = [_json_communicator(entry) for entry in entries]
    return summary


def _json_communicator(value: dict[str, Any]) -> _CommunicatorSummary:
    ranks_value = _mapping_value(value, "ranks", "rank_summary", "ranksummary")
    ranks = ranks_value if isinstance(ranks_value, dict) else value
    rank_list = ranks_value if isinstance(ranks_value, list) else []
    operations_value = _mapping_value(
        value,
        "operations",
        "collectives",
        "collective_operations",
        "operation_counts",
        "coll_ops",
    )
    operations = _operation_counts(operations_value)
    if len(operations) > _MAX_OPERATIONS_PER_COMMUNICATOR:
        raise ValueError("too many NCCL operation kinds")

    total = _integer_value(ranks, "total", "total_ranks", "rank_count", "nranks")
    if total is None and isinstance(ranks_value, int):
        total = _nonnegative_integer(ranks_value)
    if total is None and rank_list:
        total = len(rank_list)
    rank_counts = {
        "total": total,
        "missing": _integer_value(ranks, "missing", "missing_ranks"),
        "unresponsive": _integer_value(ranks, "unresponsive", "unresponsive_ranks"),
        "considered_dead": _integer_value(ranks, "considered_dead", "dead", "dead_ranks"),
    }
    if rank_list:
        states = [_string_value(rank, "state", "status").lower() for rank in rank_list if isinstance(rank, dict)]
        rank_counts["missing"] = rank_counts["missing"] or states.count("missing")
        rank_counts["unresponsive"] = rank_counts["unresponsive"] or states.count("unresponsive")
        rank_counts["considered_dead"] = rank_counts["considered_dead"] or sum(
            state in {"considered_dead", "dead"} for state in states
        )

    return _CommunicatorSummary(
        communicator_hash=_bounded_string(_string_value(value, "comm_hash", "communicator_hash", "hash")),
        secondary_hash=_bounded_string(_string_value(value, "secondary_hash", "secondary_comm_hash", "comm_hash2")),
        state=_bounded_string(_string_value(value, "state", "lifecycle_state", "status")),
        initialization=_bounded_string(_string_value(value, "initialization", "init_state", "initialization_state")),
        async_error=_bounded_string(_string_value(value, "async_error", "asyncerror")),
        mismatch=_boolean_value(value, "mismatch", "collective_mismatch", "collectivemismatch"),
        ranks={name: count for name, count in rank_counts.items() if count is not None},
        operations=operations,
    )


def _operation_counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {_bounded_string(str(name)): _nonnegative_integer(count) for name, count in value.items()}
    if isinstance(value, list):
        operations: dict[str, int] = {}
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("NCCL operation entry must be an object")
            name = _bounded_string(_string_value(item, "name", "operation", "collective"))
            count = _integer_value(item, "count", "operations", "operation_count")
            if not name or count is None:
                raise ValueError("NCCL operation entry is incomplete")
            operations[name] = count
        return operations
    if value is None:
        return {}
    raise ValueError("NCCL operations must be an object or list")


def _parse_text(output: bytes) -> _RasSummary:
    """Parse pre-2.28.7 output; remove after older deployed NCCL versions retire."""
    text = output.decode("utf-8")
    lines = text.splitlines()
    if len(lines) > _MAX_TEXT_LINES or any(len(line.encode()) > _MAX_TEXT_LINE_BYTES for line in lines):
        raise ValueError("legacy NCCL RAS output exceeds parser limits")
    summary = _RasSummary()
    current: _CommunicatorSummary | None = None
    in_operations = False
    recognized = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if match := _VERSION_PATTERN.search(line):
            summary.nccl_version = _bounded_string(match.group("version"))
            recognized = True
            continue
        if match := _RUNTIME_VERSION_PATTERN.search(line):
            summary.cuda_runtime_version = _bounded_string(match.group("version"))
            recognized = True
            continue
        if match := _DRIVER_VERSION_PATTERN.search(line):
            summary.cuda_driver_version = _bounded_string(match.group("version"))
            recognized = True
            continue
        if match := _COMMUNICATOR_PATTERN.match(line):
            if len(summary.communicators) >= _MAX_COMMUNICATORS:
                raise ValueError("too many NCCL communicators")
            current = _CommunicatorSummary(
                communicator_hash=_bounded_string(match.group("hash")),
                secondary_hash=_bounded_string(match.group("secondary") or ""),
                state=_bounded_string(_line_field(line, "state")),
            )
            summary.communicators.append(current)
            in_operations = False
            recognized = True
            continue
        if current is None:
            continue
        if line.lower().startswith("ranks:"):
            for output_name, aliases in {
                "total": ("total",),
                "missing": ("missing",),
                "unresponsive": ("unresponsive",),
                "considered_dead": ("considered_dead", "considered-dead", "dead"),
            }.items():
                value = _line_integer_field(line, *aliases)
                if value is not None:
                    current.ranks[output_name] = value
            recognized = True
            continue
        if line.lower().startswith("initialization:"):
            current.initialization = _bounded_string(line.split(":", 1)[1].strip())
            recognized = True
            continue
        if line.lower().startswith("async error:"):
            current.async_error = _bounded_string(line.split(":", 1)[1].strip())
            recognized = True
            continue
        if line.lower().startswith("mismatch:"):
            current.mismatch = _parse_boolean(line.split(":", 1)[1].strip())
            recognized = True
            continue
        if line.lower() == "operations:":
            in_operations = True
            continue
        if in_operations and (match := _OPERATION_PATTERN.match(line)):
            operation = match.group("operation")
            if operation not in _LEGACY_OPERATIONS:
                raise ValueError("unknown legacy NCCL operation")
            current.operations[operation] = int(match.group("count"))
            if len(current.operations) > _MAX_OPERATIONS_PER_COMMUNICATOR:
                raise ValueError("too many NCCL operation kinds")
            recognized = True

    if not recognized:
        raise ValueError("unrecognized legacy NCCL RAS output")
    return summary


def _summary_collection(summary: _RasSummary) -> ProbeCollection:
    metrics = [
        ProbeMetric(
            "communicators",
            MetricKind.GAUGE,
            float(len(summary.communicators)),
            unit="{communicator}",
            attributes={"source_kind": "nccl_ras"},
        )
    ]
    runtime_attributes = {"runtime_kind": "nccl"}
    for name, value in {
        "nccl_version": summary.nccl_version,
        "cuda_runtime_version": summary.cuda_runtime_version,
        "cuda_driver_version": summary.cuda_driver_version,
    }.items():
        if value:
            runtime_attributes[name] = value
    if len(runtime_attributes) > 1:
        metrics.append(ProbeMetric("runtime_inventory", MetricKind.GAUGE, 1.0, attributes=runtime_attributes))

    events: list[ProbeEvent] = []
    for communicator in summary.communicators:
        identity = {}
        if communicator.communicator_hash:
            identity["communicator_hash"] = communicator.communicator_hash
        if communicator.secondary_hash:
            identity["secondary_hash"] = communicator.secondary_hash
        state_attributes = dict(identity)
        for name, value in {
            "lifecycle_state": communicator.state,
            "initialization_state": communicator.initialization,
            "async_error": communicator.async_error,
        }.items():
            if value:
                state_attributes[name] = value
        metrics.append(ProbeMetric("communicator_state", MetricKind.GAUGE, 1.0, attributes=state_attributes))
        for rank_state, count in communicator.ranks.items():
            metrics.append(
                ProbeMetric(
                    "communicator_ranks",
                    MetricKind.GAUGE,
                    float(count),
                    unit="{rank}",
                    attributes={**identity, "rank_state": rank_state},
                )
            )
        for operation, count in communicator.operations.items():
            metrics.append(
                ProbeMetric(
                    "collective_operations",
                    MetricKind.GAUGE,
                    float(count),
                    unit="{operation}",
                    attributes={
                        **identity,
                        "collective": operation,
                        **telemetry.snapshot_attributes("nccl_ras", telemetry.CUMULATIVE_SNAPSHOT),
                    },
                )
            )
        for anomaly in _anomalies(communicator):
            if len(events) >= MAX_RESULT_EVENTS:
                break
            events.append(
                ProbeEvent(
                    "nccl_ras_anomaly",
                    {
                        "state": communicator.state or "unknown",
                        "missing_ranks": communicator.ranks.get("missing", 0),
                        "unresponsive_ranks": communicator.ranks.get("unresponsive", 0),
                        "considered_dead_ranks": communicator.ranks.get("considered_dead", 0),
                        "mismatch": communicator.mismatch,
                    },
                    attributes={**identity, "anomaly": anomaly},
                )
            )

    if len(metrics) > MAX_RESULT_METRICS:
        return ProbeCollection(ProbeOutcome.FAILED, ProbeReason.RESULT_LIMIT)
    return ProbeCollection.succeeded(metrics=tuple(metrics), events=tuple(events))


def _anomalies(communicator: _CommunicatorSummary) -> list[str]:
    anomalies: list[str] = []
    if communicator.ranks.get("missing", 0) > 0:
        anomalies.append("incomplete_communicator")
    if communicator.ranks.get("unresponsive", 0) > 0:
        anomalies.append("unresponsive_rank")
    if communicator.ranks.get("considered_dead", 0) > 0:
        anomalies.append("dead_peer")
    if communicator.mismatch:
        anomalies.append("collective_mismatch")
    if communicator.state and communicator.state.lower() not in _HEALTHY_STATES:
        anomalies.append("communicator_state")
    if communicator.async_error.lower() not in _NO_ASYNC_ERROR:
        anomalies.append("async_error")
    return anomalies


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


def _communicator_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if _normalized_key(key) in {"communicators", "comms"}:
                    if not isinstance(child, list) or not all(isinstance(item, dict) for item in child):
                        raise ValueError("NCCL communicators must be a list of objects")
                    return child
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    raise ValueError("NCCL RAS JSON has no communicator summary")


def _version_value(payload: dict[str, Any], *aliases: str) -> str:
    aliases_normalized = {_normalized_key(alias) for alias in aliases}
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if _normalized_key(key) in aliases_normalized and isinstance(child, str | int | float):
                    return _bounded_string(str(child))
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return ""


def _mapping_value(value: dict[str, Any], *aliases: str) -> Any:
    aliases_normalized = {_normalized_key(alias) for alias in aliases}
    return next((child for key, child in value.items() if _normalized_key(key) in aliases_normalized), None)


def _string_value(value: dict[str, Any], *aliases: str) -> str:
    child = _mapping_value(value, *aliases)
    if child is None:
        return ""
    if not isinstance(child, str | int | float):
        raise ValueError("NCCL RAS string field has invalid type")
    return str(child)


def _integer_value(value: dict[str, Any], *aliases: str) -> int | None:
    child = _mapping_value(value, *aliases)
    return None if child is None else _nonnegative_integer(child)


def _boolean_value(value: dict[str, Any], *aliases: str) -> bool:
    child = _mapping_value(value, *aliases)
    return False if child is None else _parse_boolean(child)


def _parse_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.lower() in {"false", "no", "none", "0"}:
        return False
    if isinstance(value, str) and value.lower() in {"true", "yes", "1"}:
        return True
    raise ValueError("NCCL RAS boolean field has invalid value")


def _nonnegative_integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("NCCL RAS count must be an integer")
    integer = int(value)
    if integer < 0 or integer != float(value):
        raise ValueError("NCCL RAS count must be a nonnegative integer")
    return integer


def _line_field(line: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*[:=]\s*(\S+)", line, re.IGNORECASE)
    return match.group(1) if match else ""


def _line_integer_field(line: str, *names: str) -> int | None:
    for name in names:
        if match := re.search(rf"\b{re.escape(name)}\s*(?:[:=]\s*|\s+)(\d+)\b", line, re.IGNORECASE):
            return int(match.group(1))
    return None


def _bounded_string(value: str) -> str:
    if len(value.encode()) > MAX_FIELD_BYTES:
        raise ValueError("NCCL RAS field exceeds string limit")
    return value


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())
