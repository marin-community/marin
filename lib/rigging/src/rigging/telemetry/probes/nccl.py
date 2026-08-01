# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect the supported NCCL RAS communicator and rank fields."""

import json
import logging
import math
from collections.abc import Mapping
from typing import NamedTuple

from rigging import telemetry
from rigging.telemetry.probes.runner import BoundedCommandRunner, PeriodicProbe

TIMEOUT = 8.0
_MAX_COMMUNICATORS = 256
_MAX_RANKS_PER_COMMUNICATOR = 2_048
_MAX_COLLECTIVES_PER_RANK = 64
_MAX_METRICS = 4_096
_MAX_FIELD_BYTES = 256
_MAX_EXACT_COUNT = 2**53 - 1

logger = logging.getLogger(__name__)


def start() -> PeriodicProbe:
    """Collect NCCL RAS communicator evidence until shutdown."""
    return PeriodicProbe("nccl_ras", collect)


class _Metric(NamedTuple):
    name: str
    value: float
    unit: str
    attributes: dict[str, str]


class _ReportedRank(NamedTuple):
    rank: int
    init_state: int
    async_error: int
    finalized: bool
    destroyed: bool
    aborted: bool
    collectives: dict[str, int]


class _MissingRank(NamedTuple):
    rank: int
    unresponsive: bool
    considered_dead: bool


def collect(runner: BoundedCommandRunner) -> None:
    client_timeout = max(1, math.ceil(TIMEOUT * 0.75))
    result = runner.run(("ncclras", "-v", "-t", str(client_timeout), "-f", "json"), TIMEOUT)
    if result is None or result.returncode != 0:
        return
    try:
        metrics = _metrics(json.loads(result.stdout))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        logger.warning("could not parse NCCL RAS telemetry: %s", error)
        return
    for metric in metrics:
        telemetry.gauge(metric.name, unit=metric.unit).set(metric.value, attributes=metric.attributes)


def _metrics(payload: object) -> list[_Metric]:
    root = _mapping(payload, "root")
    communicators = _list(root, "communicators")
    communicator_count = _count(root, "communicators_count")
    if communicator_count != len(communicators) or communicator_count > _MAX_COMMUNICATORS:
        raise ValueError("invalid NCCL communicator count")

    current = telemetry.snapshot_attributes("nccl_ras", telemetry.CURRENT_SNAPSHOT)
    ras = _mapping(root.get("ras"), "ras")
    metrics: list[_Metric] = [
        _Metric("communicators", float(communicator_count), "{communicator}", current),
        _Metric(
            "runtime_inventory",
            1.0,
            "",
            {
                "runtime_kind": "nccl",
                "nccl_version": _text(root, "nccl_version"),
                "cuda_runtime_version": _version(root, "cuda_runtime_version"),
                "cuda_driver_version": _version(root, "cuda_driver_version"),
                **current,
            },
        ),
        _Metric(
            "ras_collection_duration_seconds",
            _nonnegative_number(ras, "collection_time_sec"),
            "s",
            current,
        ),
        _Metric("ras_collection_timeouts", float(_count(ras, "timeouts_count")), "{timeout}", current),
    ]
    for value in communicators:
        metrics.extend(_communicator_metrics(_mapping(value, "communicator"), current))
        if len(metrics) > _MAX_METRICS:
            raise ValueError("NCCL RAS metric limit exceeded")
    return metrics


def _communicator_metrics(value: dict[str, object], current: dict[str, str]) -> list[_Metric]:
    size = _count(value, "size")
    ranks = _list(value, "ranks")
    missing_ranks = _list(value, "missing_ranks")
    if size > _MAX_RANKS_PER_COMMUNICATOR:
        raise ValueError("too many NCCL ranks")
    if _count(value, "ranks_count") != len(ranks) or _count(value, "missing_ranks_count") != len(missing_ranks):
        raise ValueError("NCCL rank counts do not match rank lists")
    if size != len(ranks) + len(missing_ranks):
        raise ValueError("NCCL communicator size does not match rank counts")

    reported = [_reported_rank(_mapping(rank, "rank")) for rank in ranks]
    missing = [_missing_rank(_mapping(rank, "missing rank")) for rank in missing_ranks]
    rank_numbers = [rank.rank for rank in reported] + [rank.rank for rank in missing]
    if len(rank_numbers) != len(set(rank_numbers)):
        raise ValueError("duplicate NCCL communicator rank")

    identity = {
        "communicator_hash": _text(value, "hash"),
        "secondary_hash": _text(value, "secondary_hash"),
    }
    metrics: list[_Metric] = [
        _Metric(
            "communicator_state",
            1.0,
            "",
            {
                **identity,
                "lifecycle_state": _lifecycle_state(reported, missing),
                "collective_mismatch": str(_collective_mismatch(reported)).lower(),
                **current,
            },
        )
    ]
    rank_counts = {
        "total": size,
        "reported": len(reported),
        "missing": len(missing),
        "unresponsive": sum(rank.unresponsive for rank in missing),
        "considered_dead": sum(rank.considered_dead for rank in missing),
    }
    metrics.extend(
        _Metric("communicator_ranks", float(count), "{rank}", {**identity, "rank_state": state, **current})
        for state, count in rank_counts.items()
    )
    cumulative = telemetry.snapshot_attributes("nccl_ras", telemetry.CUMULATIVE_SNAPSHOT)
    for rank in reported:
        rank_identity = {**identity, "rank": str(rank.rank)}
        metrics.append(
            _Metric(
                "communicator_rank_status",
                1.0,
                "",
                {
                    **rank_identity,
                    "rank_state": "reported",
                    "init_state": str(rank.init_state),
                    "async_error": str(rank.async_error),
                    "finalize_called": str(rank.finalized).lower(),
                    "destroy_flag": str(rank.destroyed).lower(),
                    "abort_flag": str(rank.aborted).lower(),
                    **current,
                },
            )
        )
        metrics.extend(
            _Metric(
                "collective_operations",
                float(count),
                "{operation}",
                {**rank_identity, "collective": name, **cumulative},
            )
            for name, count in rank.collectives.items()
        )
    metrics.extend(
        _Metric(
            "communicator_rank_status",
            0.0,
            "",
            {
                **identity,
                "rank": str(rank.rank),
                "rank_state": "missing",
                "unresponsive": str(rank.unresponsive).lower(),
                "considered_dead": str(rank.considered_dead).lower(),
                **current,
            },
        )
        for rank in missing
    )
    return metrics


def _reported_rank(value: dict[str, object]) -> _ReportedRank:
    status = _mapping(value.get("status"), "rank status")
    counts = _mapping(value.get("collective_counts"), "collective counts")
    if len(counts) > _MAX_COLLECTIVES_PER_RANK:
        raise ValueError("too many NCCL collective kinds")
    collectives = {_bounded_text(name): _nonnegative_integer(count) for name, count in counts.items()}
    return _ReportedRank(
        _count(value, "rank"),
        _count(status, "init_state"),
        _count(status, "async_error"),
        _boolean(status, "finalize_called"),
        _boolean(status, "destroy_flag"),
        _boolean(status, "abort_flag"),
        collectives,
    )


def _missing_rank(value: dict[str, object]) -> _MissingRank:
    status = _mapping(value.get("status"), "missing rank status")
    return _MissingRank(
        _count(value, "rank"),
        _boolean(status, "unresponsive"),
        _boolean(status, "considered_dead"),
    )


def _lifecycle_state(reported: list[_ReportedRank], missing: list[_MissingRank]) -> str:
    if any(rank.aborted or rank.async_error for rank in reported):
        return "error"
    if missing:
        return "incomplete"
    if any(rank.finalized or rank.destroyed for rank in reported):
        return "finalizing"
    if any(rank.init_state == 7 for rank in reported):
        return "initializing"
    if reported and all(rank.init_state == 0 for rank in reported):
        return "running"
    return "error"


def _collective_mismatch(reported: list[_ReportedRank]) -> bool:
    if len(reported) < 2:
        return False
    collectives = set().union(*(rank.collectives for rank in reported))
    return any(len({rank.collectives.get(collective, 0) for rank in reported}) > 1 for collective in collectives)


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"NCCL RAS {context} must be an object")
    return dict(value)


def _list(value: dict[str, object], field: str) -> list[object]:
    child = value.get(field)
    if not isinstance(child, list):
        raise ValueError(f"NCCL RAS {field} must be a list")
    return child


def _text(value: dict[str, object], field: str) -> str:
    child = value.get(field)
    if not isinstance(child, str):
        raise ValueError(f"NCCL RAS {field} must be a string")
    return _bounded_text(child)


def _version(value: dict[str, object], field: str) -> str:
    child = value.get(field)
    if isinstance(child, bool) or not isinstance(child, str | int):
        raise ValueError(f"NCCL RAS {field} must be a string or integer")
    return _bounded_text(str(child))


def _count(value: dict[str, object], field: str) -> int:
    if field not in value:
        raise ValueError(f"NCCL RAS {field} is required")
    return _nonnegative_integer(value[field])


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_EXACT_COUNT:
        raise ValueError("NCCL RAS count must be a bounded nonnegative integer")
    return value


def _nonnegative_number(value: dict[str, object], field: str) -> float:
    child = value.get(field)
    if isinstance(child, bool) or not isinstance(child, int | float):
        raise ValueError(f"NCCL RAS {field} must be numeric")
    number = float(child)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"NCCL RAS {field} must be finite and nonnegative")
    return number


def _boolean(value: dict[str, object], field: str) -> bool:
    child = value.get(field)
    if not isinstance(child, bool):
        raise ValueError(f"NCCL RAS {field} must be boolean")
    return child


def _bounded_text(value: str) -> str:
    if not value or len(value.encode()) > _MAX_FIELD_BYTES:
        raise ValueError("NCCL RAS field exceeds string limit")
    return value
