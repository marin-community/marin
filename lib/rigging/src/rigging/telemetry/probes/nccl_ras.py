# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed parsing and bounded reduction for NCCL RAS verbose status responses."""

import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

MAX_COMMUNICATORS = 256
MAX_FIELD_BYTES = 256
MAX_EXACT_COUNT = 2**53 - 1
_MAX_FAILURE_MESSAGE_CHARS = 1_024
_COLLECTIVE_OUTLIER_PREFIX = "collective_outlier:"
_RUNNING_INIT_STATE = 0
_INITIALIZING_INIT_STATE = 7


def _bounded_text(value: str) -> str:
    if not value or len(value.encode()) > MAX_FIELD_BYTES:
        raise ValueError("field exceeds string limit")
    return value


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode()
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode(errors="ignore")


BoundedText = Annotated[str, AfterValidator(_bounded_text)]
Count = Annotated[int, Field(strict=True, ge=0, le=MAX_EXACT_COUNT)]


class RasDetail(StrEnum):
    """Amount of diagnostic evidence retained from one RAS response."""

    PERIODIC = "periodic"
    STALL = "stall"


class NcclRasResult(StrEnum):
    """Terminal outcome of one NCCL RAS collection attempt."""

    SUCCESS = "success"
    CLIENT_TIMEOUT = "client_timeout"
    UNAVAILABLE = "unavailable"
    INVALID_CLIENT_CONFIG = "invalid_client_config"
    CLIENT_RESPONSE_LIMIT = "client_response_limit"
    INVALID_PAYLOAD = "invalid_payload"
    NONZERO_EXIT = "nonzero_exit"
    INVALID_CLIENT_OUTPUT = "invalid_client_output"
    RUNNER_OUTPUT_LIMIT = "runner_output_limit"
    CANCELLED = "cancelled"
    START_FAILED = "start_failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    OUTPUT_FAILED = "output_failed"
    INVALID_TIMEOUT = "invalid_timeout"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _ReportedStatus(_Model):
    init_state: Count
    async_error: Count
    finalize_called: bool
    destroy_flag: bool
    abort_flag: bool


class _MissingStatus(_Model):
    unresponsive: bool
    considered_dead: bool


class _ReportedRank(_Model):
    rank: Count
    host: BoundedText
    pid: Count
    cuda_dev: Count
    nvml_dev: Count
    status: _ReportedStatus
    collective_counts: dict[BoundedText, Count]


class _MissingRank(_Model):
    rank: Count
    host: BoundedText
    pid: Count
    cuda_dev: Count
    nvml_dev: Count
    status: _MissingStatus


class _Communicator(_Model):
    hash: BoundedText
    secondary_hash: BoundedText
    size: Count
    ranks_count: Count
    missing_ranks_count: Count
    ranks: tuple[_ReportedRank, ...]
    missing_ranks: tuple[_MissingRank, ...]

    @model_validator(mode="after")
    def _validate_ranks(self) -> "_Communicator":
        if self.ranks_count != len(self.ranks) or self.missing_ranks_count != len(self.missing_ranks):
            raise ValueError("rank counts do not match rank lists")
        if self.size != len(self.ranks) + len(self.missing_ranks):
            raise ValueError("communicator size does not match rank counts")
        ranks = [rank.rank for rank in self.ranks] + [rank.rank for rank in self.missing_ranks]
        if len(ranks) != len(set(ranks)):
            raise ValueError("duplicate communicator rank")
        return self


class _RasMetadata(_Model):
    collection_time_sec: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    timeouts_count: Count


class _Envelope(_Model):
    nccl_version: BoundedText
    cuda_runtime_version: BoundedText
    cuda_driver_version: BoundedText
    communicators_count: Count
    communicators: tuple[dict[str, Any], ...]
    ras: _RasMetadata

    @field_validator("cuda_runtime_version", "cuda_driver_version", mode="before")
    @classmethod
    def _normalize_version(cls, value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, str | int):
            raise ValueError("version must be a string or integer")
        return str(value)

    @model_validator(mode="after")
    def _validate_communicator_count(self) -> "_Envelope":
        if self.communicators_count != len(self.communicators):
            raise ValueError("communicator count does not match communicator list")
        return self


class CollectiveProgress(_Model):
    """Range of one collective counter across the reported communicator ranks."""

    communicator_hash: BoundedText
    secondary_hash: BoundedText
    collective: BoundedText
    minimum: Count
    maximum: Count


class RankObservation(_Model):
    """One rank retained because it is missing, unhealthy, or a progress outlier."""

    communicator_hash: BoundedText
    secondary_hash: BoundedText
    rank: Count
    host: BoundedText
    pid: Count
    cuda_device: Count
    nvml_device: Count
    rank_state: Literal["reported", "missing"]
    reasons: tuple[BoundedText, ...]
    init_state: Count = 0
    async_error: Count = 0
    finalize_called: bool = False
    destroy_flag: bool = False
    abort_flag: bool = False
    unresponsive: bool = False
    considered_dead: bool = False


class CommunicatorSummary(_Model):
    """Health and rank-count summary for one validated communicator."""

    communicator_hash: BoundedText
    secondary_hash: BoundedText
    size: Count
    reported_ranks: Count
    missing_ranks: Count
    unresponsive_ranks: Count
    considered_dead_ranks: Count
    lifecycle_state: Literal["running", "initializing", "finalizing", "incomplete", "error"]
    collective_mismatch: bool


@dataclass(frozen=True)
class _CommunicatorReduction:
    summary: CommunicatorSummary
    progress: list[CollectiveProgress]
    observations: list[RankObservation]


class NcclRasReport(_Model):
    """Compact, bounded representation of one NCCL RAS response."""

    detail: RasDetail
    nccl_version: BoundedText
    cuda_runtime_version: BoundedText
    cuda_driver_version: BoundedText
    collection_time_seconds: float
    ras_timeouts: Count
    input_communicators: Count
    emitted_communicators: Count
    invalid_communicators: Count
    omitted_communicators: Count
    communicators: tuple[CommunicatorSummary, ...]
    progress: tuple[CollectiveProgress, ...]
    rank_observations: tuple[RankObservation, ...]


class NcclRasClientOutput(_Model):
    """Typed wire result emitted by the NCCL RAS client subprocess."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: NcclRasResult
    report: NcclRasReport | None = None
    message: str | None = Field(default=None, max_length=_MAX_FAILURE_MESSAGE_CHARS)
    observed_bytes: Count | None = None
    limit_bytes: Count | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if self.result is NcclRasResult.SUCCESS:
            if self.report is None:
                raise ValueError("successful NCCL RAS result requires a report")
            if self.message is not None or self.observed_bytes is not None or self.limit_bytes is not None:
                raise ValueError("successful NCCL RAS result must not contain failure context")
        elif self.report is not None or self.message is None:
            raise ValueError("failed NCCL RAS result requires a message and must not contain a report")
        return self

    @classmethod
    def success(cls, report: NcclRasReport) -> Self:
        """Return a successful result for ``report``."""
        return cls(result=NcclRasResult.SUCCESS, report=report)

    @classmethod
    def failure(
        cls,
        result: NcclRasResult,
        message: str,
        *,
        observed_bytes: int | None = None,
        limit_bytes: int | None = None,
    ) -> Self:
        """Return a failed result with bounded diagnostic context."""
        if result is NcclRasResult.SUCCESS:
            raise ValueError("failure result must not be success")
        return cls(
            result=result,
            message=message[:_MAX_FAILURE_MESSAGE_CHARS],
            observed_bytes=observed_bytes,
            limit_bytes=limit_bytes,
        )

    def to_string(self) -> str:
        """Serialize this result as JSON text."""
        return self.model_dump_json()

    def to_bytes(self) -> bytes:
        """Serialize this result for a binary subprocess stream."""
        return self.to_string().encode()

    @classmethod
    def from_bytes(cls, payload: bytes) -> Self:
        """Validate a result returned by the client subprocess."""
        return cls.model_validate_json(payload)


def reduce_response(response: bytes, *, detail: RasDetail) -> NcclRasReport:
    """Parse one JSON status response and retain only bounded diagnostic evidence."""
    object_start = response.find(b"{")
    if object_start < 0:
        raise ValueError("NCCL RAS response did not contain JSON")
    payload = json.loads(response[object_start:])
    envelope = _Envelope.model_validate(payload)

    summaries: list[CommunicatorSummary] = []
    progress: list[CollectiveProgress] = []
    observations: list[RankObservation] = []
    invalid_communicators = 0
    selected = envelope.communicators[:MAX_COMMUNICATORS]
    for value in selected:
        try:
            communicator = _Communicator.model_validate(value)
        except ValidationError:
            invalid_communicators += 1
            continue
        reduced = _reduce_communicator(communicator, detail)
        summaries.append(reduced.summary)
        progress.extend(reduced.progress)
        observations.extend(reduced.observations)

    summaries.sort(key=lambda summary: (summary.communicator_hash, summary.secondary_hash))
    progress.sort(key=lambda item: (item.communicator_hash, item.secondary_hash, item.collective))
    observations.sort(key=lambda item: (item.communicator_hash, item.secondary_hash, item.rank))
    return NcclRasReport(
        detail=detail,
        nccl_version=envelope.nccl_version,
        cuda_runtime_version=str(envelope.cuda_runtime_version),
        cuda_driver_version=str(envelope.cuda_driver_version),
        collection_time_seconds=envelope.ras.collection_time_sec,
        ras_timeouts=envelope.ras.timeouts_count,
        input_communicators=envelope.communicators_count,
        emitted_communicators=len(summaries),
        invalid_communicators=invalid_communicators,
        omitted_communicators=max(0, len(envelope.communicators) - len(selected)),
        communicators=tuple(summaries),
        progress=tuple(progress),
        rank_observations=tuple(observations),
    )


def _reduce_communicator(
    communicator: _Communicator,
    detail: RasDetail,
) -> _CommunicatorReduction:
    collective_names = sorted(set().union(*(rank.collective_counts for rank in communicator.ranks)))
    progress = [
        CollectiveProgress(
            communicator_hash=communicator.hash,
            secondary_hash=communicator.secondary_hash,
            collective=name,
            minimum=min((rank.collective_counts.get(name, 0) for rank in communicator.ranks), default=0),
            maximum=max((rank.collective_counts.get(name, 0) for rank in communicator.ranks), default=0),
        )
        for name in collective_names
    ]
    mismatch = any(item.minimum != item.maximum for item in progress)
    outliers = _progress_outliers(communicator) if detail is RasDetail.STALL else {}
    observations = [_missing_observation(communicator, rank) for rank in communicator.missing_ranks]
    for rank in communicator.ranks:
        reasons = []
        if rank.status.async_error:
            reasons.append("async_error")
        if rank.status.abort_flag:
            reasons.append("aborted")
        if rank.status.init_state not in {_RUNNING_INIT_STATE, _INITIALIZING_INIT_STATE}:
            reasons.append("unexpected_init_state")
        reasons.extend(outliers.get(rank.rank, ()))
        if reasons:
            observations.append(_reported_observation(communicator, rank, tuple(reasons)))

    return _CommunicatorReduction(
        summary=CommunicatorSummary(
            communicator_hash=communicator.hash,
            secondary_hash=communicator.secondary_hash,
            size=communicator.size,
            reported_ranks=len(communicator.ranks),
            missing_ranks=len(communicator.missing_ranks),
            unresponsive_ranks=sum(rank.status.unresponsive for rank in communicator.missing_ranks),
            considered_dead_ranks=sum(rank.status.considered_dead for rank in communicator.missing_ranks),
            lifecycle_state=_lifecycle_state(communicator),
            collective_mismatch=mismatch,
        ),
        progress=progress if detail is RasDetail.STALL else [item for item in progress if item.minimum != item.maximum],
        observations=observations,
    )


def _progress_outliers(communicator: _Communicator) -> dict[int, tuple[str, ...]]:
    reasons: dict[int, list[str]] = {}
    collectives = sorted(set().union(*(rank.collective_counts for rank in communicator.ranks)))
    for collective in collectives:
        counts = Counter(rank.collective_counts.get(collective, 0) for rank in communicator.ranks)
        most_common = counts.most_common(2)
        if len(most_common) < 2 or most_common[0][1] == most_common[1][1]:
            continue
        expected = most_common[0][0]
        for rank in communicator.ranks:
            if rank.collective_counts.get(collective, 0) != expected:
                collective_budget = MAX_FIELD_BYTES - len(_COLLECTIVE_OUTLIER_PREFIX.encode())
                bounded_collective = _truncate_utf8(collective, collective_budget)
                reasons.setdefault(rank.rank, []).append(f"{_COLLECTIVE_OUTLIER_PREFIX}{bounded_collective}")
    return {rank: tuple(rank_reasons) for rank, rank_reasons in reasons.items()}


def _lifecycle_state(
    communicator: _Communicator,
) -> Literal["running", "initializing", "finalizing", "incomplete", "error"]:
    if any(rank.status.abort_flag or rank.status.async_error for rank in communicator.ranks):
        return "error"
    if communicator.missing_ranks:
        return "incomplete"
    if any(rank.status.finalize_called or rank.status.destroy_flag for rank in communicator.ranks):
        return "finalizing"
    if any(rank.status.init_state == _INITIALIZING_INIT_STATE for rank in communicator.ranks):
        return "initializing"
    if communicator.ranks and all(rank.status.init_state == _RUNNING_INIT_STATE for rank in communicator.ranks):
        return "running"
    return "error"


def _reported_observation(
    communicator: _Communicator,
    rank: _ReportedRank,
    reasons: tuple[str, ...],
) -> RankObservation:
    return RankObservation(
        communicator_hash=communicator.hash,
        secondary_hash=communicator.secondary_hash,
        rank=rank.rank,
        host=rank.host,
        pid=rank.pid,
        cuda_device=rank.cuda_dev,
        nvml_device=rank.nvml_dev,
        rank_state="reported",
        reasons=reasons,
        init_state=rank.status.init_state,
        async_error=rank.status.async_error,
        finalize_called=rank.status.finalize_called,
        destroy_flag=rank.status.destroy_flag,
        abort_flag=rank.status.abort_flag,
    )


def _missing_observation(communicator: _Communicator, rank: _MissingRank) -> RankObservation:
    reasons = ["missing"]
    if rank.status.unresponsive:
        reasons.append("unresponsive")
    if rank.status.considered_dead:
        reasons.append("considered_dead")
    return RankObservation(
        communicator_hash=communicator.hash,
        secondary_hash=communicator.secondary_hash,
        rank=rank.rank,
        host=rank.host,
        pid=rank.pid,
        cuda_device=rank.cuda_dev,
        nvml_device=rank.nvml_dev,
        rank_state="missing",
        reasons=tuple(reasons),
        unresponsive=rank.status.unresponsive,
        considered_dead=rank.status.considered_dead,
    )
