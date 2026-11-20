"""Transport-neutral teacher scoring and lifecycle ownership."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol, Self

import torch

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import (
    TeacherEvidenceBatch,
    TeacherScoreRequest,
    validate_teacher_evidence,
    validate_teacher_score_request,
)


@dataclass(frozen=True)
class TeacherCapabilities:
    """Static scoring contract advertised by one logical teacher."""

    teacher_id: str
    teacher_revision: str
    tokenizer_fingerprint: str
    evidence_kinds: frozenset[TeacherEvidenceKind]
    max_sequence_length: int
    supports_prompt_token_scoring: bool
    max_concurrency: int

    def __post_init__(self) -> None:
        for field_name in ("teacher_id", "teacher_revision", "tokenizer_fingerprint"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"teacher capabilities {field_name} must be a non-empty string")
        if not self.evidence_kinds:
            raise ValueError("teacher capabilities must advertise at least one evidence kind")
        if self.max_sequence_length <= 0:
            raise ValueError("teacher capabilities max_sequence_length must be positive")
        if self.max_concurrency <= 0:
            raise ValueError("teacher capabilities max_concurrency must be positive")


class TeacherOracle(Protocol):
    """Score exact student-token sequences without prescribing deployment."""

    capabilities: TeacherCapabilities

    async def score(self, request: TeacherScoreRequest) -> TeacherEvidenceBatch: ...

    async def close(self) -> None: ...


class TeacherOracleCollection(Protocol):
    """A logical-teacher collection consumed by the shared coordinator."""

    @property
    def teacher_ids(self) -> frozenset[str]: ...

    async def score(self, teacher_id: str, request: TeacherScoreRequest) -> TeacherEvidenceBatch: ...

    async def close(self) -> None: ...


class TeacherEndpointUnavailable(RuntimeError):
    """A retryable failure isolated to one physical teacher endpoint."""


def validate_teacher_request_against_capabilities(
    request: TeacherScoreRequest, capabilities: TeacherCapabilities
) -> None:
    """Validate a request against an oracle before performing remote work."""
    validate_teacher_score_request(request)
    if capabilities.teacher_id != request.teacher_id:
        raise ValueError("teacher capability identity does not match the score request")
    if capabilities.tokenizer_fingerprint != request.tokenizer_fingerprint:
        raise ValueError("teacher and student tokenizer fingerprints must match for vocabulary-level scoring")
    if request.evidence not in capabilities.evidence_kinds:
        raise ValueError(f"teacher {request.teacher_id!r} cannot provide {request.evidence.value} evidence")
    if not capabilities.supports_prompt_token_scoring:
        raise ValueError(f"teacher {request.teacher_id!r} cannot score prompt tokens")

    sequence_lengths = request.prompt_mask.sum(dim=1) + request.response_mask.sum(dim=1)
    if torch.any(sequence_lengths > capabilities.max_sequence_length):
        raise ValueError(
            f"teacher {request.teacher_id!r} maximum sequence length is {capabilities.max_sequence_length}"
        )


class ValidatedTeacherOracle:
    """Enforce capabilities, provenance, concurrency, and idempotent closure."""

    def __init__(self, oracle: TeacherOracle) -> None:
        self.capabilities = oracle.capabilities
        self._oracle = oracle
        self._capacity = asyncio.Semaphore(self.capabilities.max_concurrency)
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._in_flight: set[asyncio.Task] = set()

    async def score(self, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        if self._closed:
            raise RuntimeError("teacher oracle is closed")
        validate_teacher_request_against_capabilities(request, self.capabilities)
        task = asyncio.current_task()
        assert task is not None
        self._in_flight.add(task)
        try:
            async with self._capacity:
                evidence = await self._oracle.score(request)
        finally:
            self._in_flight.remove(task)
        validate_teacher_evidence(request, evidence)
        if evidence.teacher_revision != self.capabilities.teacher_revision:
            raise ValueError("teacher evidence does not match the negotiated teacher revision")
        return evidence

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._in_flight:
                await asyncio.gather(*tuple(self._in_flight), return_exceptions=True)
            await self._oracle.close()


TeacherOracleFactory = Callable[[], Awaitable[TeacherOracle]]


@dataclass(frozen=True)
class TeacherEndpoint:
    """One named physical replica of a logical teacher."""

    endpoint_id: str
    oracle: TeacherOracle


@dataclass
class _TeacherEndpointState:
    endpoint_id: str
    oracle: ValidatedTeacherOracle
    pending_tokens: int = 0
    unhealthy_until: float = 0


class TeacherEndpointPool:
    """Select healthy replicas by pending token load and fail over retryable requests."""

    def __init__(
        self,
        endpoints: tuple[TeacherEndpoint, ...],
        *,
        failure_cooldown_seconds: float = 30,
    ) -> None:
        if not endpoints:
            raise ValueError("teacher endpoint pool requires at least one endpoint")
        if len({endpoint.endpoint_id for endpoint in endpoints}) != len(endpoints):
            raise ValueError("teacher endpoint IDs must be unique within a pool")
        if any(not endpoint.endpoint_id.strip() for endpoint in endpoints):
            raise ValueError("teacher endpoint IDs must be non-empty strings")
        if failure_cooldown_seconds < 0:
            raise ValueError("teacher endpoint failure cooldown must be non-negative")

        first_capabilities = endpoints[0].oracle.capabilities
        capability_identity = replace(first_capabilities, max_concurrency=1)
        for endpoint in endpoints[1:]:
            if replace(endpoint.oracle.capabilities, max_concurrency=1) != capability_identity:
                raise ValueError("teacher endpoints in a pool must advertise the same logical teacher contract")
        self.capabilities = replace(
            first_capabilities,
            max_concurrency=sum(endpoint.oracle.capabilities.max_concurrency for endpoint in endpoints),
        )
        self._endpoints = tuple(
            _TeacherEndpointState(endpoint.endpoint_id, ValidatedTeacherOracle(endpoint.oracle))
            for endpoint in endpoints
        )
        self._failure_cooldown_seconds = failure_cooldown_seconds
        self._state_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def score(self, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        if self._closed:
            raise RuntimeError("teacher endpoint pool is closed")
        request_tokens = int(request.prompt_mask.sum().item() + request.response_mask.sum().item())
        attempted: set[str] = set()
        failures: list[BaseException] = []
        while len(attempted) < len(self._endpoints):
            endpoint = await self._select_endpoint(request_tokens, attempted)
            if endpoint is None:
                break
            attempted.add(endpoint.endpoint_id)
            try:
                return await endpoint.oracle.score(request)
            except TeacherEndpointUnavailable as error:
                failures.append(error)
                async with self._state_lock:
                    endpoint.unhealthy_until = time.monotonic() + self._failure_cooldown_seconds
            finally:
                async with self._state_lock:
                    endpoint.pending_tokens -= request_tokens

        if failures:
            raise ExceptionGroup("all available teacher endpoints failed", failures)
        raise TeacherEndpointUnavailable(f"no healthy endpoints for teacher {request.teacher_id!r}")

    async def _select_endpoint(
        self,
        request_tokens: int,
        attempted: set[str],
    ) -> _TeacherEndpointState | None:
        """Return a healthy least-loaded endpoint and reserve its token load, or ``None`` if none remain."""
        async with self._state_lock:
            now = time.monotonic()
            candidates = [
                endpoint
                for endpoint in self._endpoints
                if endpoint.endpoint_id not in attempted and endpoint.unhealthy_until <= now
            ]
            if not candidates:
                return None
            endpoint = min(candidates, key=lambda candidate: candidate.pending_tokens)
            endpoint.pending_tokens += request_tokens
            return endpoint

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await _close_owned_oracles(
                (endpoint.oracle for endpoint in self._endpoints),
                message="teacher endpoint cleanup failed",
            )


@dataclass
class _ResidentTeacher:
    oracle: ValidatedTeacherOracle
    loaded_at: float
    last_used_sequence: int
    in_flight: int = 0


class RotatingTeacherOracleOwner:
    """Own a bounded set of local teachers and rotate only at drained residency boundaries."""

    def __init__(
        self,
        factories: Mapping[str, TeacherOracleFactory],
        *,
        max_resident: int,
        minimum_residency_seconds: float,
    ) -> None:
        if not factories:
            raise ValueError("rotating teacher owner requires at least one factory")
        if any(not teacher_id.strip() for teacher_id in factories):
            raise ValueError("rotating teacher factory names must be non-empty strings")
        if max_resident <= 0:
            raise ValueError("max_resident must be positive")
        if minimum_residency_seconds < 0:
            raise ValueError("minimum_residency_seconds must be non-negative")
        self._factories = dict(factories)
        self._max_resident = max_resident
        self._minimum_residency_seconds = minimum_residency_seconds
        self._residents: dict[str, _ResidentTeacher] = {}
        self._condition = asyncio.Condition()
        self._usage_counter = 0
        self._closed = False

    @property
    def teacher_ids(self) -> frozenset[str]:
        return frozenset(self._factories)

    async def score(self, teacher_id: str, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        resident = await self._acquire(teacher_id)
        try:
            return await resident.oracle.score(request)
        finally:
            async with self._condition:
                resident.in_flight -= 1
                self._condition.notify_all()

    async def _acquire(self, teacher_id: str) -> _ResidentTeacher:
        if teacher_id not in self._factories:
            raise ValueError(f"unknown rotating teacher oracle {teacher_id!r}")
        async with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("rotating teacher oracle owner is closed")
                self._usage_counter += 1
                resident = self._residents.get(teacher_id)
                if resident is not None:
                    resident.in_flight += 1
                    resident.last_used_sequence = self._usage_counter
                    return resident
                if len(self._residents) < self._max_resident:
                    return await self._start_resident(teacher_id)

                now = time.monotonic()
                drained = [resident for resident in self._residents.values() if resident.in_flight == 0]
                eligible = [
                    resident for resident in drained if now - resident.loaded_at >= self._minimum_residency_seconds
                ]
                if eligible:
                    victim = min(eligible, key=lambda candidate: candidate.last_used_sequence)
                    victim_id = next(key for key, value in self._residents.items() if value is victim)
                    del self._residents[victim_id]
                    await victim.oracle.close()
                    return await self._start_resident(teacher_id)

                remaining_windows = [
                    self._minimum_residency_seconds - (now - resident.loaded_at) for resident in drained
                ]
                if remaining_windows:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=min(remaining_windows))
                    except TimeoutError:
                        pass
                else:
                    await self._condition.wait()

    async def _start_resident(self, teacher_id: str) -> _ResidentTeacher:
        oracle = ValidatedTeacherOracle(await self._factories[teacher_id]())
        if oracle.capabilities.teacher_id != teacher_id:
            await oracle.close()
            raise ValueError(
                f"rotating teacher oracle factory {teacher_id!r} returned {oracle.capabilities.teacher_id!r}"
            )
        resident = _ResidentTeacher(
            oracle=oracle,
            loaded_at=time.monotonic(),
            last_used_sequence=self._usage_counter,
            in_flight=1,
        )
        self._residents[teacher_id] = resident
        return resident

    async def close(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
            while any(resident.in_flight for resident in self._residents.values()):
                await self._condition.wait()
            residents = tuple(self._residents.values())
            self._residents.clear()
        await _close_owned_oracles(
            (resident.oracle for resident in residents),
            message="rotating teacher oracle cleanup failed",
        )


def _require_teacher_oracle_entries(oracles: Mapping[str, object]) -> None:
    if not oracles:
        raise ValueError("teacher oracle owner requires at least one entry")
    if any(not teacher_id.strip() for teacher_id in oracles):
        raise ValueError("teacher oracle names must be non-empty strings")


def _validate_teacher_oracle_identity(teacher_id: str, oracle: TeacherOracle) -> None:
    if oracle.capabilities.teacher_id != teacher_id:
        raise ValueError(f"teacher oracle {teacher_id!r} returned {oracle.capabilities.teacher_id!r}")


class TeacherOracleOwner:
    """Own a set of logical oracles, including partial-startup cleanup."""

    def __init__(self, oracles: dict[str, ValidatedTeacherOracle]) -> None:
        self._oracles = oracles
        self._close_lock = asyncio.Lock()
        self._closed = False

    @property
    def teacher_ids(self) -> frozenset[str]:
        return frozenset(self._oracles)

    @classmethod
    async def create(cls, factories: Mapping[str, TeacherOracleFactory]) -> Self:
        _require_teacher_oracle_entries(factories)

        oracles: dict[str, ValidatedTeacherOracle] = {}
        try:
            for teacher_id, factory in factories.items():
                oracle = ValidatedTeacherOracle(await factory())
                oracles[teacher_id] = oracle
                _validate_teacher_oracle_identity(teacher_id, oracle)
        except BaseException as startup_error:
            cleanup_errors = await _close_oracles(oracles.values())
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "teacher oracle startup and cleanup failed",
                    [startup_error, *cleanup_errors],
                )
            raise
        return cls(oracles)

    async def score(self, teacher_id: str, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        if self._closed:
            raise RuntimeError("teacher oracle owner is closed")
        try:
            oracle = self._oracles[teacher_id]
        except KeyError as error:
            raise ValueError(f"unknown teacher oracle {teacher_id!r}") from error
        return await oracle.score(request)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await _close_owned_oracles(self._oracles.values(), message="teacher oracle cleanup failed")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()


class TeacherOracleFleet:
    """Combine fixed endpoint pools and bounded rotating teachers behind one scorer."""

    def __init__(
        self,
        *,
        fixed: TeacherOracleOwner | None,
        rotating: RotatingTeacherOracleOwner | None,
    ) -> None:
        if fixed is None and rotating is None:
            raise ValueError("teacher oracle fleet requires at least one teacher")
        self._owners = tuple(owner for owner in (rotating, fixed) if owner is not None)
        self._owners_by_teacher: dict[str, TeacherOracleCollection] = {}
        for owner in self._owners:
            for teacher_id in owner.teacher_ids:
                if teacher_id in self._owners_by_teacher:
                    raise ValueError(f"teacher role cannot belong to multiple oracle collections: {teacher_id}")
                self._owners_by_teacher[teacher_id] = owner
        self._close_lock = asyncio.Lock()
        self._closed = False

    @property
    def teacher_ids(self) -> frozenset[str]:
        return frozenset(self._owners_by_teacher)

    async def score(self, teacher_id: str, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        if self._closed:
            raise RuntimeError("teacher oracle fleet is closed")
        try:
            owner = self._owners_by_teacher[teacher_id]
        except KeyError as error:
            raise ValueError(f"unknown teacher oracle {teacher_id!r}") from error
        return await owner.score(teacher_id, request)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            results = await asyncio.gather(*(owner.close() for owner in self._owners), return_exceptions=True)
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise BaseExceptionGroup("teacher oracle fleet cleanup failed", errors)


async def _close_oracles(oracles: Iterable[ValidatedTeacherOracle]) -> list[BaseException]:
    errors: list[BaseException] = []
    for oracle in reversed(tuple(oracles)):
        try:
            await oracle.close()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            errors.append(error)
    return errors


async def _close_owned_oracles(oracles: Iterable[ValidatedTeacherOracle], *, message: str) -> None:
    cleanup_errors = await _close_oracles(oracles)
    if cleanup_errors:
        raise BaseExceptionGroup(message, cleanup_errors)
