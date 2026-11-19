# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Share a bounded quarantine reserve without expiring unresolved sandbox claims."""

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object

from shellbox.backends.daytona.ownership import OwnershipResolution, TrialOwnershipRecord

CAS_ATTEMPTS = 16


class QuarantineReserveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    state_uri: str
    pool_size: int = Field(gt=0)
    max_extra_slots_per_root: Literal[1]
    root_keys: tuple[str, ...]
    allocation_uri: str
    allocation_sha256: str

    @model_validator(mode="after")
    def unique_roots(self) -> "QuarantineReserveConfig":
        if len(self.root_keys) != len(set(self.root_keys)):
            raise ValueError("Reserve root keys must be unique")
        return self


class ReserveClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    root_key: str
    owner: str
    created_at: datetime
    quarantines: tuple[TrialOwnershipRecord, ...]
    resolutions: dict[str, OwnershipResolution] = Field(default_factory=dict)


class ReserveState(BaseModel):
    config: QuarantineReserveConfig
    claims: dict[str, ReserveClaim] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded_claims(self) -> "ReserveState":
        if len(self.claims) > self.config.pool_size:
            raise ValueError("Quarantine reserve exceeds its granted capacity")
        for key, claim in self.claims.items():
            if key != claim.root_key or key not in self.config.root_keys:
                raise ValueError("Reserve claim does not belong to a configured root")
        return self


class ReserveDecision(StrEnum):
    ACQUIRED = "acquired"
    WAIT = "wait"
    TRACKED = "tracked"
    TRANSFERRED = "transferred"
    RELEASED = "released"


class ReserveReceipt(BaseModel):
    config: QuarantineReserveConfig
    root_key: str
    owner: str
    decision: ReserveDecision
    claim: ReserveClaim | None
    occupied_reserve_slots: int
    state_version: str | None
    observed_at: datetime


def _key(record: TrialOwnershipRecord) -> str:
    return f"{record.row_id}/{record.attempt}"


class QuarantineReservePool:
    """Reserve at most one extra slot per root until tracked sandboxes are confirmed absent."""

    def __init__(self, config: QuarantineReserveConfig):
        self.config = config
        self.object = conditional_object(config.state_uri)

    def _read(self, *, allow_missing: bool = False) -> tuple[ReserveState, str | None]:
        snapshot = self.object.read()
        if snapshot is None:
            if not allow_missing:
                raise ValueError("Shared reserve state is missing; do not recreate occupied capacity")
            return ReserveState(config=self.config), None
        state = ReserveState.model_validate_json(snapshot.data)
        if state.config != self.config:
            raise ValueError("Shared quarantine reserve configuration changed")
        return state, snapshot.version

    def snapshot(self) -> ReserveState:
        return self._read()[0]

    def initialize(self) -> None:
        state, version = self._read(allow_missing=True)
        if version is not None:
            return
        try:
            self.object.write(state.model_dump_json().encode(), expected_version=None)
        except ConditionalWriteError:
            self._read()

    def claim(self, root_key: str, owner: str, quarantines: tuple[TrialOwnershipRecord, ...]) -> ReserveReceipt:
        """Claim at most one extra sandbox slot; crashes never make a claim expire."""
        if root_key not in self.config.root_keys or not quarantines:
            raise ValueError("A reserve claim requires a configured root and frozen quarantine evidence")
        claim = ReserveClaim(
            claim_id=uuid4().hex, root_key=root_key, owner=owner, created_at=datetime.now(UTC), quarantines=quarantines
        )
        for _ in range(CAS_ATTEMPTS):
            state, version = self._read()
            existing = state.claims.get(root_key)
            if existing is not None:
                decision = ReserveDecision.ACQUIRED if existing.owner == owner else ReserveDecision.WAIT
                return self._receipt(state, version, root_key, owner, decision, existing)
            if len(state.claims) >= self.config.pool_size:
                return self._receipt(state, version, root_key, owner, ReserveDecision.WAIT, None)
            updated = ReserveState(config=self.config, claims={**state.claims, root_key: claim})
            try:
                version = self.object.write(updated.model_dump_json().encode(), expected_version=version)
            except ConditionalWriteError:
                continue
            return self._receipt(updated, version, root_key, owner, ReserveDecision.ACQUIRED, claim)
        state, version = self._read()
        return self._receipt(state, version, root_key, owner, ReserveDecision.WAIT, None)

    def handoff(self, claim: ReserveClaim, new_owner: str) -> ReserveReceipt:
        """Transfer a stopped root's claim without releasing its unresolved capacity.

        The caller must verify the previous producer is terminal and preserve its
        ownership ledger before handing capacity to a replacement producer.
        """
        for _ in range(CAS_ATTEMPTS):
            state, version = self._read()
            current = state.claims.get(claim.root_key)
            if current != claim:
                raise ValueError("Frozen claim changed before its terminal-root handoff")
            replacement = claim.model_copy(update={"claim_id": uuid4().hex, "owner": new_owner})
            updated = ReserveState(config=self.config, claims={**state.claims, claim.root_key: replacement})
            try:
                version = self.object.write(updated.model_dump_json().encode(), expected_version=version)
            except ConditionalWriteError:
                continue
            return self._receipt(updated, version, claim.root_key, new_owner, ReserveDecision.TRANSFERRED, replacement)
        raise ConditionalWriteError("Quarantine handoff remained contended; retain the old claim")

    def track(self, claim: ReserveClaim, record: TrialOwnershipRecord) -> ReserveReceipt:
        """Retain every additional quarantine that can occupy the root's extra slot."""
        return self._update(claim, record, None)

    def reconciled(
        self, claim: ReserveClaim, record: TrialOwnershipRecord, resolution: OwnershipResolution
    ) -> ReserveReceipt:
        """Release only after all tracked frozen ownership evidence is confirmed absent."""
        if resolution.evidence != record.ownership:
            raise ValueError("Reserve release proof belongs to another frozen ownership record")
        return self._update(claim, record, resolution)

    def _update(
        self, claim: ReserveClaim, record: TrialOwnershipRecord, resolution: OwnershipResolution | None
    ) -> ReserveReceipt:
        for _ in range(CAS_ATTEMPTS):
            state, version = self._read()
            current = state.claims.get(claim.root_key)
            if current is None or (current.claim_id, current.owner) != (claim.claim_id, claim.owner):
                raise ValueError("Quarantine reserve claim is no longer owned by this root attempt")
            records = {_key(row): row for row in current.quarantines}
            key = _key(record)
            if key in records and records[key] != record:
                raise ValueError("Frozen quarantine evidence changed")
            if resolution is not None and key not in records:
                raise ValueError("Release proof does not cover a tracked quarantine")
            records[key] = record
            resolutions = dict(current.resolutions)
            if resolution is not None:
                resolutions[key] = resolution
            updated_claim = current.model_copy(
                update={"quarantines": tuple(records.values()), "resolutions": resolutions}
            )
            released = all(key in resolutions and resolutions[key].resolved for key in records)
            claims = dict(state.claims)
            if released:
                del claims[claim.root_key]
            else:
                claims[claim.root_key] = updated_claim
            updated = ReserveState(config=self.config, claims=claims)
            try:
                version = self.object.write(updated.model_dump_json().encode(), expected_version=version)
            except ConditionalWriteError:
                continue
            decision = ReserveDecision.RELEASED if released else ReserveDecision.TRACKED
            return self._receipt(updated, version, claim.root_key, claim.owner, decision, updated_claim)
        raise ConditionalWriteError("Quarantine reserve remained contended; retain the slot")

    def _receipt(self, state, version, root_key, owner, decision, claim) -> ReserveReceipt:
        return ReserveReceipt(
            config=self.config,
            root_key=root_key,
            owner=owner,
            decision=decision,
            claim=claim,
            occupied_reserve_slots=len(state.claims),
            state_version=version,
            observed_at=datetime.now(UTC),
        )


class RootQuarantineReserve:
    """Bind one root attempt to a shared pool and persist every capacity decision."""

    def __init__(
        self, pool: QuarantineReservePool, root_key: str, owner: str, persist: Callable[[ReserveReceipt], None]
    ):
        self.pool, self.root_key, self.owner, self.persist = pool, root_key, owner, persist
        self.claim: ReserveClaim | None = None

    def try_acquire(self, quarantines: tuple[TrialOwnershipRecord, ...]) -> bool:
        receipt = self.pool.claim(self.root_key, self.owner, quarantines)
        self.persist(receipt)
        if receipt.decision is not ReserveDecision.ACQUIRED:
            return False
        self.claim = receipt.claim
        return True

    def track(self, record: TrialOwnershipRecord) -> None:
        if self.claim is None:
            return
        receipt = self.pool.track(self.claim, record)
        self.persist(receipt)
        self.claim = receipt.claim

    def reconciled(self, record: TrialOwnershipRecord, resolution: OwnershipResolution) -> None:
        if self.claim is None:
            return
        receipt = self.pool.reconciled(self.claim, record, resolution)
        self.persist(receipt)
        self.claim = None if receipt.decision is ReserveDecision.RELEASED else receipt.claim
