# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The persistence surface the federation manager drives, as a Protocol.

The manager owns the *orchestration* of handoff, sync, and cancel (retry loops,
per-peer sync ticks, race handling); every durable mutation goes through a
:class:`FederationStore`. The controller implements it against its own tables,
so the manager stays a self-contained module that depends only on this Protocol
and can be exercised with a fake store.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from iris.cluster.types import JobName
from iris.rpc import controller_pb2


class HandoffState(IntEnum):
    """The ``federated_jobs.handoff_state`` lifecycle for one handle."""

    PENDING_HANDOFF = 0  # persisted locally, peer has not yet acked LaunchJob
    HANDED_OFF = 1  # peer acked; the sync loop now mirrors its state
    HANDOFF_FAILED = 2  # peer rejected the handoff terminally


class HandoffAdmission(IntEnum):
    """The outcome of admitting and persisting a handoff handle."""

    ADMITTED = 0  # new handle persisted in PENDING_HANDOFF
    ALREADY_EXISTS = 1  # a live handle for this job id already existed (idempotent resubmit)
    REJECTED = 2  # budget admission denied the handoff


@dataclass(frozen=True)
class HandoffSpec:
    """Everything the store needs to admit and persist one handoff handle.

    The store derives the budget subject (``parent_job_id.user``) and the
    reservation (``resource_value`` of the request's shape) itself, so those units
    stay in the controller alongside the local budget accounting.
    """

    parent_job_id: JobName  # this cluster's local (root) job id
    remote_job_id: str  # deterministic, globally-unique id the peer runs it under
    peer_id: str
    owner_principal: str  # end-user identity asserted to the peer
    request: controller_pb2.Controller.LaunchJobRequest  # normalized request, for job_config


@dataclass(frozen=True)
class HandoffOutcome:
    admission: HandoffAdmission
    reject_reason: str = ""


@dataclass(frozen=True)
class PendingHandoff:
    """A persisted handle awaiting first delivery (or re-delivery) to its peer."""

    parent_job_id: JobName
    remote_job_id: str
    peer_id: str
    owner_principal: str
    request: controller_pb2.Controller.LaunchJobRequest


@dataclass(frozen=True)
class CancelTarget:
    """What a routed cancel must address on the peer."""

    peer_id: str
    remote_job_id: str


class FederationStore(Protocol):
    """Durable operations the federation manager performs against the parent DB."""

    def admit_and_persist_handoff(self, spec: HandoffSpec) -> HandoffOutcome:
        """In one transaction: re-check existence (idempotent resubmit), run the
        federated budget admission, and persist the ``jobs`` row (``child_cluster``
        set, no tasks) + ``job_config`` + the ``federated_jobs`` handle in
        ``PENDING_HANDOFF``."""
        ...

    def mark_handed_off(self, parent_job_id: JobName, *, now_ms: int) -> None:
        """Flip a handle to ``HANDED_OFF`` after the peer acks its ``LaunchJob``."""
        ...

    def mark_handoff_failed(self, parent_job_id: JobName, error: str) -> None:
        """Record a terminal handoff failure on the handle."""
        ...

    def pending_handoffs(self) -> list[PendingHandoff]:
        """Every handle still in ``PENDING_HANDOFF`` (boot re-drive + retry)."""
        ...

    def read_cursor(self, peer_id: str) -> str:
        """The persisted sync cursor for ``peer_id`` ("" on first contact)."""
        ...

    def apply_sync_batch(
        self,
        peer_id: str,
        deltas: Sequence[controller_pb2.Controller.FederationJobDelta],
        *,
        next_cursor: str,
        cursor_stale: bool,
    ) -> None:
        """Apply one sync batch in a single transaction: mirror each delta's job
        and task state into the local ``jobs``/``tasks`` rows (stamped
        ``child_cluster``), apply tombstones, advance the cursor. When
        ``cursor_stale`` the batch is the peer's full active set, so also
        set-replace: drop any local handle for ``peer_id`` absent from it."""
        ...

    def bump_cancel_intent(self, parent_job_id: JobName) -> CancelTarget | None:
        """Bump ``cancel_intent_version`` and return the peer/remote-id to cancel,
        or ``None`` if ``parent_job_id`` is not a federated handle."""
        ...

    def active_federated_job_count(self, peer_id: str) -> int:
        """Count of non-terminal federated handles delegated to ``peer_id``."""
        ...
