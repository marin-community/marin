# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Slice lifecycle state machine: typed events, table of transitions, variant outcomes.

Events are discriminated-union dataclasses carrying the exact payload each
transition needs; there is no untyped context dict at the API boundary.
Cross-machine cascades (short-lived failure → group backoff) live on
ScalingGroup, which holds _slices_lock across both the slice mutation and
the group failure accounting so the two stay consistent.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from iris.cluster.controller.autoscaler.models import SliceLifecycleState
from iris.cluster.providers.types import CloudSliceState, RemoteWorkerHandle, SliceHandle
from rigging.timing import Duration, Timestamp

logger = logging.getLogger(__name__)

SHORT_LIVED_SLICE_THRESHOLD = Duration.from_minutes(5)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceEvent:
    """Base class for slice lifecycle events.

    Subclasses carry exactly the payload the transition needs. Class-level
    attributes (counts_toward_backoff, label) are read from the concrete type
    so dispatch logic never branches on event kind.
    """

    counts_toward_backoff: ClassVar[bool] = False
    label: ClassVar[str] = "slice_event"


@dataclass(frozen=True)
class CloudInitializing(SliceEvent):
    label: ClassVar[str] = "cloud_initializing"


@dataclass(frozen=True)
class CloudReady(SliceEvent):
    workers: tuple[RemoteWorkerHandle, ...]
    label: ClassVar[str] = "cloud_ready"


@dataclass(frozen=True)
class CloudFailed(SliceEvent):
    error_message: str = ""
    counts_toward_backoff: ClassVar[bool] = True
    label: ClassVar[str] = "cloud_failed"


@dataclass(frozen=True)
class UnknownTimeout(SliceEvent):
    counts_toward_backoff: ClassVar[bool] = True
    label: ClassVar[str] = "cloud_unknown_timeout"


@dataclass(frozen=True)
class WorkerFailure(SliceEvent):
    failed_worker_ids: tuple[str, ...]
    counts_toward_backoff: ClassVar[bool] = True
    label: ClassVar[str] = "worker_failure"


@dataclass(frozen=True)
class IdleTimeout(SliceEvent):
    target_capacity: int
    ready_before: int
    label: ClassVar[str] = "idle_timeout"


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

_B = SliceLifecycleState.BOOTING
_I = SliceLifecycleState.INITIALIZING
_R = SliceLifecycleState.READY
_F = SliceLifecycleState.FAILED

TRANSITIONS: dict[tuple[SliceLifecycleState, type[SliceEvent]], SliceLifecycleState] = {
    (_B, CloudInitializing): _I,
    (_B, CloudReady):        _R,
    (_B, CloudFailed):       _F,
    (_B, UnknownTimeout):    _F,
    (_B, WorkerFailure):     _F,
    (_I, CloudReady):        _R,
    (_I, CloudFailed):       _F,
    (_I, UnknownTimeout):    _F,
    (_I, WorkerFailure):     _F,
    (_R, WorkerFailure):     _F,
    (_R, IdleTimeout):       _F,
}  # fmt: skip


# ---------------------------------------------------------------------------
# Outcomes: variant type the caller match-dispatches on.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoOp:
    """Unknown slice or invalid (state, event) pair; caller has no work to do."""

    slice_id: str
    reason: str


@dataclass(frozen=True)
class InternalTransition:
    """State changed but the transition has no externally visible side effect
    (e.g. BOOTING → INITIALIZING). Persistence is handled inside dispatch."""

    slice_id: str
    prior: SliceLifecycleState
    new_state: SliceLifecycleState


@dataclass(frozen=True)
class BecameReady:
    """Slice reached READY. Caller registers workers and logs a slice_ready action."""

    slice_id: str
    workers: tuple[RemoteWorkerHandle, ...]


@dataclass(frozen=True)
class BecameFailed:
    """Slice reached FAILED and has been detached from the group's _slices map.

    Caller unregisters workers, async-terminates the handle, and logs. If
    triggered_backoff is True, the group's consecutive_failures/backoff_until
    were also updated atomically under the same lock as the slice detach.
    """

    slice_id: str
    handle: SliceHandle
    event: SliceEvent
    triggered_backoff: bool


TransitionOutcome = NoOp | InternalTransition | BecameReady | BecameFailed


# ---------------------------------------------------------------------------
# Event derivation from cloud observations.
# ---------------------------------------------------------------------------


def cloud_event(
    cloud_state: CloudSliceState,
    handle_created_at: Timestamp,
    now: Timestamp,
    unresolvable_timeout: Duration,
    *,
    workers: Sequence[RemoteWorkerHandle] = (),
    error_message: str = "",
) -> SliceEvent | None:
    """Map a cloud-side observation to a typed lifecycle event.

    Returns None for UNKNOWN within the timeout (retryable) and for any
    cloud_state the state machine does not model.
    """
    if cloud_state == CloudSliceState.READY:
        return CloudReady(workers=tuple(workers))
    if cloud_state == CloudSliceState.FAILED:
        return CloudFailed(error_message=error_message)
    if cloud_state in (CloudSliceState.BOOTSTRAPPING, CloudSliceState.CREATING):
        return CloudInitializing()
    if cloud_state == CloudSliceState.UNKNOWN:
        age = Duration.from_ms(now.epoch_ms() - handle_created_at.epoch_ms())
        if age >= unresolvable_timeout:
            return UnknownTimeout()
        logger.debug("Slice UNKNOWN (age %s < timeout %s); will retry", age, unresolvable_timeout)
        return None
    return None
