# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Slice lifecycle state machine: pure transition table and event mapping.

This module owns only the data — the discrete state transitions and the
helper that maps a CloudSliceState observation to a SliceEvent. Dispatch
logic and cross-machine cascades (e.g., short-lived failure → group backoff)
live on ScalingGroup, which can mutate slice and group state atomically
under a single lock.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NamedTuple

from iris.cluster.controller.autoscaler.models import SliceLifecycleState
from iris.cluster.providers.types import CloudSliceState, RemoteWorkerHandle, SliceHandle
from rigging.timing import Duration, Timestamp

logger = logging.getLogger(__name__)

SHORT_LIVED_SLICE_THRESHOLD = Duration.from_minutes(5)


class SliceEvent(StrEnum):
    CLOUD_STATE_INITIALIZING = "cloud_state_initializing"
    CLOUD_STATE_READY = "cloud_state_ready"
    CLOUD_STATE_FAILED = "cloud_state_failed"
    CLOUD_STATE_UNKNOWN_TIMEOUT = "cloud_state_unknown_timeout"
    WORKER_FAILURE_REPORTED = "worker_failure_reported"
    IDLE_TIMEOUT = "idle_timeout"


# Events whose FAILED transitions count toward group-level backoff (when the
# slice was short-lived). IDLE_TIMEOUT is a healthy scaledown and never counts.
BACKOFF_TRIGGERS: frozenset[SliceEvent] = frozenset(
    {
        SliceEvent.CLOUD_STATE_FAILED,
        SliceEvent.CLOUD_STATE_UNKNOWN_TIMEOUT,
        SliceEvent.WORKER_FAILURE_REPORTED,
    }
)


class Transition(NamedTuple):
    to_state: SliceLifecycleState


_B, _I, _RDY, _F = (
    SliceLifecycleState.BOOTING,
    SliceLifecycleState.INITIALIZING,
    SliceLifecycleState.READY,
    SliceLifecycleState.FAILED,
)
_C_INIT = SliceEvent.CLOUD_STATE_INITIALIZING
_C_READY = SliceEvent.CLOUD_STATE_READY
_C_FAILED = SliceEvent.CLOUD_STATE_FAILED
_C_TIMEOUT = SliceEvent.CLOUD_STATE_UNKNOWN_TIMEOUT
_W_FAILED = SliceEvent.WORKER_FAILURE_REPORTED
_IDLE = SliceEvent.IDLE_TIMEOUT

TRANSITIONS: dict[tuple[SliceLifecycleState, SliceEvent], Transition] = {
    (_B, _C_INIT):      Transition(_I),
    (_B, _C_READY):     Transition(_RDY),
    (_B, _C_FAILED):    Transition(_F),
    (_B, _C_TIMEOUT):   Transition(_F),
    (_B, _W_FAILED):    Transition(_F),
    (_I, _C_READY):     Transition(_RDY),
    (_I, _C_FAILED):    Transition(_F),
    (_I, _C_TIMEOUT):   Transition(_F),
    (_I, _W_FAILED):    Transition(_F),
    (_RDY, _W_FAILED):  Transition(_F),
    (_RDY, _IDLE):      Transition(_F),
}  # fmt: skip


@dataclass(frozen=True)
class TransitionResult:
    """Result of dispatching a slice event.

    `applied` is False when the (state, event) pair is not in TRANSITIONS or
    the slice is unknown — callers can blindly call _handle_transition without
    a None check.

    `detached_handle` is set when the transition removed the slice from
    tracking; the caller spawns the async termination thread on it.

    `registered_workers` is set on transitions to READY; the caller adds
    these to its worker registry.
    """

    slice_id: str
    prior_state: SliceLifecycleState
    new_state: SliceLifecycleState
    event: SliceEvent
    applied: bool = True
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    detached_handle: SliceHandle | None = None
    registered_workers: list[RemoteWorkerHandle] = field(default_factory=list)
    triggered_backoff: bool = False


NOOP = TransitionResult(
    slice_id="",
    prior_state=SliceLifecycleState.FAILED,
    new_state=SliceLifecycleState.FAILED,
    event=SliceEvent.CLOUD_STATE_FAILED,
    applied=False,
)


def cloud_state_to_event(
    cloud_state: CloudSliceState,
    handle_created_at: Timestamp,
    now: Timestamp,
    unresolvable_timeout: Duration,
) -> SliceEvent | None:
    """Map a cloud slice state to a lifecycle event, or None to skip (e.g., UNKNOWN within timeout)."""
    if cloud_state == CloudSliceState.READY:
        return SliceEvent.CLOUD_STATE_READY
    if cloud_state == CloudSliceState.FAILED:
        return SliceEvent.CLOUD_STATE_FAILED
    if cloud_state in (CloudSliceState.BOOTSTRAPPING, CloudSliceState.CREATING):
        return SliceEvent.CLOUD_STATE_INITIALIZING
    if cloud_state == CloudSliceState.UNKNOWN:
        age = Duration.from_ms(now.epoch_ms() - handle_created_at.epoch_ms())
        if age >= unresolvable_timeout:
            return SliceEvent.CLOUD_STATE_UNKNOWN_TIMEOUT
        logger.debug("Slice UNKNOWN (age %s < timeout %s); will retry", age, unresolvable_timeout)
        return None
    return None
