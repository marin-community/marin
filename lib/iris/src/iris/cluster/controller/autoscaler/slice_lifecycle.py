# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NamedTuple

from iris.cluster.controller.autoscaler.models import SliceLifecycleState, SliceState
from iris.cluster.controller.db import ControllerDB
from rigging.timing import Duration, Timestamp

logger = logging.getLogger(__name__)

SHORT_LIVED_SLICE_THRESHOLD = Duration.from_minutes(5)


class SliceEvent(StrEnum):
    PLATFORM_CALL_SUCCEEDED = "platform_call_succeeded"
    PLATFORM_CALL_FAILED = "platform_call_failed"
    CLOUD_STATE_INITIALIZING = "cloud_state_initializing"
    CLOUD_STATE_READY = "cloud_state_ready"
    CLOUD_STATE_FAILED = "cloud_state_failed"
    CLOUD_STATE_UNKNOWN_TIMEOUT = "cloud_state_unknown_timeout"
    WORKER_FAILURE_REPORTED = "worker_failure_reported"
    IDLE_TIMEOUT = "idle_timeout"


class SliceSideEffectKind(StrEnum):
    REGISTER_WORKERS = "register_workers"
    DEREGISTER_WORKERS = "deregister_workers"
    RECORD_GROUP_FAILURE = "record_group_failure"
    TERMINATE_SLICE = "terminate_slice"


_EMPTY_EFFECTS: list[SliceSideEffectKind] = []


@dataclass(frozen=True)
class TransitionResult:
    slice_id: str
    prior_state: SliceLifecycleState
    new_state: SliceLifecycleState
    event: SliceEvent
    side_effects: list[SliceSideEffectKind] = field(default_factory=list)
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    applied: bool = True


# Sentinel for "no transition happened" — callers don't need to check for None
NOOP = TransitionResult(
    slice_id="",
    prior_state=SliceLifecycleState.FAILED,
    new_state=SliceLifecycleState.FAILED,
    event=SliceEvent.CLOUD_STATE_FAILED,
    applied=False,
)


class SliceTransition(NamedTuple):
    to_state: SliceLifecycleState
    side_effects: Callable[[SliceState, Timestamp], list[SliceSideEffectKind]] | None = None


# ---------------------------------------------------------------------------
# Side-effect computation functions
# ---------------------------------------------------------------------------

_TEARDOWN_EFFECTS: list[SliceSideEffectKind] = [
    SliceSideEffectKind.DEREGISTER_WORKERS,
    SliceSideEffectKind.TERMINATE_SLICE,
]


def _on_ready(state: SliceState, now: Timestamp) -> list[SliceSideEffectKind]:
    return [SliceSideEffectKind.REGISTER_WORKERS]


def _on_failure_with_teardown(state: SliceState, now: Timestamp) -> list[SliceSideEffectKind]:
    effects = list(_TEARDOWN_EFFECTS)
    age = Duration.from_ms(now.epoch_ms() - state.handle.created_at.epoch_ms())
    if age < SHORT_LIVED_SLICE_THRESHOLD:
        effects.append(SliceSideEffectKind.RECORD_GROUP_FAILURE)
    return effects


def _on_creation_failed(state: SliceState, now: Timestamp) -> list[SliceSideEffectKind]:
    return [SliceSideEffectKind.RECORD_GROUP_FAILURE]


def _on_idle_teardown(state: SliceState, now: Timestamp) -> list[SliceSideEffectKind]:
    return list(_TEARDOWN_EFFECTS)


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

_R, _B, _I, _RDY, _F = (
    SliceLifecycleState.REQUESTING,
    SliceLifecycleState.BOOTING,
    SliceLifecycleState.INITIALIZING,
    SliceLifecycleState.READY,
    SliceLifecycleState.FAILED,
)
_SUCCEEDED = SliceEvent.PLATFORM_CALL_SUCCEEDED
_P_FAILED = SliceEvent.PLATFORM_CALL_FAILED
_C_INIT = SliceEvent.CLOUD_STATE_INITIALIZING
_C_READY = SliceEvent.CLOUD_STATE_READY
_C_FAILED = SliceEvent.CLOUD_STATE_FAILED
_C_TIMEOUT = SliceEvent.CLOUD_STATE_UNKNOWN_TIMEOUT
_W_FAILED = SliceEvent.WORKER_FAILURE_REPORTED
_IDLE = SliceEvent.IDLE_TIMEOUT

TRANSITIONS: dict[tuple[SliceLifecycleState, SliceEvent], SliceTransition] = {
    (_R, _SUCCEEDED):   SliceTransition(_B),
    (_R, _P_FAILED):    SliceTransition(_F, _on_creation_failed),
    (_B, _C_INIT):      SliceTransition(_I),
    (_B, _C_READY):     SliceTransition(_RDY, _on_ready),
    (_B, _C_FAILED):    SliceTransition(_F, _on_failure_with_teardown),
    (_B, _C_TIMEOUT):   SliceTransition(_F, _on_failure_with_teardown),
    (_B, _W_FAILED):    SliceTransition(_F, _on_failure_with_teardown),
    (_I, _C_READY):     SliceTransition(_RDY, _on_ready),
    (_I, _C_FAILED):    SliceTransition(_F, _on_failure_with_teardown),
    (_I, _C_TIMEOUT):   SliceTransition(_F, _on_failure_with_teardown),
    (_I, _W_FAILED):    SliceTransition(_F, _on_failure_with_teardown),
    (_RDY, _W_FAILED):  SliceTransition(_F, _on_failure_with_teardown),
    (_RDY, _IDLE):      SliceTransition(_F, _on_idle_teardown),
}  # fmt: skip


# ---------------------------------------------------------------------------
# Cloud state → event mapping
# ---------------------------------------------------------------------------

from iris.cluster.providers.types import CloudSliceState  # noqa: E402


def cloud_state_to_event(
    cloud_state: CloudSliceState,
    handle_created_at: Timestamp,
    now: Timestamp,
    unresolvable_timeout: Duration,
) -> SliceEvent | None:
    """Map a cloud slice state to a lifecycle event, or None to skip (e.g. UNKNOWN within timeout)."""
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
        logger.debug(
            "Slice UNKNOWN (age %s < timeout %s); will retry",
            age,
            unresolvable_timeout,
        )
        return None
    return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_slice_event(
    slices: dict[str, SliceState],
    lock: threading.Lock,
    slice_id: str,
    event: SliceEvent,
    context: dict[str, Any] | None = None,
    *,
    now: Timestamp | None = None,
    db: ControllerDB | None = None,
    group_name: str = "",
) -> TransitionResult:
    """Apply a lifecycle event to a slice. Returns NOOP if the transition is invalid."""
    ctx = context or {}
    if now is None:
        now = Timestamp.now()

    with lock:
        state = slices.get(slice_id)
        if state is None:
            logger.warning("slice event for unknown slice %s (event=%s)", slice_id, event)
            return NOOP

        key = (state.lifecycle, event)
        transition = TRANSITIONS.get(key)
        if transition is None:
            logger.warning(
                "no transition for slice %s in state %s on event %s",
                slice_id,
                state.lifecycle,
                event,
            )
            return NOOP

        prior_state = state.lifecycle
        state.lifecycle = transition.to_state

        if transition.to_state == SliceLifecycleState.READY:
            state.last_active = now
            state.worker_ids = ctx.get("worker_ids", [])

        # Compute side effects while we still hold the lock (reads state.handle.created_at)
        side_effects = transition.side_effects(state, now) if transition.side_effects else _EMPTY_EFFECTS

    _log_transition(db, group_name, slice_id, prior_state, transition.to_state, event, ctx, now)

    logger.info(
        "slice transition: %s %s → %s (event=%s, effects=%s)",
        slice_id,
        prior_state,
        transition.to_state,
        event,
        [e.value for e in side_effects],
    )

    return TransitionResult(
        slice_id=slice_id,
        prior_state=prior_state,
        new_state=transition.to_state,
        event=event,
        side_effects=side_effects,
        timestamp=now,
    )


def _log_transition(
    db: ControllerDB | None,
    group_name: str,
    slice_id: str,
    prior: SliceLifecycleState,
    new: SliceLifecycleState,
    event: SliceEvent,
    context: dict[str, Any],
    timestamp: Timestamp,
) -> None:
    if db is None:
        return
    safe_ctx = {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
    db.execute(
        "INSERT INTO slice_transitions"
        " (group_name, slice_id, timestamp_ms, event, from_state, to_state, context_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (group_name, slice_id, timestamp.epoch_ms(), event.value, prior.value, new.value, json.dumps(safe_ctx)),
    )
