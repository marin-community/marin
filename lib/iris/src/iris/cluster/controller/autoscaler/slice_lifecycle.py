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
from rigging.timing import Timestamp

logger = logging.getLogger(__name__)


class SliceEvent(StrEnum):
    PLATFORM_CALL_STARTED = "platform_call_started"
    PLATFORM_CALL_SUCCEEDED = "platform_call_succeeded"
    PLATFORM_CALL_FAILED = "platform_call_failed"
    CLOUD_STATE_INITIALIZING = "cloud_state_initializing"
    CLOUD_STATE_READY = "cloud_state_ready"
    CLOUD_STATE_FAILED = "cloud_state_failed"
    CLOUD_STATE_UNKNOWN_TIMEOUT = "cloud_state_unknown_timeout"
    WORKER_FAILURE_REPORTED = "worker_failure_reported"
    IDLE_TIMEOUT = "idle_timeout"
    TEARDOWN_COMPLETE = "teardown_complete"


class SliceSideEffectKind(StrEnum):
    REGISTER_WORKERS = "register_workers"
    DEREGISTER_WORKERS = "deregister_workers"
    RECORD_GROUP_FAILURE = "record_group_failure"
    TERMINATE_SLICE = "terminate_slice"


@dataclass(frozen=True)
class SliceSideEffect:
    kind: SliceSideEffectKind
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionResult:
    slice_id: str
    prior_state: SliceLifecycleState
    new_state: SliceLifecycleState
    event: SliceEvent
    side_effects: list[SliceSideEffect]
    timestamp: Timestamp


class InvalidTransitionError(Exception):
    pass


class SliceTransition(NamedTuple):
    to_state: SliceLifecycleState
    side_effects: Callable[[SliceState, dict[str, Any]], list[SliceSideEffect]] | None = None


# ---------------------------------------------------------------------------
# Side-effect computation functions
# ---------------------------------------------------------------------------

REQUESTING = SliceLifecycleState.REQUESTING
BOOTING = SliceLifecycleState.BOOTING
INITIALIZING = SliceLifecycleState.INITIALIZING
READY = SliceLifecycleState.READY
FAILED = SliceLifecycleState.FAILED

PLATFORM_CALL_SUCCEEDED = SliceEvent.PLATFORM_CALL_SUCCEEDED
PLATFORM_CALL_FAILED = SliceEvent.PLATFORM_CALL_FAILED
CLOUD_STATE_INITIALIZING = SliceEvent.CLOUD_STATE_INITIALIZING
CLOUD_STATE_READY = SliceEvent.CLOUD_STATE_READY
CLOUD_STATE_FAILED = SliceEvent.CLOUD_STATE_FAILED
CLOUD_STATE_UNKNOWN_TIMEOUT = SliceEvent.CLOUD_STATE_UNKNOWN_TIMEOUT
WORKER_FAILURE_REPORTED = SliceEvent.WORKER_FAILURE_REPORTED
IDLE_TIMEOUT = SliceEvent.IDLE_TIMEOUT


def _on_ready(state: SliceState, ctx: dict[str, Any]) -> list[SliceSideEffect]:
    return [SliceSideEffect(SliceSideEffectKind.REGISTER_WORKERS, {"worker_ids": ctx["worker_ids"]})]


def _on_bootstrap_failed(state: SliceState, ctx: dict[str, Any]) -> list[SliceSideEffect]:
    effects: list[SliceSideEffect] = [
        SliceSideEffect(SliceSideEffectKind.DEREGISTER_WORKERS),
        SliceSideEffect(SliceSideEffectKind.TERMINATE_SLICE),
    ]
    if ctx.get("is_short_lived"):
        effects.append(SliceSideEffect(SliceSideEffectKind.RECORD_GROUP_FAILURE))
    return effects


def _on_timeout(state: SliceState, ctx: dict[str, Any]) -> list[SliceSideEffect]:
    return _on_bootstrap_failed(state, ctx)


def _on_creation_failed(state: SliceState, ctx: dict[str, Any]) -> list[SliceSideEffect]:
    return [SliceSideEffect(SliceSideEffectKind.RECORD_GROUP_FAILURE)]


def _on_worker_failure(state: SliceState, ctx: dict[str, Any]) -> list[SliceSideEffect]:
    effects: list[SliceSideEffect] = [
        SliceSideEffect(SliceSideEffectKind.DEREGISTER_WORKERS),
        SliceSideEffect(SliceSideEffectKind.TERMINATE_SLICE),
    ]
    if ctx.get("is_short_lived"):
        effects.append(SliceSideEffect(SliceSideEffectKind.RECORD_GROUP_FAILURE))
    return effects


def _on_idle_teardown(state: SliceState, ctx: dict[str, Any]) -> list[SliceSideEffect]:
    return [
        SliceSideEffect(SliceSideEffectKind.DEREGISTER_WORKERS),
        SliceSideEffect(SliceSideEffectKind.TERMINATE_SLICE),
    ]


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

TRANSITIONS: dict[tuple[SliceLifecycleState, SliceEvent], SliceTransition] = {
    # REQUESTING -> BOOTING (platform call succeeded)
    (REQUESTING, PLATFORM_CALL_SUCCEEDED): SliceTransition(BOOTING),
    (REQUESTING, PLATFORM_CALL_FAILED): SliceTransition(FAILED, _on_creation_failed),
    # BOOTING -> INITIALIZING, READY, or FAILED
    (BOOTING, CLOUD_STATE_INITIALIZING): SliceTransition(INITIALIZING),
    (BOOTING, CLOUD_STATE_READY): SliceTransition(READY, _on_ready),
    (BOOTING, CLOUD_STATE_FAILED): SliceTransition(FAILED, _on_bootstrap_failed),
    (BOOTING, CLOUD_STATE_UNKNOWN_TIMEOUT): SliceTransition(FAILED, _on_timeout),
    (BOOTING, WORKER_FAILURE_REPORTED): SliceTransition(FAILED, _on_worker_failure),
    # INITIALIZING -> READY or FAILED
    (INITIALIZING, CLOUD_STATE_READY): SliceTransition(READY, _on_ready),
    (INITIALIZING, CLOUD_STATE_FAILED): SliceTransition(FAILED, _on_bootstrap_failed),
    (INITIALIZING, CLOUD_STATE_UNKNOWN_TIMEOUT): SliceTransition(FAILED, _on_timeout),
    (INITIALIZING, WORKER_FAILURE_REPORTED): SliceTransition(FAILED, _on_worker_failure),
    # READY -> FAILED (worker failure or idle teardown)
    (READY, WORKER_FAILURE_REPORTED): SliceTransition(FAILED, _on_worker_failure),
    (READY, IDLE_TIMEOUT): SliceTransition(FAILED, _on_idle_teardown),
}


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
) -> TransitionResult | None:
    """Apply a lifecycle event to a slice, returning the transition result or None if invalid."""
    context = context or {}
    if now is None:
        now = Timestamp.now()

    with lock:
        state = slices.get(slice_id)
        if state is None:
            logger.warning("slice event for unknown slice %s (event=%s)", slice_id, event)
            return None

        key = (state.lifecycle, event)
        transition = TRANSITIONS.get(key)
        if transition is None:
            logger.warning(
                "no transition for slice %s in state %s on event %s",
                slice_id,
                state.lifecycle,
                event,
            )
            return None

        prior_state = state.lifecycle
        state.lifecycle = transition.to_state

    side_effects = transition.side_effects(state, context) if transition.side_effects else []

    _log_transition(db, group_name, slice_id, prior_state, transition.to_state, event, context, now)

    logger.info(
        "slice transition: %s %s → %s (event=%s, effects=%s)",
        slice_id,
        prior_state,
        transition.to_state,
        event,
        [e.kind.value for e in side_effects],
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
    # Filter context to JSON-serializable values for the audit log
    safe_ctx = {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
    db.execute(
        "INSERT INTO slice_transitions"
        " (group_name, slice_id, timestamp_ms, event, from_state, to_state, context_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (group_name, slice_id, timestamp.epoch_ms(), event.value, prior.value, new.value, json.dumps(safe_ctx)),
    )
