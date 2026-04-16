# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for slice lifecycle state machine transitions."""

import threading


from iris.cluster.controller.autoscaler.models import SliceLifecycleState, SliceState
from iris.cluster.controller.autoscaler.slice_lifecycle import (
    TRANSITIONS,
    SliceEvent,
    SliceSideEffectKind,
    dispatch_slice_event,
)
from rigging.timing import Timestamp
from tests.cluster.providers.conftest import make_fake_slice_handle


def _make_slices(
    slice_id: str = "slice-001",
    lifecycle: SliceLifecycleState = SliceLifecycleState.BOOTING,
    worker_ids: list[str] | None = None,
) -> tuple[dict[str, SliceState], threading.Lock]:
    handle = make_fake_slice_handle(slice_id)
    state = SliceState(
        handle=handle,
        lifecycle=lifecycle,
        worker_ids=worker_ids or [],
    )
    return {slice_id: state}, threading.Lock()


def test_booting_to_ready():
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.BOOTING)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_READY,
        {"worker_ids": ["w1", "w2"]},
        now=Timestamp.from_ms(1000),
    )
    assert result.applied
    assert result.prior_state == SliceLifecycleState.BOOTING
    assert result.new_state == SliceLifecycleState.READY
    assert result.event == SliceEvent.CLOUD_STATE_READY
    assert slices["slice-001"].lifecycle == SliceLifecycleState.READY
    effect_kinds = result.side_effects
    assert SliceSideEffectKind.REGISTER_WORKERS in effect_kinds


def test_booting_to_initializing():
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.BOOTING)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_INITIALIZING,
        now=Timestamp.from_ms(1000),
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.INITIALIZING
    assert result.side_effects == []


def test_initializing_to_ready():
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.INITIALIZING)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_READY,
        {"worker_ids": ["w1"]},
        now=Timestamp.from_ms(1000),
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.READY
    assert SliceSideEffectKind.REGISTER_WORKERS in result.side_effects


def test_booting_to_failed_on_cloud_failure():
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.BOOTING)
    # Fail well past SHORT_LIVED_SLICE_THRESHOLD so no backoff
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_FAILED,
        {"error_message": "VM crashed"},
        now=Timestamp.from_ms(1_600_000),
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    assert SliceSideEffectKind.DEREGISTER_WORKERS in result.side_effects
    assert SliceSideEffectKind.TERMINATE_SLICE in result.side_effects
    assert SliceSideEffectKind.RECORD_GROUP_FAILURE not in result.side_effects


def test_short_lived_slice_failure_triggers_backoff():
    """Slice that fails within SHORT_LIVED_SLICE_THRESHOLD of creation triggers RECORD_GROUP_FAILURE."""
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.BOOTING)
    # Default handle created_at is 1_000_000ms; fail 60s later (well within 5min threshold)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_FAILED,
        {"error_message": "preempted"},
        now=Timestamp.from_ms(1_060_000),
    )
    assert result.applied
    assert SliceSideEffectKind.RECORD_GROUP_FAILURE in result.side_effects


def test_long_lived_slice_failure_no_backoff():
    """Slice that fails after SHORT_LIVED_SLICE_THRESHOLD does NOT trigger RECORD_GROUP_FAILURE."""
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.BOOTING)
    # Default handle created_at is 1_000_000ms; fail 10min later (past 5min threshold)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_FAILED,
        {"error_message": "crashed"},
        now=Timestamp.from_ms(1_600_000),
    )
    assert result.applied
    assert SliceSideEffectKind.RECORD_GROUP_FAILURE not in result.side_effects
    assert SliceSideEffectKind.TERMINATE_SLICE in result.side_effects


def test_unknown_timeout():
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.BOOTING)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_UNKNOWN_TIMEOUT,
        now=Timestamp.from_ms(1000),
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    effect_kinds = result.side_effects
    assert SliceSideEffectKind.TERMINATE_SLICE in effect_kinds


def test_ready_to_failed_on_worker_failure():
    slices, lock = _make_slices(
        lifecycle=SliceLifecycleState.READY,
        worker_ids=["w1", "w2"],
    )
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.WORKER_FAILURE_REPORTED,
        {"failed_workers": ["w1"]},
        now=Timestamp.from_ms(1000),
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    effect_kinds = result.side_effects
    assert SliceSideEffectKind.DEREGISTER_WORKERS in effect_kinds
    assert SliceSideEffectKind.TERMINATE_SLICE in effect_kinds


def test_ready_to_failed_on_idle_timeout():
    slices, lock = _make_slices(
        lifecycle=SliceLifecycleState.READY,
        worker_ids=["w1"],
    )
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.IDLE_TIMEOUT,
        now=Timestamp.from_ms(1000),
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    effect_kinds = result.side_effects
    assert SliceSideEffectKind.DEREGISTER_WORKERS in effect_kinds
    assert SliceSideEffectKind.TERMINATE_SLICE in effect_kinds


def test_requesting_to_booting():
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.REQUESTING)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.PLATFORM_CALL_SUCCEEDED,
        now=Timestamp.from_ms(1000),
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.BOOTING
    assert result.side_effects == []


def test_requesting_to_failed_on_platform_error():
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.REQUESTING)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.PLATFORM_CALL_FAILED,
        now=Timestamp.from_ms(1000),
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    assert SliceSideEffectKind.RECORD_GROUP_FAILURE in result.side_effects


def test_invalid_transition_returns_none():
    """READY + CLOUD_STATE_INITIALIZING is not a valid transition."""
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.READY)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_INITIALIZING,
        now=Timestamp.from_ms(1000),
    )
    assert not result.applied
    assert slices["slice-001"].lifecycle == SliceLifecycleState.READY


def test_unknown_slice_returns_none():
    slices, lock = _make_slices()
    result = dispatch_slice_event(
        slices,
        lock,
        "nonexistent",
        SliceEvent.CLOUD_STATE_READY,
        {"worker_ids": ["w1"]},
        now=Timestamp.from_ms(1000),
    )
    assert not result.applied


def test_failed_is_terminal():
    """Once FAILED, no further transitions should be possible."""
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.FAILED)
    for event in SliceEvent:
        result = dispatch_slice_event(
            slices,
            lock,
            "slice-001",
            event,
            {"worker_ids": ["w1"]},
            now=Timestamp.from_ms(1000),
        )
        assert not result.applied, f"FAILED + {event} should not produce an applied transition"


def test_transition_table_exhaustiveness():
    """Every non-terminal state has at least one outgoing transition."""
    non_terminal = {
        SliceLifecycleState.REQUESTING,
        SliceLifecycleState.BOOTING,
        SliceLifecycleState.INITIALIZING,
        SliceLifecycleState.READY,
    }
    for state in non_terminal:
        transitions_from = [k for k in TRANSITIONS if k[0] == state]
        assert transitions_from, f"State {state} has no outgoing transitions"


def test_result_contains_timestamp():
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.BOOTING)
    ts = Timestamp.from_ms(42000)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_READY,
        {"worker_ids": ["w1"]},
        now=ts,
    )
    assert result.applied
    assert result.timestamp == ts


def test_ready_transition_sets_last_active_and_worker_ids():
    """Transitioning to READY must set last_active to prevent immediate scaledown."""
    slices, lock = _make_slices(lifecycle=SliceLifecycleState.BOOTING)
    ts = Timestamp.from_ms(5000)
    result = dispatch_slice_event(
        slices,
        lock,
        "slice-001",
        SliceEvent.CLOUD_STATE_READY,
        {"worker_ids": ["w1", "w2"]},
        now=ts,
    )
    assert result.applied
    state = slices["slice-001"]
    assert state.last_active == ts
    assert state.worker_ids == ["w1", "w2"]
