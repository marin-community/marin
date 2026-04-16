# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for slice lifecycle state machine transitions, exercised via ScalingGroup.dispatch()."""

import pytest

from iris.cluster.controller.autoscaler.models import SliceLifecycleState
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.autoscaler.slice_lifecycle import (
    SHORT_LIVED_SLICE_THRESHOLD,
    TRANSITIONS,
    SliceEvent,
)
from rigging.timing import Duration, Timestamp
from tests.cluster.controller.conftest import make_scale_group_config
from tests.cluster.providers.conftest import make_fake_slice_handle, make_mock_platform

CREATED_AT_MS = 1_000_000
SHORT_LIVED_NOW = Timestamp.from_ms(CREATED_AT_MS + 60_000)  # 60s after creation, well within threshold
LONG_LIVED_NOW = Timestamp.from_ms(CREATED_AT_MS + int(SHORT_LIVED_SLICE_THRESHOLD.to_ms()) + 60_000)


@pytest.fixture
def group() -> ScalingGroup:
    config = make_scale_group_config(name="test", buffer_slices=0, max_slices=10)
    return ScalingGroup(config, make_mock_platform(), backoff_initial=Duration.from_seconds(5))


def _seed(
    group: ScalingGroup, slice_id: str = "slice-001", lifecycle: SliceLifecycleState = SliceLifecycleState.BOOTING
):
    handle = make_fake_slice_handle(slice_id, created_at_ms=CREATED_AT_MS)
    from iris.cluster.controller.autoscaler.models import SliceState

    state = SliceState(handle=handle, lifecycle=lifecycle)
    with group._slices_lock:
        group._slices[slice_id] = state
    return handle


def test_booting_to_ready(group: ScalingGroup):
    handle = _seed(group, lifecycle=SliceLifecycleState.BOOTING)
    fake_workers = handle.describe().workers

    result = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_READY, {"workers": fake_workers}, now=SHORT_LIVED_NOW)

    assert result.applied
    assert result.prior_state == SliceLifecycleState.BOOTING
    assert result.new_state == SliceLifecycleState.READY
    assert result.registered_workers == list(fake_workers)
    assert group._slices["slice-001"].lifecycle == SliceLifecycleState.READY
    assert group._slices["slice-001"].last_active == SHORT_LIVED_NOW


def test_booting_to_initializing(group: ScalingGroup):
    _seed(group)
    result = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_INITIALIZING, now=SHORT_LIVED_NOW)
    assert result.applied
    assert result.new_state == SliceLifecycleState.INITIALIZING
    assert result.detached_handle is None
    assert not result.triggered_backoff


def test_initializing_to_ready(group: ScalingGroup):
    handle = _seed(group, lifecycle=SliceLifecycleState.INITIALIZING)
    fake_workers = handle.describe().workers
    result = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_READY, {"workers": fake_workers}, now=SHORT_LIVED_NOW)
    assert result.applied
    assert result.new_state == SliceLifecycleState.READY
    assert result.registered_workers == list(fake_workers)


def test_short_lived_failure_triggers_backoff(group: ScalingGroup):
    """Slice that fails within SHORT_LIVED_SLICE_THRESHOLD bumps consecutive_failures."""
    handle = _seed(group)
    result = group.dispatch(
        "slice-001", SliceEvent.CLOUD_STATE_FAILED, {"error_message": "preempted"}, now=SHORT_LIVED_NOW
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    assert result.detached_handle is handle
    assert result.triggered_backoff
    assert group.consecutive_failures == 1
    assert group._backoff_until is not None


def test_long_lived_failure_no_backoff(group: ScalingGroup):
    """Slice that fails past the threshold detaches but does NOT trigger backoff."""
    handle = _seed(group)
    result = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_FAILED, {"error_message": "crashed"}, now=LONG_LIVED_NOW)
    assert result.applied
    assert result.detached_handle is handle
    assert not result.triggered_backoff
    assert group.consecutive_failures == 0


def test_unknown_timeout_short_lived_triggers_backoff(group: ScalingGroup):
    _seed(group)
    result = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_UNKNOWN_TIMEOUT, now=SHORT_LIVED_NOW)
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    assert result.triggered_backoff


def test_worker_failure_short_lived_triggers_backoff(group: ScalingGroup):
    handle = _seed(group, lifecycle=SliceLifecycleState.READY)
    result = group.dispatch(
        "slice-001", SliceEvent.WORKER_FAILURE_REPORTED, {"failed_workers": ["w1"]}, now=SHORT_LIVED_NOW
    )
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    assert result.detached_handle is handle
    assert result.triggered_backoff


def test_idle_timeout_no_backoff(group: ScalingGroup):
    """IDLE_TIMEOUT is healthy scaledown — never counts toward group backoff."""
    handle = _seed(group, lifecycle=SliceLifecycleState.READY)
    result = group.dispatch("slice-001", SliceEvent.IDLE_TIMEOUT, now=SHORT_LIVED_NOW)
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    assert result.detached_handle is handle
    assert not result.triggered_backoff
    assert group.consecutive_failures == 0


def test_requesting_to_booting(group: ScalingGroup):
    _seed(group, lifecycle=SliceLifecycleState.REQUESTING)
    result = group.dispatch("slice-001", SliceEvent.PLATFORM_CALL_SUCCEEDED, now=SHORT_LIVED_NOW)
    assert result.applied
    assert result.new_state == SliceLifecycleState.BOOTING
    assert result.detached_handle is None


def test_requesting_failure_triggers_backoff_unconditionally(group: ScalingGroup):
    """PLATFORM_CALL_FAILED triggers backoff regardless of slice age — platform errors always count."""
    handle = _seed(group, lifecycle=SliceLifecycleState.REQUESTING)
    result = group.dispatch("slice-001", SliceEvent.PLATFORM_CALL_FAILED, now=SHORT_LIVED_NOW)
    assert result.applied
    assert result.new_state == SliceLifecycleState.FAILED
    assert result.detached_handle is handle
    assert result.triggered_backoff


def test_invalid_transition_returns_noop(group: ScalingGroup):
    """READY + CLOUD_STATE_INITIALIZING is not a valid transition."""
    _seed(group, lifecycle=SliceLifecycleState.READY)
    result = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_INITIALIZING, now=SHORT_LIVED_NOW)
    assert not result.applied
    assert group._slices["slice-001"].lifecycle == SliceLifecycleState.READY


def test_unknown_slice_returns_noop(group: ScalingGroup):
    result = group.dispatch("nonexistent", SliceEvent.CLOUD_STATE_READY, {"worker_ids": ["w1"]})
    assert not result.applied


def test_failed_is_terminal(group: ScalingGroup):
    """A FAILED slice is detached, so any event afterwards finds no slice."""
    _seed(group)
    group.dispatch("slice-001", SliceEvent.CLOUD_STATE_FAILED, now=SHORT_LIVED_NOW)
    for event in SliceEvent:
        result = group.dispatch("slice-001", event, now=SHORT_LIVED_NOW)
        assert not result.applied, f"FAILED-then-{event} should noop"


def test_transition_table_exhaustiveness():
    """Every non-terminal state has at least one outgoing transition."""
    non_terminal = {
        SliceLifecycleState.REQUESTING,
        SliceLifecycleState.BOOTING,
        SliceLifecycleState.INITIALIZING,
        SliceLifecycleState.READY,
    }
    for state in non_terminal:
        outgoing = [k for k in TRANSITIONS if k[0] == state]
        assert outgoing, f"State {state} has no outgoing transitions"


def test_backoff_is_exponential(group: ScalingGroup):
    """Repeated short-lived failures double the backoff each time."""
    for i in range(3):
        _seed(group, slice_id=f"s-{i}")
        group.dispatch(f"s-{i}", SliceEvent.CLOUD_STATE_FAILED, now=SHORT_LIVED_NOW)
    assert group.consecutive_failures == 3
    # 5s * 2^2 = 20s
    assert group._backoff_until is not None
    expected_ms = SHORT_LIVED_NOW.epoch_ms() + 20_000
    assert group._backoff_until.as_timestamp().epoch_ms() == expected_ms


def test_complete_scale_up_clears_backoff(group: ScalingGroup):
    """Successful scale-up clears accumulated failure state."""
    _seed(group)
    group.dispatch("slice-001", SliceEvent.CLOUD_STATE_FAILED, now=SHORT_LIVED_NOW)
    assert group.consecutive_failures == 1

    handle = make_fake_slice_handle("slice-002", created_at_ms=CREATED_AT_MS)
    group.begin_scale_up(timestamp=Timestamp.from_ms(CREATED_AT_MS))
    group.complete_scale_up(handle, Timestamp.from_ms(CREATED_AT_MS))

    assert group.consecutive_failures == 0
    assert group._backoff_until is None
