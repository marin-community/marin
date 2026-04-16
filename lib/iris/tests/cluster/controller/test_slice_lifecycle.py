# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for slice lifecycle state machine transitions, exercised via ScalingGroup.dispatch().

Organization:
- `test_table_transitions_produce_expected_state` is a parametrized sanity
  check that every entry in TRANSITIONS yields the declared new_state. Catches
  typos in the table without writing one test per entry.
- Invariant tests below cover behavior that isn't readable off the table:
  cross-machine cascades (backoff triggers), timestamp/worker-id bookkeeping
  on READY, exponential backoff math, exhaustiveness, and no-op paths.
"""

import pytest

from iris.cluster.controller.autoscaler.models import SliceLifecycleState, SliceState
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.autoscaler.slice_lifecycle import (
    BACKOFF_TRIGGERS,
    SHORT_LIVED_SLICE_THRESHOLD,
    TRANSITIONS,
    SliceEvent,
)
from rigging.timing import Duration, Timestamp
from tests.cluster.controller.conftest import make_scale_group_config
from tests.cluster.providers.conftest import make_fake_slice_handle, make_mock_platform

CREATED_AT_MS = 1_000_000
SHORT_LIVED_NOW = Timestamp.from_ms(CREATED_AT_MS + 60_000)
LONG_LIVED_NOW = Timestamp.from_ms(CREATED_AT_MS + int(SHORT_LIVED_SLICE_THRESHOLD.to_ms()) + 60_000)


@pytest.fixture
def group() -> ScalingGroup:
    config = make_scale_group_config(name="test", buffer_slices=0, max_slices=10)
    return ScalingGroup(config, make_mock_platform(), backoff_initial=Duration.from_seconds(5))


def _seed(
    group: ScalingGroup,
    slice_id: str = "slice-001",
    lifecycle: SliceLifecycleState = SliceLifecycleState.BOOTING,
):
    handle = make_fake_slice_handle(slice_id, created_at_ms=CREATED_AT_MS)
    with group._slices_lock:
        group._slices[slice_id] = SliceState(handle=handle, lifecycle=lifecycle)
    return handle


# ---------------------------------------------------------------------------
# Transition table sanity — every declared transition is reachable and
# produces its declared target state.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_state,event,expected_to_state",
    [(key[0], key[1], transition.to_state) for key, transition in TRANSITIONS.items()],
    ids=[f"{key[0].value}+{key[1].value}" for key in TRANSITIONS],
)
def test_table_transitions_produce_expected_state(
    from_state: SliceLifecycleState,
    event: SliceEvent,
    expected_to_state: SliceLifecycleState,
    group: ScalingGroup,
):
    handle = _seed(group, lifecycle=from_state)
    workers = handle.describe().workers
    result = group.dispatch("slice-001", event, {"workers": workers}, now=LONG_LIVED_NOW)
    assert result.applied
    assert result.prior_state == from_state
    assert result.new_state == expected_to_state


# ---------------------------------------------------------------------------
# READY invariants: last_active and worker_ids must be set atomically.
# Without this, freshly-ready slices are immediately scaledown-eligible and
# find_slice_for_worker() returns nothing, breaking worker-failure teardown.
# ---------------------------------------------------------------------------


def test_ready_transition_sets_last_active_and_worker_ids(group: ScalingGroup):
    handle = _seed(group)
    workers = handle.describe().workers
    result = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_READY, {"workers": workers}, now=SHORT_LIVED_NOW)

    assert result.registered_workers == list(workers)
    state = group._slices["slice-001"]
    assert state.last_active == SHORT_LIVED_NOW
    assert state.worker_ids == [w.worker_id for w in workers]


# ---------------------------------------------------------------------------
# Cross-machine backoff cascade: slice failure triggers (or doesn't trigger)
# group backoff based on slice age, event kind, and platform-vs-cloud origin.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event", sorted(BACKOFF_TRIGGERS))
def test_short_lived_failure_triggers_backoff(event: SliceEvent, group: ScalingGroup):
    """Every event in BACKOFF_TRIGGERS counts toward backoff when the slice was short-lived."""
    from_state = (
        SliceLifecycleState.READY if event == SliceEvent.WORKER_FAILURE_REPORTED else SliceLifecycleState.BOOTING
    )
    _seed(group, lifecycle=from_state)
    result = group.dispatch("slice-001", event, now=SHORT_LIVED_NOW)
    assert result.triggered_backoff
    assert group.consecutive_failures == 1


@pytest.mark.parametrize("event", sorted(BACKOFF_TRIGGERS))
def test_long_lived_failure_no_backoff(event: SliceEvent, group: ScalingGroup):
    from_state = (
        SliceLifecycleState.READY if event == SliceEvent.WORKER_FAILURE_REPORTED else SliceLifecycleState.BOOTING
    )
    _seed(group, lifecycle=from_state)
    result = group.dispatch("slice-001", event, now=LONG_LIVED_NOW)
    assert not result.triggered_backoff
    assert group.consecutive_failures == 0


def test_idle_timeout_never_triggers_backoff(group: ScalingGroup):
    """IDLE_TIMEOUT is healthy scaledown. Regression-guards against anyone adding
    IDLE_TIMEOUT to BACKOFF_TRIGGERS."""
    assert SliceEvent.IDLE_TIMEOUT not in BACKOFF_TRIGGERS
    _seed(group, lifecycle=SliceLifecycleState.READY)
    result = group.dispatch("slice-001", SliceEvent.IDLE_TIMEOUT, now=SHORT_LIVED_NOW)
    assert not result.triggered_backoff
    assert group.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Terminal-and-detached invariant: FAILED removes the slice from tracking and
# hands the handle back so the caller can async-terminate.
# ---------------------------------------------------------------------------


def test_failed_transition_detaches_slice_and_is_terminal(group: ScalingGroup):
    handle = _seed(group)
    result = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_FAILED, now=LONG_LIVED_NOW)
    assert result.detached_handle is handle
    assert "slice-001" not in group._slices

    # Terminal: subsequent events find no slice and noop.
    again = group.dispatch("slice-001", SliceEvent.CLOUD_STATE_READY, now=LONG_LIVED_NOW)
    assert not again.applied


# ---------------------------------------------------------------------------
# Backoff math: exponential progression and clearing on successful scale-up.
# ---------------------------------------------------------------------------


def test_exponential_backoff_progression(group: ScalingGroup):
    """Three short-lived failures → 5s, 10s, 20s. Asserts the exponent, not just count."""
    for i in range(3):
        _seed(group, slice_id=f"s-{i}")
        group.dispatch(f"s-{i}", SliceEvent.CLOUD_STATE_FAILED, now=SHORT_LIVED_NOW)
    assert group.consecutive_failures == 3
    assert group._backoff_until is not None
    expected_ms = SHORT_LIVED_NOW.epoch_ms() + 20_000
    assert group._backoff_until.as_timestamp().epoch_ms() == expected_ms


def test_complete_scale_up_clears_backoff(group: ScalingGroup):
    _seed(group)
    group.dispatch("slice-001", SliceEvent.CLOUD_STATE_FAILED, now=SHORT_LIVED_NOW)
    assert group.consecutive_failures == 1

    handle = make_fake_slice_handle("slice-002", created_at_ms=CREATED_AT_MS)
    ts = Timestamp.from_ms(CREATED_AT_MS)
    group.begin_scale_up(timestamp=ts)
    group.complete_scale_up(handle, ts)

    assert group.consecutive_failures == 0
    assert group._backoff_until is None


# ---------------------------------------------------------------------------
# No-op paths and structural invariants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slice_exists,from_state,event",
    [
        (False, None, SliceEvent.CLOUD_STATE_READY),  # unknown slice
        (True, SliceLifecycleState.READY, SliceEvent.CLOUD_STATE_INITIALIZING),  # invalid transition
    ],
    ids=["unknown_slice", "invalid_transition"],
)
def test_noop_returns_unapplied_result(
    slice_exists: bool,
    from_state: SliceLifecycleState | None,
    event: SliceEvent,
    group: ScalingGroup,
):
    if slice_exists:
        _seed(group, lifecycle=from_state)
    result = group.dispatch("slice-001", event, now=SHORT_LIVED_NOW)
    assert not result.applied
    assert result.detached_handle is None
    assert not result.triggered_backoff


def test_every_tracked_state_has_outgoing_transitions():
    """Every lifecycle state that can appear on a tracked slice must have outgoing
    transitions, or the slice gets stuck. (REQUESTING is not tracked on a slice — it
    is derived from the pending-scale-up counter.)"""
    tracked_non_terminal = {
        SliceLifecycleState.BOOTING,
        SliceLifecycleState.INITIALIZING,
        SliceLifecycleState.READY,
    }
    for state in tracked_non_terminal:
        assert any(key[0] == state for key in TRANSITIONS), f"{state.value} is a dead state"
