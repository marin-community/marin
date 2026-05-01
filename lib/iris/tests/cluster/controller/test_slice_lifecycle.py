# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for slice lifecycle state machine transitions, exercised via ScalingGroup.dispatch().

Organization:
- `test_table_transitions_produce_expected_state` is a parametrized sanity
  check that every entry in TRANSITIONS yields the declared new_state. Catches
  typos in the table without writing one test per entry.
- Invariant tests below cover behavior that isn't readable off the table:
  cross-machine cascades (backoff triggers), timestamp/worker-id bookkeeping
  on READY, exponential backoff math, exhaustiveness, atomic failure
  accounting under concurrent dispatches, and no-op paths.
"""

import threading

import pytest
from iris.cluster.controller.autoscaler.models import SliceLifecycleState, SliceState
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.autoscaler.slice_lifecycle import (
    SHORT_LIVED_SLICE_THRESHOLD,
    TRANSITIONS,
    BecameFailed,
    BecameReady,
    CloudFailed,
    CloudInitializing,
    CloudReady,
    IdleTimeout,
    InternalTransition,
    NoOp,
    SliceEvent,
    UnknownTimeout,
    WorkerFailure,
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


def _example_event(event_type: type[SliceEvent], workers=()) -> SliceEvent:
    """Construct a sample event of the given type with minimal valid payload."""
    if event_type is CloudReady:
        return CloudReady(workers=tuple(workers))
    if event_type is CloudFailed:
        return CloudFailed(error_message="boom")
    if event_type is CloudInitializing:
        return CloudInitializing()
    if event_type is UnknownTimeout:
        return UnknownTimeout()
    if event_type is WorkerFailure:
        return WorkerFailure(failed_worker_ids=("w1",))
    if event_type is IdleTimeout:
        return IdleTimeout(target_capacity=0, ready_before=1)
    raise AssertionError(f"no example for {event_type}")


# ---------------------------------------------------------------------------
# Transition table sanity — every declared transition is reachable and
# produces its declared target state.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_state,event_type,expected_to_state",
    [(key[0], key[1], to) for key, to in TRANSITIONS.items()],
    ids=[f"{key[0].value}+{key[1].__name__}" for key in TRANSITIONS],
)
def test_table_transitions_produce_expected_state(
    from_state: SliceLifecycleState,
    event_type: type[SliceEvent],
    expected_to_state: SliceLifecycleState,
    group: ScalingGroup,
):
    handle = _seed(group, lifecycle=from_state)
    event = _example_event(event_type, workers=handle.describe().workers)
    outcome = group.dispatch("slice-001", event, now=LONG_LIVED_NOW)
    if expected_to_state == SliceLifecycleState.READY:
        assert isinstance(outcome, BecameReady)
    elif expected_to_state == SliceLifecycleState.FAILED:
        assert isinstance(outcome, BecameFailed)
    else:
        assert isinstance(outcome, InternalTransition)
        assert outcome.new_state == expected_to_state
        assert outcome.prior == from_state


# ---------------------------------------------------------------------------
# READY invariants: last_active and worker_ids must be set atomically.
# ---------------------------------------------------------------------------


def test_ready_transition_sets_last_active_and_worker_ids(group: ScalingGroup):
    handle = _seed(group)
    workers = tuple(handle.describe().workers)
    outcome = group.dispatch("slice-001", CloudReady(workers=workers), now=SHORT_LIVED_NOW)

    assert isinstance(outcome, BecameReady)
    assert outcome.workers == workers
    state = group._slices["slice-001"]
    assert state.last_active == SHORT_LIVED_NOW
    assert state.worker_ids == [w.worker_id for w in workers]


# ---------------------------------------------------------------------------
# Cross-machine backoff cascade.
# ---------------------------------------------------------------------------


BACKOFF_TRIGGER_TYPES: list[type[SliceEvent]] = [CloudFailed, UnknownTimeout, WorkerFailure]


@pytest.mark.parametrize("event_type", BACKOFF_TRIGGER_TYPES, ids=lambda t: t.__name__)
def test_short_lived_failure_triggers_backoff(event_type: type[SliceEvent], group: ScalingGroup):
    from_state = SliceLifecycleState.READY if event_type is WorkerFailure else SliceLifecycleState.BOOTING
    _seed(group, lifecycle=from_state)
    outcome = group.dispatch("slice-001", _example_event(event_type), now=SHORT_LIVED_NOW)
    assert isinstance(outcome, BecameFailed)
    assert outcome.triggered_backoff
    assert group.consecutive_failures == 1


@pytest.mark.parametrize("event_type", BACKOFF_TRIGGER_TYPES, ids=lambda t: t.__name__)
def test_long_lived_failure_no_backoff(event_type: type[SliceEvent], group: ScalingGroup):
    from_state = SliceLifecycleState.READY if event_type is WorkerFailure else SliceLifecycleState.BOOTING
    _seed(group, lifecycle=from_state)
    outcome = group.dispatch("slice-001", _example_event(event_type), now=LONG_LIVED_NOW)
    assert isinstance(outcome, BecameFailed)
    assert not outcome.triggered_backoff
    assert group.consecutive_failures == 0


def test_idle_timeout_never_triggers_backoff(group: ScalingGroup):
    """IDLE_TIMEOUT is healthy scaledown; counts_toward_backoff must stay False."""
    assert not IdleTimeout.counts_toward_backoff
    _seed(group, lifecycle=SliceLifecycleState.READY)
    outcome = group.dispatch("slice-001", IdleTimeout(target_capacity=0, ready_before=1), now=SHORT_LIVED_NOW)
    assert isinstance(outcome, BecameFailed)
    assert not outcome.triggered_backoff
    assert group.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Terminal-and-detached invariant.
# ---------------------------------------------------------------------------


def test_failed_transition_detaches_slice_and_is_terminal(group: ScalingGroup):
    handle = _seed(group)
    outcome = group.dispatch("slice-001", CloudFailed(error_message="x"), now=LONG_LIVED_NOW)
    assert isinstance(outcome, BecameFailed)
    assert outcome.handle is handle
    assert "slice-001" not in group._slices

    # Terminal: subsequent events find no slice and noop.
    again = group.dispatch("slice-001", CloudReady(workers=()), now=LONG_LIVED_NOW)
    assert isinstance(again, NoOp)


# ---------------------------------------------------------------------------
# Backoff math.
# ---------------------------------------------------------------------------


def test_exponential_backoff_progression(group: ScalingGroup):
    """Three short-lived failures → 5s, 10s, 20s. Asserts the exponent, not just count."""
    for i in range(3):
        _seed(group, slice_id=f"s-{i}")
        group.dispatch(f"s-{i}", CloudFailed(error_message=""), now=SHORT_LIVED_NOW)
    assert group.consecutive_failures == 3
    assert group._backoff_until is not None
    expected_ms = SHORT_LIVED_NOW.epoch_ms() + 20_000
    assert group._backoff_until.as_timestamp().epoch_ms() == expected_ms


def test_complete_scale_up_clears_backoff(group: ScalingGroup):
    _seed(group)
    group.dispatch("slice-001", CloudFailed(error_message=""), now=SHORT_LIVED_NOW)
    assert group.consecutive_failures == 1

    handle = make_fake_slice_handle("slice-002", created_at_ms=CREATED_AT_MS)
    ts = Timestamp.from_ms(CREATED_AT_MS)
    group.begin_scale_up(timestamp=ts)
    group.complete_scale_up(handle, ts)

    assert group.consecutive_failures == 0
    assert group._backoff_until is None


# ---------------------------------------------------------------------------
# Atomic failure accounting under concurrent dispatches.
# Guards the lock widening: _apply_failure_locked is called under _slices_lock,
# so concurrent failures on different slices in the same group must sum
# consecutive_failures without lost updates.
# ---------------------------------------------------------------------------


def test_concurrent_failures_account_atomically(group: ScalingGroup):
    n = 32
    for i in range(n):
        _seed(group, slice_id=f"s-{i}")

    start = threading.Event()
    threads = []
    for i in range(n):
        sid = f"s-{i}"

        def run(sid=sid):
            start.wait()
            group.dispatch(sid, CloudFailed(error_message=""), now=SHORT_LIVED_NOW)

        t = threading.Thread(target=run)
        t.start()
        threads.append(t)

    start.set()
    for t in threads:
        t.join()

    assert group.consecutive_failures == n
    assert len(group._slices) == 0


# ---------------------------------------------------------------------------
# No-op paths and structural invariants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slice_exists,from_state,event",
    [
        (False, None, CloudReady(workers=())),  # unknown slice
        (True, SliceLifecycleState.READY, CloudInitializing()),  # invalid transition
    ],
    ids=["unknown_slice", "invalid_transition"],
)
def test_noop_returns_noop_outcome(
    slice_exists: bool,
    from_state: SliceLifecycleState | None,
    event: SliceEvent,
    group: ScalingGroup,
):
    if slice_exists:
        _seed(group, lifecycle=from_state)
    outcome = group.dispatch("slice-001", event, now=SHORT_LIVED_NOW)
    assert isinstance(outcome, NoOp)


def test_every_tracked_state_has_outgoing_transitions():
    """Every lifecycle state that can appear on a tracked slice must have outgoing
    transitions, or the slice gets stuck. REQUESTING is not tracked on a slice —
    it is derived from the pending-scale-up counter."""
    tracked_non_terminal = {
        SliceLifecycleState.BOOTING,
        SliceLifecycleState.INITIALIZING,
        SliceLifecycleState.READY,
    }
    for state in tracked_non_terminal:
        assert any(key[0] == state for key in TRANSITIONS), f"{state.value} is a dead state"
