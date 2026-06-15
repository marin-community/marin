# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``marin.execution.sweep``.

These tests use a temp directory as ``sweep_root`` so the underlying
``step_lock`` / ``StatusFile`` machinery exercises real local-filesystem
locking — the only dependency we mock out is wall-clock time via the lock's
own heartbeat mechanism (which we don't need here because tests run quickly).
"""

import os
import threading

import pytest
from marin.execution.executor_step_status import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    StatusFile,
)
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.execution.sweep_coordination import _STOP, GangRole, _decode_round, _encode_round


@pytest.mark.parametrize("target_id", ["t01", "stop", _STOP, "run:t01", "weird id/with.chars"])
def test_round_codec_round_trips_any_target_id(target_id):
    """A target id round-trips through the wire encoding, including one equal to
    the stop sentinel — the ``run:`` prefix keeps targets and stop disjoint."""
    assert _decode_round(_encode_round(target_id)) == target_id


def test_round_codec_stop_is_none():
    assert _decode_round(_encode_round(None)) is None
    assert _encode_round(None) == _STOP


def _make_targets(n: int) -> list[SweepTarget]:
    return [SweepTarget(target_id=f"t{i:02d}", config={"i": i}) for i in range(n)]


class _FakeChannel:
    """In-memory round stream shared by a fake leader and its followers.

    Stands in for the Iris endpoint registry: ``publish`` records round ``seq``,
    ``receive`` blocks until it appears. ``target_id is None`` is the stop
    signal.
    """

    def __init__(self):
        self._rounds: dict[int, str | None] = {}
        self._cond = threading.Condition()

    def publish(self, seq: int, target_id: str | None) -> None:
        with self._cond:
            self._rounds[seq] = target_id
            self._cond.notify_all()

    def receive(self, seq: int, timeout: float = 10.0) -> str | None:
        with self._cond:
            if not self._cond.wait_for(lambda: seq in self._rounds, timeout=timeout):
                raise TimeoutError(f"round {seq} not published within {timeout}s")
            return self._rounds[seq]


class _LeaderCoordinator:
    def __init__(self, channel: _FakeChannel):
        self._channel = channel

    @property
    def role(self) -> GangRole:
        return GangRole.LEADER

    def publish(self, seq: int, target_id: str | None) -> None:
        self._channel.publish(seq, target_id)

    def receive(self, seq: int) -> str | None:
        raise AssertionError("leader must not receive")


class _FollowerCoordinator:
    def __init__(self, channel: _FakeChannel):
        self._channel = channel

    @property
    def role(self) -> GangRole:
        return GangRole.FOLLOWER

    def publish(self, seq: int, target_id: str | None) -> None:
        raise AssertionError("follower must not publish")

    def receive(self, seq: int) -> str | None:
        return self._channel.receive(seq)


def _status_at(sweep_root: str, target_id: str) -> str | None:
    return StatusFile(os.path.join(sweep_root, target_id), worker_id="check").status


@pytest.mark.parametrize(
    "num_workers,num_targets",
    [
        (1, 5),  # single worker, sequential
        (2, 5),  # contention, fewer workers than targets
        (3, 3),  # one worker per target
        (5, 2),  # more workers than targets
    ],
)
def test_claim_and_run_processes_every_target_exactly_once(num_workers, num_targets, tmp_path):
    """Across any (workers, targets) combination, every target runs exactly once.

    The contract is *exactly-once* across the worker pool — no duplicates and
    no misses. We do not assert each worker grabbed at least one target: with
    fast run_fns one worker can legitimately win every race.
    """
    sweep_root = str(tmp_path)
    targets = _make_targets(num_targets)

    # Concurrent writes would surface as a duplicate-claim assertion below.
    claimed: dict[str, str] = {}
    claim_lock = threading.Lock()
    errors: list[BaseException] = []
    # A barrier party-count of 1 makes barrier.wait() a no-op, so the
    # single-worker case skips the rendezvous cleanly.
    barrier = threading.Barrier(num_workers)

    def make_run(worker_name: str):
        def run(target: SweepTarget) -> None:
            with claim_lock:
                assert target.target_id not in claimed, (
                    f"{target.target_id} already claimed by {claimed.get(target.target_id)}, "
                    f"now also by {worker_name}"
                )
                claimed[target.target_id] = worker_name

        return run

    def worker(worker_name: str):
        try:
            barrier.wait(timeout=5)
            claim_and_run(sweep_root, targets, make_run(worker_name))
        except BaseException as exc:
            errors.append(exc)
            raise

    threads = [threading.Thread(target=worker, args=(f"W{i}",)) for i in range(num_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"worker errors: {errors}"
    assert set(claimed.keys()) == {t.target_id for t in targets}
    assert len(claimed) == num_targets
    for target in targets:
        assert _status_at(sweep_root, target.target_id) == STATUS_SUCCESS


def test_pre_existing_success_is_skipped(tmp_path):
    sweep_root = str(tmp_path)
    targets = _make_targets(3)

    pre_done = targets[1]
    pre_done_path = os.path.join(sweep_root, pre_done.target_id)
    os.makedirs(pre_done_path, exist_ok=True)
    StatusFile(pre_done_path, worker_id="seed").write_status(STATUS_SUCCESS)

    seen: list[str] = []

    def run(target: SweepTarget) -> None:
        seen.append(target.target_id)

    claim_and_run(sweep_root, targets, run)

    assert pre_done.target_id not in seen
    assert seen == [targets[0].target_id, targets[2].target_id]
    for target in targets:
        assert _status_at(sweep_root, target.target_id) == STATUS_SUCCESS


def test_run_fn_failure_marks_target_failed_and_propagates(tmp_path):
    """A failing target should mark itself FAILED and stop the worker.

    The remaining targets are left for a peer to retry. Since this test runs
    a single worker, the un-attempted targets simply remain unstarted.
    """
    sweep_root = str(tmp_path)
    targets = _make_targets(4)
    failing_id = targets[1].target_id

    seen: list[str] = []

    def run(target: SweepTarget) -> None:
        seen.append(target.target_id)
        if target.target_id == failing_id:
            raise RuntimeError(f"boom: {target.target_id}")

    with pytest.raises(RuntimeError, match=f"boom: {failing_id}"):
        claim_and_run(sweep_root, targets, run)

    # Saw target 0 (success) and target 1 (failure); never reached 2 or 3.
    assert seen == [targets[0].target_id, failing_id]
    assert _status_at(sweep_root, targets[0].target_id) == STATUS_SUCCESS
    assert _status_at(sweep_root, failing_id) == STATUS_FAILED
    # Untouched targets have no status yet.
    assert _status_at(sweep_root, targets[2].target_id) is None
    assert _status_at(sweep_root, targets[3].target_id) is None


def test_peer_can_retry_a_failed_target(tmp_path):
    """After one worker marks a target FAILED, a fresh worker retries it.

    ``step_lock`` defaults to ``force_run_failed=True`` so a STATUS_FAILED
    target is not treated as terminal across workers.
    """
    sweep_root = str(tmp_path)
    targets = _make_targets(2)
    failing_id = targets[0].target_id

    first_attempts: list[str] = []
    attempts_by_id: dict[str, int] = {}

    def first_run(target: SweepTarget) -> None:
        first_attempts.append(target.target_id)
        attempts_by_id[target.target_id] = attempts_by_id.get(target.target_id, 0) + 1
        if target.target_id == failing_id:
            raise RuntimeError("first attempt fails")

    with pytest.raises(RuntimeError, match="first attempt fails"):
        claim_and_run(sweep_root, targets, first_run)

    assert _status_at(sweep_root, failing_id) == STATUS_FAILED

    second_attempts: list[str] = []

    def second_run(target: SweepTarget) -> None:
        second_attempts.append(target.target_id)
        attempts_by_id[target.target_id] = attempts_by_id.get(target.target_id, 0) + 1

    claim_and_run(sweep_root, targets, second_run)

    # Second worker retried the failed target and ran the unstarted one.
    assert failing_id in second_attempts
    assert targets[1].target_id in second_attempts
    assert attempts_by_id[failing_id] == 2
    assert _status_at(sweep_root, failing_id) == STATUS_SUCCESS
    assert _status_at(sweep_root, targets[1].target_id) == STATUS_SUCCESS


# ---------------------------------------------------------------------------
# Gang coordination: leader claims + publishes; followers mirror.
# ---------------------------------------------------------------------------


def test_leader_publishes_only_claimed_rounds_then_stop(tmp_path):
    """The leader announces a round per *claimed* target plus a final stop.

    A target a peer already finished is skipped (``StepAlreadyDone``) and never
    announced, so followers never see it.
    """
    sweep_root = str(tmp_path)
    targets = _make_targets(3)

    pre_done = targets[1]
    pre_done_path = os.path.join(sweep_root, pre_done.target_id)
    os.makedirs(pre_done_path, exist_ok=True)
    StatusFile(pre_done_path, worker_id="seed").write_status(STATUS_SUCCESS)

    channel = _FakeChannel()
    leader_ran: list[str] = []

    claim_and_run(
        sweep_root,
        targets,
        lambda t: leader_ran.append(t.target_id),
        coordinator=_LeaderCoordinator(channel),
    )

    # Leader ran the two fresh targets, skipped the pre-done one.
    assert leader_ran == [targets[0].target_id, targets[2].target_id]
    # Rounds: one per claimed target (re-indexed, skip not announced) + stop.
    assert channel._rounds == {0: targets[0].target_id, 1: targets[2].target_id, 2: None}
    for target in targets:
        assert _status_at(sweep_root, target.target_id) == STATUS_SUCCESS


def test_follower_runs_exactly_the_announced_rounds(tmp_path):
    """A follower runs each announced target in order and stops on the sentinel.

    It never touches the lock or the status files — only the leader does.
    """
    targets = _make_targets(3)
    channel = _FakeChannel()
    # Leader claimed t00 and t02 (t01 was already done, so never announced).
    channel.publish(0, targets[0].target_id)
    channel.publish(1, targets[2].target_id)
    channel.publish(2, None)

    seen: list[str] = []
    claim_and_run(
        str(tmp_path),
        targets,
        lambda t: seen.append(t.target_id),
        coordinator=_FollowerCoordinator(channel),
    )

    assert seen == [targets[0].target_id, targets[2].target_id]
    # Follower wrote no status anywhere.
    for target in targets:
        assert _status_at(str(tmp_path), target.target_id) is None


def test_follower_run_fn_failure_propagates(tmp_path):
    """A failure inside a follower's ``run_fn`` propagates out of the worker."""
    targets = _make_targets(2)
    channel = _FakeChannel()
    channel.publish(0, targets[0].target_id)
    channel.publish(1, targets[1].target_id)
    channel.publish(2, None)

    def run(target: SweepTarget) -> None:
        if target.target_id == targets[1].target_id:
            raise RuntimeError("follower boom")

    with pytest.raises(RuntimeError, match="follower boom"):
        claim_and_run(str(tmp_path), targets, run, coordinator=_FollowerCoordinator(channel))


def test_leader_and_follower_agree_on_targets(tmp_path):
    """End-to-end: a concurrent leader and follower run the same target set.

    The leader claims every target (no peers), announcing each; the follower
    mirrors it. Both run all targets in the same order.
    """
    sweep_root = str(tmp_path)
    targets = _make_targets(4)
    channel = _FakeChannel()

    leader_ran: list[str] = []
    follower_ran: list[str] = []
    errors: list[BaseException] = []

    def leader():
        try:
            claim_and_run(
                sweep_root, targets, lambda t: leader_ran.append(t.target_id), coordinator=_LeaderCoordinator(channel)
            )
        except BaseException as exc:
            errors.append(exc)

    def follower():
        try:
            claim_and_run(
                sweep_root,
                targets,
                lambda t: follower_ran.append(t.target_id),
                coordinator=_FollowerCoordinator(channel),
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=leader), threading.Thread(target=follower)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"worker errors: {errors}"
    expected = [t.target_id for t in targets]
    assert leader_ran == expected
    assert follower_ran == expected
    for target in targets:
        assert _status_at(sweep_root, target.target_id) == STATUS_SUCCESS
