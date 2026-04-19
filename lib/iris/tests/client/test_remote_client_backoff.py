# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for RemoteClusterClient job-wait backoff."""

from __future__ import annotations

from iris.cluster.client.remote_client import MAX_STATE_POLL_INTERVAL, RemoteClusterClient
from iris.cluster.types import JobName
from iris.rpc import job_pb2, logging_pb2


class _FakeRemoteClient(RemoteClusterClient):
    """RemoteClusterClient wired to counters instead of real RPCs.

    Returns RUNNING for the first ``running_polls`` state queries, then
    SUCCEEDED. ``get_job_status`` and ``fetch_logs`` return empty fixtures so
    the streaming path can exercise its full loop.
    """

    def __init__(self, running_polls: int):
        # Skip RemoteClusterClient.__init__ — no RPC transport is needed since
        # every outbound call is overridden below.
        self._running_polls = running_polls
        self._poll_count = 0
        self._fetch_log_count = 0

    def _poll_job_state(self, job_id: JobName) -> int:
        state = (
            job_pb2.JOB_STATE_RUNNING if self._poll_count < self._running_polls else job_pb2.JOB_STATE_SUCCEEDED
        )
        self._poll_count += 1
        return state

    def get_job_status(self, job_id: JobName) -> job_pb2.JobStatus:
        return job_pb2.JobStatus(job_id=job_id.to_wire(), state=job_pb2.JOB_STATE_SUCCEEDED)

    def fetch_logs(self, source: str, **_kwargs) -> logging_pb2.FetchLogsResponse:
        self._fetch_log_count += 1
        return logging_pb2.FetchLogsResponse()


def _record_sleeps(monkeypatch) -> list[float]:
    """Replace ``time.sleep`` in remote_client with a non-blocking recorder."""
    recorded: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("iris.cluster.client.remote_client.time.sleep", _fake_sleep)
    return recorded


def test_wait_for_job_backoff_caps_at_max_state_poll_interval(monkeypatch):
    """State polls back off up to ~30s, not the old ~2s ceiling."""
    sleeps = _record_sleeps(monkeypatch)

    client = _FakeRemoteClient(running_polls=40)
    client.wait_for_job(JobName.root("user", "job"), timeout=10_000.0)

    assert sleeps, "expected at least one sleep interval"

    # Early polls ramp rapidly from the 100ms initial value.
    assert sleeps[0] < 0.5, f"initial sleep should start small, got {sleeps[0]}"
    reached_one_second = next((i for i, s in enumerate(sleeps) if s >= 1.0), None)
    assert reached_one_second is not None and reached_one_second <= 8, (
        f"should reach 1s within a few polls, got sleeps={sleeps[:10]}"
    )

    # Backoff must actually exceed the old 2s cap.
    assert max(sleeps) > 2.0, f"backoff still stuck at old ceiling: {sleeps}"

    # And must not exceed the documented max (allowing a small jitter margin).
    assert max(sleeps) <= MAX_STATE_POLL_INTERVAL * 1.15, f"backoff exceeded cap: {max(sleeps)}"


def test_wait_for_job_with_streaming_uses_exponential_backoff(monkeypatch):
    """Streaming loop also backs off, not fixed-interval polling."""
    sleeps = _record_sleeps(monkeypatch)

    client = _FakeRemoteClient(running_polls=20)
    client.wait_for_job_with_streaming(JobName.root("user", "job"), timeout=10_000.0)

    # The final recorded sleep is the 1s terminal-drain pause; the rest are the
    # exponential-backoff sleeps between running-state polls.
    assert sleeps, "expected state-poll sleeps before termination"
    assert sleeps[-1] == 1.0, f"last sleep should be terminal drain, got {sleeps[-1]}"
    loop_sleeps = sleeps[:-1]

    assert loop_sleeps[0] < loop_sleeps[min(5, len(loop_sleeps) - 1)], (
        f"streaming loop should ramp up, got {loop_sleeps[:10]}"
    )
    assert max(loop_sleeps) > 2.0, f"streaming loop still capped at old 2s: {loop_sleeps}"


def test_wait_for_job_respects_custom_poll_interval(monkeypatch):
    """Tests pass poll_interval=0.1 for fast polling — must still be honored."""
    sleeps = _record_sleeps(monkeypatch)

    client = _FakeRemoteClient(running_polls=5)
    client.wait_for_job(JobName.root("user", "job"), timeout=100.0, poll_interval=0.1)

    assert sleeps and max(sleeps) <= 0.12, f"custom poll_interval cap violated: {sleeps}"
