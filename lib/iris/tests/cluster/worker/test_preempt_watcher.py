# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the worker-side GCP preemption watcher.

The watcher polls the GCP instance metadata server for the ``preempted``
endpoint. It latches the ``on_preempt`` callback exactly once on a confirmed
``"TRUE"`` body; metadata-server errors, timeouts, and any other response keep
polling. On non-GCP hosts the loop is a no-op.
"""

import threading

import pytest
from iris.cluster.worker.preempt_watcher import PreemptWatcher
from rigging.timing import Duration

_FAST_POLL = Duration.from_seconds(0.01)


def _make_fetcher(bodies: list[str | None], stop_event: threading.Event | None = None):
    """Return a fetcher that yields successive ``bodies`` and stops the event when drained.

    Setting ``stop_event`` once the list is exhausted lets the watcher exit
    deterministically without ``time.sleep`` or wall-clock waiting.
    """
    iterator = iter(bodies)

    def fetcher(_key: str) -> str | None:
        try:
            return next(iterator)
        except StopIteration:
            if stop_event is not None:
                stop_event.set()
            return None

    return fetcher


def test_non_gcp_host_returns_immediately_without_firing_callback() -> None:
    """``is_gcp_vm`` False short-circuits: no polling, callback never invoked."""
    calls: list[None] = []
    fetcher_calls: list[str] = []

    def fetcher(key: str) -> str | None:
        fetcher_calls.append(key)
        return "TRUE"  # would latch if reached

    watcher = PreemptWatcher(
        on_preempt=lambda: calls.append(None),
        poll=_FAST_POLL,
        metadata_fetcher=fetcher,
        is_gcp_vm=lambda: False,
    )
    watcher.run(stop_event=threading.Event())

    assert calls == []
    assert fetcher_calls == []


def test_confirmed_true_body_fires_callback_exactly_once_and_returns() -> None:
    """A ``"TRUE"`` response latches: callback fires once and ``run`` returns."""
    calls: list[None] = []
    watcher = PreemptWatcher(
        on_preempt=lambda: calls.append(None),
        poll=_FAST_POLL,
        metadata_fetcher=_make_fetcher(["", "FALSE", "TRUE", "TRUE"]),
        is_gcp_vm=lambda: True,
    )
    watcher.run(stop_event=threading.Event())

    assert len(calls) == 1


def test_metadata_errors_never_latch() -> None:
    """Repeated ``None`` (fetch errors) keep polling; callback is never fired."""
    calls: list[None] = []
    stop_event = threading.Event()
    # Five Nones then we set the stop event to break the loop.
    fetcher = _make_fetcher([None, None, None, None, None], stop_event=stop_event)
    watcher = PreemptWatcher(
        on_preempt=lambda: calls.append(None),
        poll=_FAST_POLL,
        metadata_fetcher=fetcher,
        is_gcp_vm=lambda: True,
    )
    watcher.run(stop_event=stop_event)

    assert calls == []


@pytest.mark.parametrize("body", ["", "FALSE", "false", "True", "true", "yes", "1"])
def test_non_true_bodies_never_latch(body: str) -> None:
    """Only the exact ``"TRUE"`` body latches; case and synonyms are ignored."""
    calls: list[None] = []
    stop_event = threading.Event()
    fetcher = _make_fetcher([body, body, body], stop_event=stop_event)
    watcher = PreemptWatcher(
        on_preempt=lambda: calls.append(None),
        poll=_FAST_POLL,
        metadata_fetcher=fetcher,
        is_gcp_vm=lambda: True,
    )
    watcher.run(stop_event=stop_event)

    assert calls == []


def test_external_stop_event_returns_without_firing() -> None:
    """``stop_event`` set before entering the loop returns immediately without polling."""
    calls: list[None] = []
    fetcher_calls: list[str] = []

    def fetcher(key: str) -> str | None:
        fetcher_calls.append(key)
        return "TRUE"

    stop_event = threading.Event()
    stop_event.set()
    watcher = PreemptWatcher(
        on_preempt=lambda: calls.append(None),
        poll=_FAST_POLL,
        metadata_fetcher=fetcher,
        is_gcp_vm=lambda: True,
    )
    watcher.run(stop_event=stop_event)

    assert calls == []
    assert fetcher_calls == []
