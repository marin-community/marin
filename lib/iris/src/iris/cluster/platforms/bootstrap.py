# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap lifecycle tracking for slices that boot asynchronously.

A provider creates a slice handle and returns it immediately, then runs
bootstrap (image pull, worker start, health probe) on a background thread.
``SliceBootstrap`` is the thread-safe handoff between that thread and
``describe()``: the bootstrap thread records the terminal verdict, and
``slice_status_with_bootstrap`` composes it with the cloud-reported state.
"""

import threading
from dataclasses import dataclass, replace

from iris.cluster.platforms.types import CloudSliceState, SliceStatus


@dataclass(frozen=True)
class BootstrapStatus:
    """Snapshot of a slice's bootstrap verdict.

    ``state`` is ``None`` while bootstrap is still running, ``READY`` once the
    workers answered a health probe, and ``FAILED`` once it definitively gave up.
    """

    state: CloudSliceState | None
    error: str


class SliceBootstrap:
    """Thread-safe bootstrap verdict for a single slice."""

    def __init__(self, *, bootstrapping: bool) -> None:
        """Track a slice whose bootstrap is monitored when *bootstrapping* is set.

        A slice created without bootstrap monitoring — no worker config, or one
        rediscovered on controller restart — starts at the ``READY`` sentinel so
        composite state falls through to the raw cloud state.
        """
        self._state: CloudSliceState | None = None if bootstrapping else CloudSliceState.READY
        self._error = ""
        self._lock = threading.Lock()

    def mark_ready(self) -> None:
        with self._lock:
            self._state = CloudSliceState.READY
            self._error = ""

    def mark_failed(self, error: str) -> None:
        """Record a terminal bootstrap failure and its reason.

        The reason (e.g. the create-LRO "no more capacity" stockout) is surfaced
        through ``describe()`` so the autoscaler can classify the outcome rather
        than lose it.
        """
        with self._lock:
            self._state = CloudSliceState.FAILED
            self._error = error

    def status(self) -> BootstrapStatus:
        with self._lock:
            return BootstrapStatus(state=self._state, error=self._error)


def composite_slice_state(
    cloud_state: CloudSliceState,
    bootstrap_state: CloudSliceState | None,
) -> CloudSliceState:
    """Compose cloud lifecycle with bootstrap lifecycle into effective slice state.

    Worker health is canonical for liveness: once bootstrap is READY the slice
    is READY even if the cloud-reported state still lags at CREATING — a TPU
    can boot and serve long before its create operation flips to READY. A slice
    rediscovered on controller restart (no bootstrap monitoring) carries a READY
    bootstrap sentinel, so it too becomes READY and is validated by the
    autoscaler's health probe, which reaps it if the workers are in fact dead.

    A bootstrap that definitively FAILED is authoritative — even when the cloud
    is no longer describable (UNKNOWN). A stockout/quota create failure leaves no
    resource to describe, so describe() reports UNKNOWN; without surfacing the
    FAILED bootstrap verdict the autoscaler would wait out the full
    unresolvable-timeout grace period (and keep re-describing the dead slice)
    before reaping a slice it already knows never came up.

    Cloud states that mean "gone or doomed" — FAILED and DELETING — are likewise
    authoritative. A bare UNKNOWN (cloud no longer describable, bootstrap still in
    progress or a stale READY sentinel) is reported as UNKNOWN so a vanished node
    never lingers as READY and the unresolvable-timeout path can reap it.
    """
    if cloud_state in (CloudSliceState.FAILED, CloudSliceState.DELETING):
        return cloud_state
    # A definitive bootstrap failure wins over UNKNOWN: the slice never came up,
    # so reap it now instead of waiting out the unresolvable timeout.
    if bootstrap_state == CloudSliceState.FAILED:
        return CloudSliceState.FAILED
    if cloud_state == CloudSliceState.UNKNOWN:
        return CloudSliceState.UNKNOWN
    if bootstrap_state == CloudSliceState.READY:
        return CloudSliceState.READY
    # Bootstrap still in progress: surface BOOTSTRAPPING once cloud is READY,
    # otherwise reflect the raw cloud state (CREATING/REPAIRING).
    if cloud_state == CloudSliceState.READY:
        return CloudSliceState.BOOTSTRAPPING
    return cloud_state


def slice_status_with_bootstrap(cloud_status: SliceStatus, bootstrap: SliceBootstrap) -> SliceStatus:
    """Overlay *bootstrap*'s verdict on a raw cloud status, keeping its worker handles."""
    verdict = bootstrap.status()
    state = composite_slice_state(cloud_status.state, verdict.state)
    return replace(
        cloud_status,
        state=state,
        error_message=verdict.error if state == CloudSliceState.FAILED else "",
    )
