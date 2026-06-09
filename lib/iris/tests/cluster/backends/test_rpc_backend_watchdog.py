# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounding policy for RpcTaskBackend control-path dispatches.

Every controller->backend call the control loop drives (reconcile, capacity
ops) runs on a dedicated pool under a fleet-size-aware watchdog, so a hung
worker fleet surfaces as a bounded error instead of pinning the control thread.
The one-off exec RPC keeps a generous (non-hour) deadline.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest
from iris.cluster.backends.rpc import backend as backend_mod
from iris.cluster.backends.rpc.backend import EXEC_IN_CONTAINER_MAX_TIMEOUT, RpcTaskBackend
from iris.cluster.controller.backend import (
    BackendReconcileInput,
    ProviderError,
    TaskTarget,
)
from iris.cluster.controller.reconcile.worker import WorkerReconcilePlan
from iris.cluster.types import WorkerId
from iris.rpc import worker_pb2
from rigging.timing import Duration

_W1 = "worker-1"
_W1_ADDR = "worker-1:8080"


def _make_plan(worker_id: str = _W1) -> WorkerReconcilePlan:
    return WorkerReconcilePlan(
        worker_id=WorkerId(worker_id),
        request=worker_pb2.Worker.ReconcileRequest(worker_id=worker_id),
    )


@dataclass
class _StubFactory:
    stub: object

    def get_stub(self, address: str) -> object:
        return self.stub

    def evict(self, address: str) -> None: ...

    def close(self) -> None: ...


def _short_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the watchdog constants so an overrun trips in well under a second."""
    monkeypatch.setattr(backend_mod, "DEFAULT_WORKER_RPC_TIMEOUT", Duration.from_seconds(0.05))
    monkeypatch.setattr(backend_mod, "CONTROL_DISPATCH_WATCHDOG_SLACK", Duration.from_seconds(0.05))


def test_watchdog_timeout_scales_with_fleet_size():
    backend = RpcTaskBackend(stub_factory=_StubFactory(stub=None), parallelism=128)
    try:
        # One batch (<=128 workers): per-worker timeout + slack.
        assert backend._watchdog_timeout(1) == pytest.approx(40.0)
        assert backend._watchdog_timeout(128) == pytest.approx(40.0)
        # Empty fleet still bounded by a single batch.
        assert backend._watchdog_timeout(0) == pytest.approx(40.0)
        # Two batches: a second per-worker timeout is added.
        assert backend._watchdog_timeout(129) == pytest.approx(50.0)
    finally:
        backend.close()


def test_reconcile_surfaces_hung_worker_as_error_within_cap(monkeypatch):
    """A worker that never responds must raise within the watchdog, not block."""
    _short_watchdog(monkeypatch)
    release = threading.Event()

    @dataclass
    class _HungStub:
        async def reconcile(self, request, *, timeout_ms=None):
            # Block the fan-out's event loop until the test releases it, modelling
            # a worker whose RPC never returns.
            release.wait()
            return worker_pb2.Worker.ReconcileResponse()

    backend = RpcTaskBackend(stub_factory=_StubFactory(stub=_HungStub()))
    try:
        batch = BackendReconcileInput(plans=[_make_plan()], worker_addresses={WorkerId(_W1): _W1_ADDR})
        with pytest.raises(ProviderError):
            backend.reconcile(batch)
    finally:
        release.set()
        backend.close()


def test_reconcile_returns_result_when_worker_responds():
    """A normal worker response flows back through the bounded dispatch."""
    observation = worker_pb2.Worker.AttemptObservation(attempt_uid="uid-a")

    @dataclass
    class _OkStub:
        async def reconcile(self, request, *, timeout_ms=None):
            return worker_pb2.Worker.ReconcileResponse(observed=[observation])

    backend = RpcTaskBackend(stub_factory=_StubFactory(stub=_OkStub()))
    try:
        batch = BackendReconcileInput(plans=[_make_plan()], worker_addresses={WorkerId(_W1): _W1_ADDR})
        result = backend.reconcile(batch)
    finally:
        backend.close()

    assert len(result.worker_results) == 1
    assert result.worker_results[0].error is None
    assert list(result.worker_results[0].observations) == [observation]


def test_exec_in_container_unlimited_uses_generous_cap_not_an_hour():
    """Negative (unlimited) timeout maps to the explicit cap, never the old hour."""

    @dataclass
    class _ExecStub:
        seen_timeout_ms: list[int] = field(default_factory=list)

        async def exec_in_container(self, request, *, timeout_ms):
            self.seen_timeout_ms.append(timeout_ms)
            return worker_pb2.Worker.ExecInContainerResponse()

    stub = _ExecStub()
    backend = RpcTaskBackend(stub_factory=_StubFactory(stub=stub))
    try:
        target = TaskTarget(task_id="t", attempt_id=0, worker_id=WorkerId(_W1), address=_W1_ADDR)
        backend.exec_in_container(target, worker_pb2.Worker.ExecInContainerRequest(), timeout_seconds=-1)
    finally:
        backend.close()

    assert stub.seen_timeout_ms == [EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms()]
    assert EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms() < 3_600_000
