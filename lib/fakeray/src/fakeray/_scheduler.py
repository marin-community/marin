# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fakeray DAG scheduler.

A ready-queue scheduler that runs in the driver process and dispatches each
remote call to a fixed pool of Fray actors. Design choices:

- **One worker thread per actor slot.** Each thread pulls a ready node, claims
  its dedicated actor, sends the payload, and blocks on the ``ActorFuture``.
  This relies only on ``ActorFuture.result()`` — the single method the Fray
  ``ActorFuture`` protocol guarantees — so it works identically on the local
  and Iris backends without depending on ``.done()`` or done-callbacks.

- **Dependencies are ObjectRefs passed as args.** A node is *ready* when every
  dep ref it references has completed. The scheduler dereferences ObjectRef
  args to concrete values just before dispatch, reproducing Ray's auto-deref.

- **Futures live in the driver.** ``ObjectRef.future`` is a
  ``concurrent.futures.Future`` the scheduler owns, so ``get``/``wait`` use
  ordinary future semantics regardless of backend.

Failure propagation: when a node fails, its exception is set on its ObjectRef
*and* propagated to all transitive descendants, so ``get`` on a downstream ref
raises instead of hanging forever.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from collections import defaultdict
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
from fray.client import Client
from fray.types import ActorConfig, ResourceConfig

from fakeray._executor import FakeRayExecutor
from fakeray._object_ref import ObjectRef
from fakeray.exceptions import GetTimeoutError

logger = logging.getLogger(__name__)

_POISON = object()  # sentinel pushed to the ready queue to stop worker threads


@dataclass
class FakeRayConfig:
    """Pool sizing and per-actor resources. Read once at ``init``.

    ``device`` selects the worker hardware for the actor pool. ``"cpu"`` uses a
    plain CPU ResourceConfig; a TPU variant string (e.g. ``"v6e-4"``) borrows
    that TPU host's VM for CPU-bound duckdb work (the chips sit idle). With
    ``preemptible=True`` and a TPU device the pool lands on the cheap
    preemptible TPU groups — pair with ``max_task_retries`` so the scheduler
    survives actor preemption (see Scheduler re-dispatch).
    """

    pool_size: int = 4
    cpu: float = 1.0
    ram: str = "4g"
    device: str = "cpu"
    preemptible: bool = True
    region: str | None = None
    max_task_retries: int = 3

    @staticmethod
    def from_env(num_cpus: int | None = None) -> FakeRayConfig:
        pool = int(os.environ.get("FAKERAY_POOL_SIZE", num_cpus or 4))
        cpu = float(os.environ.get("FAKERAY_ACTOR_CPU", "1"))
        ram = os.environ.get("FAKERAY_ACTOR_RAM", "4g")
        device = os.environ.get("FAKERAY_DEVICE", "cpu")
        region = os.environ.get("FAKERAY_REGION") or None
        preemptible = os.environ.get("FAKERAY_PREEMPTIBLE", "1") != "0"
        return FakeRayConfig(
            pool_size=max(1, pool), cpu=cpu, ram=ram, device=device, region=region, preemptible=preemptible
        )

    def resource_config(self) -> ResourceConfig:
        """Build the Fray ResourceConfig for one actor in the pool."""
        regions = [self.region] if self.region else None
        if self.device and self.device != "cpu":
            # TPU host VM: with_tpu sets sensible cpu/ram defaults for the host.
            return ResourceConfig.with_tpu(self.device, preemptible=self.preemptible, regions=regions)
        return ResourceConfig(cpu=self.cpu, ram=self.ram, preemptible=self.preemptible, regions=regions)


@dataclass
class _Node:
    """A registered remote call awaiting (or undergoing) execution."""

    ref: ObjectRef
    payload_fn: Any
    args: tuple
    kwargs: dict
    opts: dict
    pending: set[str] = field(default_factory=set)  # dep ref ids not yet completed
    queued: bool = False
    attempts: int = 0  # dispatch attempts so far (for preemption re-dispatch)


class Scheduler:
    """Owns the Fray actor pool and drives the task DAG to completion."""

    def __init__(self, client: Client, config: FakeRayConfig, *, runtime_env: dict | None = None):
        self._client = client
        self._config = config
        self._runtime_env = runtime_env or {}

        self._lock = threading.Lock()
        self._nodes: dict[str, _Node] = {}
        self._children: dict[str, list[str]] = defaultdict(list)
        self._completed: set[str] = set()  # ids whose value/exc is settled (nodes + puts)
        self._ready: queue.SimpleQueue = queue.SimpleQueue()

        self._group = None
        self._workers: list[threading.Thread] = []
        self._started = False

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        n = self._config.pool_size
        resources = self._config.resource_config()
        logger.info(
            "fakeray: starting actor pool of %d (device=%s preemptible=%s region=%s)",
            n,
            self._config.device,
            self._config.preemptible,
            self._config.region,
        )
        self._group = self._client.create_actor_group(
            FakeRayExecutor,
            name="fakeray-exec",
            count=n,
            resources=resources,
            # Auto-restart preempted actors; bound task retries so the scheduler's
            # re-dispatch (on actor death) doesn't loop forever.
            actor_config=ActorConfig(max_concurrency=1, max_task_retries=self._config.max_task_retries),
        )
        handles = self._group.wait_ready(count=n)
        for i, handle in enumerate(handles):
            t = threading.Thread(target=self._worker_loop, args=(handle,), name=f"fakeray-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        self._started = True

    def shutdown(self) -> None:
        for _ in self._workers:
            self._ready.put(_POISON)
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers = []
        if self._group is not None:
            self._group.shutdown()
            self._group = None
        self._started = False

    # ---- public object-store ops ---------------------------------------

    def put(self, value: Any) -> ObjectRef:
        fut: Future = Future()
        fut.set_result(value)
        ref = ObjectRef(id=uuid.uuid4().hex, future=fut)
        with self._lock:
            self._completed.add(ref.id)
        return ref

    def submit_task(self, fn: Any, args: tuple, kwargs: dict, opts: dict) -> ObjectRef:
        """Register a remote call; return its ObjectRef immediately."""
        ref = ObjectRef(id=uuid.uuid4().hex, future=Future())
        node = _Node(ref=ref, payload_fn=fn, args=tuple(args), kwargs=dict(kwargs), opts=dict(opts))

        dep_ids = [a.id for a in (*args, *kwargs.values()) if isinstance(a, ObjectRef)]
        with self._lock:
            for dep_id in dep_ids:
                if dep_id in self._completed:
                    continue  # put-ref or already-finished node — contributes no edge
                node.pending.add(dep_id)
                self._children[dep_id].append(ref.id)
            self._nodes[ref.id] = node
            ready_now = not node.pending
            if ready_now:
                node.queued = True
        if ready_now:
            self._ready.put(node)
        return ref

    # ---- get / wait -----------------------------------------------------

    def get(self, refs: list[ObjectRef], timeout: float | None) -> list[Any]:
        out: list[Any] = []
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        for ref in refs:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                out.append(ref.future.result(timeout=remaining))
            except FuturesTimeout as e:
                raise GetTimeoutError(f"get() timed out after {timeout}s") from e
        return out

    def wait(
        self, refs: list[ObjectRef], *, num_returns: int, timeout: float | None
    ) -> tuple[list[ObjectRef], list[ObjectRef]]:
        import time

        deadline = None if timeout is None else time.monotonic() + timeout

        def _done(r: Any) -> bool:
            return isinstance(r, ObjectRef) and r.future.done()

        while True:
            ready = [r for r in refs if _done(r)]
            not_ready = [r for r in refs if not _done(r)]
            if len(ready) >= num_returns:
                break
            if timeout == 0 or (deadline is not None and time.monotonic() >= deadline):
                break
            time.sleep(0.01)
        return ready, not_ready

    # ---- worker loop ----------------------------------------------------

    def _worker_loop(self, actor_handle: Any) -> None:
        while True:
            item = self._ready.get()
            if item is _POISON:
                return
            node: _Node = item
            if node.ref.future.done():
                continue  # poisoned by a failed dependency before we got to it
            node.attempts += 1
            try:
                values = [self._deref(a) for a in node.args]
                kwargs = {k: self._deref(v) for k, v in node.kwargs.items()}
                payload = cloudpickle.dumps((node.payload_fn, tuple(values), kwargs))
                result_bytes = actor_handle.run.remote(payload).result()
                result = cloudpickle.loads(result_bytes)
            except Exception as e:
                # An exception here is either a real task error or a dead actor
                # (preemption). We can't always tell them apart, so retry up to
                # max_task_retries: smallpond tasks are idempotent (shared-root
                # markers + unique output keys), so re-running a genuinely failing
                # task just re-fails deterministically and settles after the cap.
                # A surviving worker thread picks up the re-queued node.
                if node.attempts <= self._config.max_task_retries:
                    logger.warning(
                        "fakeray: task %s attempt %d failed (%s); re-dispatching",
                        node.ref.id[:8],
                        node.attempts,
                        type(e).__name__,
                    )
                    self._ready.put(node)
                else:
                    logger.warning(
                        "fakeray: task %s failed after %d attempts: %s",
                        node.ref.id[:8],
                        node.attempts,
                        e,
                    )
                    self._settle_failure(node.ref.id, e)
                continue
            self._settle_success(node.ref.id, result)

    @staticmethod
    def _deref(value: Any) -> Any:
        if isinstance(value, ObjectRef):
            return value.future.result()  # node is ready => already resolved
        return value

    def _settle_success(self, node_id: str, value: Any) -> None:
        with self._lock:
            self._completed.add(node_id)
            newly_ready = self._unblock_children_locked(node_id)
        self._nodes[node_id].ref.future.set_result(value)
        for n in newly_ready:
            self._ready.put(n)

    def _settle_failure(self, node_id: str, exc: BaseException) -> None:
        """Fail this node and poison all transitive descendants."""
        to_fail = [node_id]
        seen: set[str] = set()
        while to_fail:
            nid = to_fail.pop()
            if nid in seen:
                continue
            seen.add(nid)
            with self._lock:
                self._completed.add(nid)
                child_ids = list(self._children.get(nid, []))
            node = self._nodes.get(nid)
            if node is not None and not node.ref.future.done():
                node.ref.future.set_exception(exc)
            to_fail.extend(child_ids)

    def _unblock_children_locked(self, done_id: str) -> list[_Node]:
        newly_ready: list[_Node] = []
        for child_id in self._children.get(done_id, []):
            child = self._nodes.get(child_id)
            if child is None or child.queued or child.ref.future.done():
                continue
            child.pending.discard(done_id)
            if not child.pending:
                child.queued = True
                newly_ready.append(child)
        return newly_ready
