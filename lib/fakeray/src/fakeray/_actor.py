# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray-compatible stateful actors, backed by Fray actors.

Ray's actor model: ``@ray.remote`` on a class yields an ``ActorClass``;
``.remote(*args)`` constructs the actor (a process) and returns an
``ActorHandle``; ``handle.method.remote(*a)`` enqueues a call and returns an
``ObjectRef``; ``ray.get(ref)`` awaits it.

We map each of those onto Fray:
- ``ActorClass.remote()`` -> ``Client.create_actor(...)`` -> a Fray ActorHandle
- ``handle.method.remote(*a)`` -> Fray ``handle.method.remote(*a)`` (an
  ``ActorFuture``), adapted into a fakeray ``ObjectRef`` so ``get``/``wait``
  work uniformly with task refs.

Resource options on ``.options(num_gpus=, num_cpus=, memory=, resources=, ...)``
become a Fray ``ResourceConfig``; ``scheduling_strategy=PlacementGroupScheduling
Strategy(...)`` is unwrapped to the bundle's resources (best-effort — see the
placement-group shim and the design note).
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future
from typing import Any

from fray.types import ActorConfig, CpuConfig, GpuConfig, ResourceConfig

from fakeray._object_ref import ObjectRef

# process-global registry so ray.get_actor(name) can resolve handles created in
# this driver. Cross-process named lookup defers to Fray's resolver (not v1).
_actor_registry: dict[str, ActorHandle] = {}


def _resources_from_options(opts: dict) -> ResourceConfig:
    """Map Ray ``.options(...)`` kwargs to a Fray ResourceConfig.

    Reads ``num_cpus``/``num_gpus``/``memory`` and unwraps a
    ``PlacementGroupSchedulingStrategy`` into the target bundle's resources.
    Other Ray option keys are ignored (Ray accepts many; we map only the ones
    that affect Iris placement).
    """
    num_gpus = opts.get("num_gpus")
    num_cpus = opts.get("num_cpus")
    memory = opts.get("memory")

    # A placement-group bundle, if one was passed via scheduling_strategy.
    strategy = opts.get("scheduling_strategy")
    bundle = getattr(strategy, "_bundle", None)
    if bundle is not None:
        num_gpus = bundle.get("GPU", num_gpus)
        num_cpus = bundle.get("CPU", num_cpus)

    cpu = float(num_cpus) if num_cpus is not None else 1.0
    ram = _as_ram(memory)

    if num_gpus and num_gpus >= 1:
        # whole-GPU actor; fractional GPU (colocation) is not supported on Iris.
        return ResourceConfig(device=GpuConfig(variant="auto", count=int(num_gpus)), cpu=max(cpu, 1.0), ram=ram)
    return ResourceConfig(device=CpuConfig(), cpu=cpu, ram=ram)


def _as_ram(memory: Any) -> str:
    if memory is None:
        return "4g"
    if isinstance(memory, str):
        return memory
    # Ray takes bytes for `memory`; Fray wants a human string.
    return f"{int(int(memory) / (1024**3)) or 1}g"


class ActorMethodStub:
    """``handle.method`` — call ``.remote(*a)`` to invoke it."""

    def __init__(self, handle: ActorHandle, method: str):
        self._handle = handle
        self._method = method

    def remote(self, *args: Any, **kwargs: Any) -> ObjectRef:
        fray_future = getattr(self._handle._fray_handle, self._method).remote(*args, **kwargs)
        return _future_to_ref(fray_future)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Ray disallows direct calls on actor methods; mirror that.
        raise TypeError(f"actor methods must be invoked via .remote(); got {self._method}(...)")


def _future_to_ref(fray_future: Any) -> ObjectRef:
    """Adapt a Fray ActorFuture into a fakeray ObjectRef.

    A small daemon thread blocks on ``ActorFuture.result()`` (the one method the
    Fray protocol guarantees) and resolves a ``concurrent.futures.Future`` that
    backs the ObjectRef — so ``ray.get``/``ray.wait`` treat actor-method results
    exactly like task results.
    """
    fut: Future = Future()

    def _await() -> None:
        try:
            fut.set_result(fray_future.result())
        except Exception as e:
            fut.set_exception(e)

    threading.Thread(target=_await, daemon=True).start()
    return ObjectRef(id=uuid.uuid4().hex, future=fut)


class ActorHandle:
    """Handle to a live actor. ``handle.method.remote(*a)`` -> ObjectRef.

    Holds the backing Fray actor *group* (created with count=1) so the actor can
    be torn down via ``ray.kill`` — the per-actor handle itself has no terminate
    method, and on the Iris backend ``getattr(handle, "shutdown")`` would
    resolve to a *remote method call* (NotFound), not teardown.
    """

    def __init__(self, fray_handle: Any, name: str, group: Any = None):
        self._fray_handle = fray_handle
        self._name = name
        self._group = group

    def __getattr__(self, method: str) -> ActorMethodStub:
        if method.startswith("_"):
            raise AttributeError(method)
        return ActorMethodStub(self, method)


class ActorClass:
    """What ``@ray.remote`` returns for a class. Construct via ``.remote(*a)``."""

    def __init__(self, cls: type, opts: dict | None = None):
        self._cls = cls
        self._opts = dict(opts or {})

    def options(self, **opts: Any) -> ActorClass:
        return ActorClass(self._cls, {**self._opts, **opts})

    def remote(self, *args: Any, **kwargs: Any) -> ActorHandle:
        # Imported lazily to avoid a module cycle (scheduler -> actor -> client).
        from fray.current_client import current_client

        name = self._opts.get("name") or f"{self._cls.__name__}-{uuid.uuid4().hex[:8]}"
        resources = _resources_from_options(self._opts)
        client = current_client()
        # Create a 1-actor group (not create_actor) so we retain the group, which
        # is the only object exposing teardown (.shutdown) for ray.kill.
        group = client.create_actor_group(
            self._cls,
            *args,
            name=name,
            count=1,
            resources=resources,
            actor_config=ActorConfig(max_concurrency=self._opts.get("max_concurrency", 1)),
            **kwargs,
        )
        fray_handle = group.wait_ready(count=1)[0]
        handle = ActorHandle(fray_handle, name, group=group)
        _actor_registry[name] = handle
        return handle


def get_actor(name: str) -> ActorHandle:
    """Resolve a named actor created in this driver."""
    handle = _actor_registry.get(name)
    if handle is None:
        raise ValueError(f"Failed to look up actor with name '{name}'")
    return handle


def kill(handle: ActorHandle) -> None:
    """Terminate an actor (best-effort) via its backing Fray actor group.

    NB: must go through the group's ``shutdown`` — NOT ``getattr(handle,
    "shutdown")``, which on the Iris backend resolves to a remote *method call*
    (``Method 'shutdown' not found``) rather than teardown.
    """
    group = handle._group
    if group is not None and hasattr(group, "shutdown"):
        group.shutdown()
    _actor_registry.pop(handle._name, None)
