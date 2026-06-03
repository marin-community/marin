# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fakeray: a Ray-Core-compatible execution shim backed by Fray/Iris.

Implements the subset of the Ray Core API that smallpond uses
(``init``/``shutdown``/``put``/``get``/``wait``/``remote``/``ObjectRef`` +
``exceptions``), dispatching each remote call to a pool of Fray actors instead
of a real Ray cluster.

Two ways to put this in front of an ``import ray``:

1. **uv override (recommended, no source edits).** Install the sibling
   distribution named ``ray`` (under ``ray-shim/``) and force it in via
   ``[tool.uv] override-dependencies = ["ray @ <wheel>"]``. The requirer's
   ``ray>=x`` constraint is replaced and ``import ray`` resolves to this shim.
   See ``lib/fakeray/README.md``.

2. **Runtime install.** Call ``fakeray.install()`` before the requirer imports
   ``ray``; it registers this module (and ``fakeray.exceptions``) under the
   ``ray`` name in ``sys.modules``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from fray.current_client import current_client

from fakeray import exceptions, util
from fakeray._actor import ActorClass, ActorHandle
from fakeray._actor import get_actor as _get_actor
from fakeray._actor import kill as _kill
from fakeray._object_ref import ObjectRef
from fakeray._scheduler import FakeRayConfig, Scheduler

logger = logging.getLogger(__name__)

__all__ = [
    "ActorClass",
    "ActorHandle",
    "ObjectRef",
    "RemoteFunction",
    "cancel",
    "exceptions",
    "get",
    "get_actor",
    "get_gpu_ids",
    "init",
    "install",
    "is_initialized",
    "kill",
    "put",
    "remote",
    "set_config",
    "shutdown",
    "timeline",
    "util",
    "wait",
]

# Module-global scheduler, created by init(). One per driver process — matching
# Ray's single-cluster-per-process model.
_scheduler: Scheduler | None = None
_config_override: FakeRayConfig | None = None


def set_config(config: FakeRayConfig) -> None:
    """Override pool sizing/resources before ``init`` (instead of env vars)."""
    global _config_override
    _config_override = config


def is_initialized() -> bool:
    return _scheduler is not None


class _InitResult:
    """Return value of ``init``; smallpond reads ``.address_info['gcs_address']``."""

    def __init__(self, address: str):
        self.address_info = {"gcs_address": address, "address": address}


def init(
    address: str | None = None,
    *,
    num_cpus: int | None = None,
    runtime_env: dict | None = None,
    **_ignored: Any,
) -> _InitResult:
    """Start the Fray-backed scheduler + actor pool.

    Ray-specific kwargs (``_memory``, ``dashboard_host``, ``dashboard_port``,
    ``_metrics_export_port``, ``log_to_driver``, …) are accepted and ignored.
    """
    global _scheduler
    if _scheduler is not None:
        logger.info("fakeray.init: already initialized; reusing scheduler")
        return _InitResult("fakeray://existing")

    config = _config_override or FakeRayConfig.from_env(num_cpus=num_cpus)
    client = current_client()
    logger.info("fakeray.init: client=%s config=%s", type(client).__name__, config)
    _scheduler = Scheduler(client, config, runtime_env=runtime_env)
    _scheduler.start()
    return _InitResult("fakeray://local")


def shutdown() -> None:
    global _scheduler
    if _scheduler is None:
        return
    _scheduler.shutdown()
    _scheduler = None


def _require() -> Scheduler:
    if _scheduler is None:
        raise RuntimeError("fakeray not initialized; call fakeray.init() first")
    return _scheduler


def put(value: Any) -> ObjectRef:
    return _require().put(value)


def get(refs: ObjectRef | list[ObjectRef], timeout: float | None = None) -> Any:
    scalar = isinstance(refs, ObjectRef)
    ref_list = [refs] if scalar else list(refs)
    out = _require().get(ref_list, timeout=timeout)
    return out[0] if scalar else out


def wait(
    refs: list[ObjectRef],
    *,
    num_returns: int = 1,
    timeout: float | None = None,
    fetch_local: bool = True,  # accepted for Ray-compat; values are descriptors
) -> tuple[list[ObjectRef], list[ObjectRef]]:
    return _require().wait(list(refs), num_returns=num_returns, timeout=timeout)


def timeline(filename: str) -> None:
    """Write an empty Chrome-trace timeline.

    fakeray has no Ray timeline to export (use the Iris dashboard for real
    observability), but callers like smallpond immediately re-open the file and
    ``json.load`` it, so we must produce a valid empty trace rather than no file.
    """
    import json

    try:
        with open(filename, "w") as f:
            json.dump([], f)
    except OSError as e:
        logger.debug("fakeray.timeline(%s): could not write placeholder: %s", filename, e)


class RemoteFunction:
    """What ``@ray.remote`` returns. Supports ``.options(...).remote(...)``."""

    def __init__(self, fn: Any, opts: dict | None = None):
        self._fn = fn
        self._opts = dict(opts or {})

    def options(self, **opts: Any) -> RemoteFunction:
        merged = {**self._opts, **opts}
        return RemoteFunction(self._fn, merged)

    def remote(self, *args: Any, **kwargs: Any) -> ObjectRef:
        return _require().submit_task(self._fn, args, kwargs, self._opts)


def _wrap_remote(target: Any, opts: dict) -> Any:
    """Class -> stateful ActorClass; function -> RemoteFunction."""
    if isinstance(target, type):
        return ActorClass(target, opts)
    return RemoteFunction(target, opts)


def remote(fn: Any = None, **opts: Any) -> Any:
    """``@ray.remote`` decorator for both functions and actor classes.

    Bare ``@ray.remote`` or ``@ray.remote(num_gpus=1, ...)``, applied to a def
    (-> RemoteFunction) or a class (-> ActorClass).
    """
    if fn is not None and not opts:
        return _wrap_remote(fn, {})

    def wrap(target: Any) -> Any:
        return _wrap_remote(target, opts)

    return wrap


def get_actor(name: str) -> ActorHandle:
    """Resolve a named actor (Ray-compat)."""
    return _get_actor(name)


def kill(actor: ActorHandle) -> None:
    """Terminate an actor (Ray-compat)."""
    _kill(actor)


def cancel(ref: ObjectRef) -> None:
    """Best-effort task cancel (Ray-compat). No-op once dispatched."""
    logger.debug("fakeray.cancel(%s): best-effort no-op", getattr(ref, "id", ref))


def get_gpu_ids() -> list:
    """Ray-compat ``ray.get_gpu_ids()`` — GPU ids assigned to this worker.

    Reads ``CUDA_VISIBLE_DEVICES`` if Iris set it for a GPU task; otherwise
    returns ``[]`` (CPU). Callers that index ``[0]`` unconditionally (e.g.
    SkyRL's InfoActor) only run under GPU placement, where this is populated.
    """
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda:
        return []
    return [int(x) if x.isdigit() else x for x in cuda.split(",") if x]


def install() -> None:
    """Register this shim under the ``ray`` name in ``sys.modules``.

    Call before the requiring library imports ``ray``. Also registers the
    ``ray.exceptions`` submodule, because ``import ray.exceptions`` (which
    smallpond does) is NOT served by a module-level ``__getattr__`` — Python
    resolves dotted imports through ``sys.modules`` entries directly.
    """
    import importlib

    # Import the SUBMODULES by full path. (A plain `from fakeray.util import
    # placement_group` would bind the re-exported *function* of that name in
    # util/__init__, not the module — which then can't satisfy
    # `from ray.util.placement_group import placement_group`.)
    pg_mod = importlib.import_module("fakeray.util.placement_group")
    ss_mod = importlib.import_module("fakeray.util.scheduling_strategies")

    sys.modules["ray"] = sys.modules[__name__]
    sys.modules["ray.exceptions"] = exceptions
    # Dotted submodule imports resolve through sys.modules, not __getattr__, so
    # register each ray.util.* path SkyRL imports explicitly.
    sys.modules["ray.util"] = util
    sys.modules["ray.util.placement_group"] = pg_mod
    sys.modules["ray.util.scheduling_strategies"] = ss_mod
    logger.info("fakeray.install: registered shim as 'ray' (+ ray.util.*) in sys.modules")
