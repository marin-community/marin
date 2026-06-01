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
import sys
from typing import Any

from fray.current_client import current_client

from fakeray import exceptions
from fakeray._object_ref import ObjectRef
from fakeray._scheduler import FakeRayConfig, Scheduler

logger = logging.getLogger(__name__)

__all__ = [
    "ObjectRef",
    "RemoteFunction",
    "exceptions",
    "get",
    "init",
    "install",
    "is_initialized",
    "put",
    "remote",
    "set_config",
    "shutdown",
    "timeline",
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
    """Ray-compat no-op. Iris provides its own dashboard/timeline."""
    logger.debug("fakeray.timeline(%s): no-op (use the Iris dashboard)", filename)


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


def remote(fn: Any = None, **opts: Any) -> Any:
    """``@ray.remote`` and ``@ray.remote(num_cpus=..., ...)`` decorator."""
    if fn is not None and callable(fn) and not opts:
        return RemoteFunction(fn)

    def wrap(f: Any) -> RemoteFunction:
        return RemoteFunction(f, opts)

    return wrap


def install() -> None:
    """Register this shim under the ``ray`` name in ``sys.modules``.

    Call before the requiring library imports ``ray``. Also registers the
    ``ray.exceptions`` submodule, because ``import ray.exceptions`` (which
    smallpond does) is NOT served by a module-level ``__getattr__`` — Python
    resolves dotted imports through ``sys.modules`` entries directly.
    """
    sys.modules["ray"] = sys.modules[__name__]
    sys.modules["ray.exceptions"] = exceptions
    logger.info("fakeray.install: registered shim as 'ray' in sys.modules")
