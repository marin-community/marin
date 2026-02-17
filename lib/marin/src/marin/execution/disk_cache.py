# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable disk caching and distributed locking for step functions.

Two main primitives:

- ``disk_cached`` - run a function once, cache the result as an ``Artifact``,
  and return the cached value on subsequent calls.
- ``distributed_lock`` - decorator that wraps a ``fn(output_path) -> T`` with
  lease-based distributed locking via ``StatusFile``.

They compose naturally::

    tokenizer = disk_cached(
        "tokenizer",
        distributed_lock(get_tokenizer),
        hash_attrs={"model": "gpt2"},
    )
"""

from __future__ import annotations

import functools
import logging
import typing
from collections.abc import Callable
from threading import Event, Thread
from typing import Any, TypeVar

from marin.execution.artifact import Artifact
from marin.execution.executor_step_status import (
    HEARTBEAT_INTERVAL,
    STATUS_SUCCESS,
    StatusFile,
)
from marin.execution.step_model import StepSpec
from marin.execution.step_runner import should_run, worker_id

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StepAlreadyDone(Exception):
    """Raised by ``distributed_lock`` when the step has already succeeded."""


def _infer_artifact_type(fn: Callable, explicit: type[T] | None) -> type[T]:
    """Return the artifact type to use for deserialization.

    If *explicit* is provided it wins. Otherwise the return annotation of *fn*
    is used.  Raises ``TypeError`` when neither is available (common for
    lambdas which cannot carry return annotations).
    """
    if explicit is not None:
        return explicit
    hints = typing.get_type_hints(fn)
    ret = hints.get("return")
    if ret is None:
        raise TypeError(
            f"Cannot infer artifact_type from {fn!r}. "
            "Annotate the function's return type or pass artifact_type explicitly."
        )
    return ret


def distributed_lock(fn: Callable[[str], T]) -> Callable[[str], T]:
    """Decorator: wrap *fn* with lease-based distributed locking.

    The lock is keyed on the *output_path* argument passed to *fn*.  If
    another worker already completed the step (``STATUS_SUCCESS``),
    ``StepAlreadyDone`` is raised so that the caller (typically
    ``disk_cached``) can load the cached artifact instead.

    While *fn* is executing a heartbeat thread refreshes the lock so that
    other workers see it as active.

    This decorator does **not** write status or save artifacts - that is the
    responsibility of the caller.
    """

    @functools.wraps(fn)
    def wrapper(output_path: str) -> T:
        status_file = StatusFile(output_path, worker_id())
        step_label = output_path.rsplit("/", 1)[-1]

        if not should_run(status_file, step_label):
            raise StepAlreadyDone(output_path)

        stop_event = Event()

        def _heartbeat():
            while not stop_event.wait(HEARTBEAT_INTERVAL):
                status_file.refresh_lock()

        heartbeat_thread = Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            return fn(output_path)
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=5)
            status_file.release_lock()

    return wrapper


def disk_cached(
    name: str,
    fn: Callable[[str], T],
    *,
    hash_attrs: dict[str, Any] | None = None,
    output_path_prefix: str | None = None,
    artifact_type: type[T] | None = None,
) -> T:
    """Run *fn* once, cache the result as an :class:`Artifact`, return it.

    On subsequent calls with the same *name* + *hash_attrs* the cached
    artifact is loaded directly without calling *fn*.

    ``artifact_type`` is inferred from *fn*'s return annotation when
    possible.  For lambdas (which cannot carry annotations) you must pass
    it explicitly.

    Composes with :func:`distributed_lock`::

        result = disk_cached(
            "tokenizer",
            distributed_lock(get_tokenizer),
            hash_attrs={"model": "gpt2"},
        )
    """
    resolved_type: type[T] = _infer_artifact_type(fn, artifact_type)

    spec = StepSpec(
        name=name,
        hash_attrs=hash_attrs or {},
        output_path_prefix=output_path_prefix,
    )
    output_path = spec.output_path

    status_file = StatusFile(output_path, worker_id())
    if status_file.status == STATUS_SUCCESS:
        logger.info(f"disk_cached: cache hit for {spec.name_with_hash}")
        return Artifact.load(output_path, resolved_type)

    try:
        result = fn(output_path)
    except StepAlreadyDone:
        logger.info(f"disk_cached: completed by another worker for {spec.name_with_hash}")
        return Artifact.load(output_path, resolved_type)

    Artifact.save(result, output_path)
    StatusFile(output_path, worker_id()).write_status(STATUS_SUCCESS)
    logger.info(f"disk_cached: computed and cached {spec.name_with_hash}")
    return result
