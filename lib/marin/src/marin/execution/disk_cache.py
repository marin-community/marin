# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a function once, cache the result, and return it on subsequent calls."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
from collections.abc import Callable
from typing import Generic, TypeVar, ParamSpec

import cloudpickle
import fsspec
from iris.temp_buckets import get_temp_bucket_path

from marin.execution.distributed_lock import StepAlreadyDone
from marin.execution.executor_step_status import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    StatusFile,
    worker_id,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class disk_cache(Generic[P, T]):
    """Decorator that caches function results to disk via a status file.

    Supports bare decoration, decoration with arguments, and direct wrapping::

        @disk_cache
        def my_fn(*args):
            ...

        @disk_cache(output_path="gs://...", save_fn=save, load_fn=load)
        def my_fn(*args):
            ...

        cached_fn = disk_cache(fn, output_path=path, save_fn=save, load_fn=load)
        result = cached_fn(output_path)

    When *save_fn* and *load_fn* are provided, the result is persisted and
    deserialized on cache hits.  When they are ``None`` (the default),
    cloudpickle is used.

    The decorated function is called with its original arguments; the
    *output_path* is used only for cache bookkeeping.

    By itself this class does not provide any locking guarantees.  Compose with
    :func:`distributed_lock` to ensure single-writer semantics.
    """

    def __init__(
        self,
        fn: Callable[P, T] | None = None,
        *,
        output_path: str | None = None,
        save_fn: Callable[[T, str], None] | None = None,
        load_fn: Callable[[str], T] | None = None,
    ):
        self._output_path = output_path
        self._save_fn = save_fn
        self._load_fn = load_fn
        self._fn: Callable[P, T] | None = None

        # When used as a bare decorator (@disk_cache without parentheses), Python passes
        # the decorated function as `fn` directly to __init__. Bind it immediately.
        if callable(fn):
            self._fn = fn
            functools.update_wrapper(self, fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self._fn is None:
            # @disk_cache(...) — receiving the function to wrap
            if len(args) == 1 and callable(args[0]) and not kwargs:  # pyrefly: ignore[unsupported-operation]
                self._fn = args[0]  # type: ignore[assignment]
                functools.update_wrapper(self, self._fn)
                return self  # type: ignore[return-value]
            raise TypeError("disk_cache() expected a callable")

        return self._execute(*args, **kwargs)

    def _execute(self, *args: P.args, **kwargs: P.kwargs) -> T:
        def fingerprint_args(*args, **kwargs) -> str:
            """Create a deterministic fingerprint for args and kwargs."""
            # Include module and qualified name to avoid collisions across
            # different functions (e.g. two functions both named "compute").
            fn_module = getattr(self._fn, "__module__", None)
            fn_qualname = getattr(self._fn, "__qualname__", None) or getattr(self._fn, "__name__", None)
            data = cloudpickle.dumps((fn_module, fn_qualname, args, sorted(kwargs.items())))
            return hashlib.sha256(data).hexdigest()[:16]

        output_path = self._output_path
        if output_path is None:
            args_fingerprint = fingerprint_args(*args, **kwargs)
            output_path = get_temp_bucket_path(1, f"disk_cache_{args_fingerprint}")
            if output_path is None:
                output_path = os.environ["MARIN_PREFIX"] + f"/disk_cache_{args_fingerprint}"

        def load_result() -> T:
            assert output_path is not None
            if self._load_fn is not None:
                return self._load_fn(output_path)
            with fsspec.open(output_path + "/data.pkl", "rb") as f:
                return cloudpickle.loads(f.read())

        status_file = StatusFile(output_path, worker_id())
        if status_file.status == STATUS_SUCCESS:
            logger.info(f"disk_cache: cache hit for {output_path}")
            return load_result()

        assert self._fn is not None
        try:
            result = self._fn(*args, **kwargs)
        except StepAlreadyDone:
            # NOTE: this is leaky but this branch handles the case of distributed lock wrapper
            logger.info(f"disk_cache: completed by another worker for {output_path}")
            return load_result()
        except Exception:
            StatusFile(output_path, worker_id()).write_status(STATUS_FAILED)
            raise

        if self._save_fn is not None:
            self._save_fn(result, output_path)
        else:
            with fsspec.open(output_path + "/data.pkl", "wb") as f:
                f.write(cloudpickle.dumps(result))

        StatusFile(output_path, worker_id()).write_status(STATUS_SUCCESS)
        logger.info(f"disk_cache: computed and cached {output_path}")
        return result
