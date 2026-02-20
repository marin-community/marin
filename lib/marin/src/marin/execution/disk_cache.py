# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a function once, cache the result, and return it on subsequent calls."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from marin.execution.distributed_lock import StepAlreadyDone
from marin.execution.executor_step_status import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    StatusFile,
    worker_id,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def disk_cached(
    fn: Callable[[str], T],
    output_path: str,
    *,
    save: Callable[[T, str], None] | None = None,
    load: Callable[[str], T] | None = None,
) -> T | None:
    """Run *fn* once and track completion via a status file.

    When *save* and *load* are provided, the result of *fn* is persisted and
    deserialized on cache hits.  When they are ``None`` (the default), *fn* is
    expected to handle its own reading/writing at *output_path* and is called
    again on cache hits to load the result.

    By itself this function does not provide any locking guarantees, e.g. if
    you run disk_cached on multiple workers/shards, there may be a race
    condition where multiple workers evaluate the function and write to the same
    output path at the same time. To avoid this, compose with
    :func:`distributed_lock` to ensure that only one worker executes the
    function while the others wait for the result. For example::

        result = disk_cached(
            distributed_lock(get_tokenizer),
            output_path,
            save=Artifact.save,
            load=lambda p: Artifact.load(p, TokenizerInfo),
        )
    """
    status_file = StatusFile(output_path, worker_id())
    if status_file.status == STATUS_SUCCESS:
        logger.info(f"disk_cached: cache hit for {output_path}")
        return load(output_path) if load is not None else fn(output_path)

    try:
        result = fn(output_path)
    except StepAlreadyDone:
        # NOTE: this is leaky but this branch handles the case of distributed lock wrapper
        logger.info(f"disk_cached: completed by another worker for {output_path}")
        return load(output_path) if load is not None else fn(output_path)
    except Exception:
        StatusFile(output_path, worker_id()).write_status(STATUS_FAILED)
        raise

    if save is not None:
        save(result, output_path)
    StatusFile(output_path, worker_id()).write_status(STATUS_SUCCESS)
    logger.info(f"disk_cached: computed and cached {output_path}")
    return result
