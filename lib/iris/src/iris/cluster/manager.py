# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster lifecycle management.

Provides free functions for cluster lifecycle operations:
- connect_cluster(): context manager that starts controller, opens tunnel, yields address
- stop_all(): stops controller and all worker slices
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from iris.cluster.config import IrisConfig, validate_config
from iris.cluster.controller.lifecycle import (
    start_controller,
    stop_controller,
)
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)

TERMINATE_TIMEOUT_SECONDS = 60


@contextmanager
def connect_cluster(config: config_pb2.IrisClusterConfig) -> Iterator[str]:
    """Start controller, open tunnel, yield address, stop on exit.

    For local mode, starts an in-process controller.
    For GCP/Manual, creates Platform, starts controller VM, opens tunnel.
    """
    validate_config(config)
    iris_config = IrisConfig(config)
    platform = iris_config.platform()

    which = config.controller.WhichOneof("controller")
    if which == "local":
        from iris.cluster.controller.local import LocalController

        controller = LocalController(config)
        address = controller.start()
        try:
            with platform.tunnel(address) as tunnel_url:
                yield tunnel_url
        finally:
            controller.stop()
            platform.shutdown()
    else:
        address, _vm = start_controller(platform, config)
        try:
            with platform.tunnel(address) as tunnel_url:
                yield tunnel_url
        finally:
            stop_controller(platform, config)
            platform.shutdown()


def _collect_terminate_targets(
    platform,
    config: config_pb2.IrisClusterConfig,
) -> list[tuple[str, Callable[[], None]]]:
    """Enumerate all resources that need terminating: controller + slices.

    Returns a list of (name, terminate_fn) pairs. Discovery (list_all_slices) is
    a single pass across all zones; the actual deletes happen later in parallel.
    """
    label_prefix = config.platform.label_prefix or "iris"
    targets: list[tuple[str, Callable[[], None]]] = []

    targets.append(("controller", lambda: stop_controller(platform, config)))

    all_slices = platform.list_all_slices(
        labels={f"{label_prefix}-managed": "true"},
    )
    for slice_handle in all_slices:
        logger.info("Queued termination for slice %s", slice_handle.slice_id)
        targets.append((f"slice:{slice_handle.slice_id}", slice_handle.terminate))

    return targets


def stop_all(config: config_pb2.IrisClusterConfig) -> None:
    """Stop controller and all worker slices in parallel.

    First enumerates all terminate targets (controller VM + TPU slices), then
    runs all deletes concurrently via daemon threads.  Applies a hard timeout
    of TERMINATE_TIMEOUT_SECONDS — any operation still running after that is
    logged at WARNING and abandoned.

    Daemon threads are used instead of ThreadPoolExecutor so that timed-out
    threads don't block interpreter shutdown (Python's atexit handler for
    ThreadPoolExecutor joins all worker threads indefinitely).
    """
    iris_config = IrisConfig(config)
    platform = iris_config.platform()
    errors: list[str] = []

    try:
        targets = _collect_terminate_targets(platform, config)
        if not targets:
            logger.info("No resources to terminate")
            return

        logger.info("Terminating %d resource(s) in parallel", len(targets))

        results: dict[str, Exception | None] = {}
        lock = threading.Lock()

        def _run(name: str, fn: Callable[[], None]) -> None:
            try:
                fn()
            except Exception as exc:
                with lock:
                    results[name] = exc
                return
            with lock:
                results[name] = None

        threads: dict[str, threading.Thread] = {}
        for name, fn in targets:
            t = threading.Thread(target=_run, args=(name, fn), daemon=True)
            t.start()
            threads[name] = t

        deadline = time.monotonic() + TERMINATE_TIMEOUT_SECONDS
        for _name, t in threads.items():
            remaining = max(0, deadline - time.monotonic())
            t.join(timeout=remaining)

        for name, t in threads.items():
            if t.is_alive():
                logger.warning(
                    "Termination of %s still running after %ds, giving up",
                    name,
                    TERMINATE_TIMEOUT_SECONDS,
                )
                errors.append(f"timeout:{name}")
            else:
                exc = results.get(name)
                if exc is not None:
                    logger.exception("Failed to terminate %s", name, exc_info=exc)
                    errors.append(name)
    finally:
        platform.shutdown()

    if errors:
        logger.error("Errors when stopping cluster: %s", errors)
