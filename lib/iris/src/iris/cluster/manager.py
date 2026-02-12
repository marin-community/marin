# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster lifecycle management.

Provides free functions for cluster lifecycle operations:
- connect_cluster(): context manager that starts controller, opens tunnel, yields address
- stop_all(): stops controller and all worker slices
"""

from __future__ import annotations

import concurrent.futures
import logging
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
    runs all gcloud delete calls concurrently via a thread pool. Applies a hard
    timeout of TERMINATE_TIMEOUT_SECONDS — any operation still running after
    that is logged at WARNING and abandoned.
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

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(targets))
        try:
            future_to_name: dict[concurrent.futures.Future, str] = {executor.submit(fn): name for name, fn in targets}

            done, not_done = concurrent.futures.wait(future_to_name, timeout=TERMINATE_TIMEOUT_SECONDS)

            for future in done:
                name = future_to_name[future]
                try:
                    future.result()
                except Exception:
                    logger.exception("Failed to terminate %s", name)
                    errors.append(name)

            for future in not_done:
                name = future_to_name[future]
                logger.warning(
                    "Termination of %s still running after %ds, giving up",
                    name,
                    TERMINATE_TIMEOUT_SECONDS,
                )
                errors.append(f"timeout:{name}")
        finally:
            # wait=False so we don't block on timed-out gcloud subprocesses;
            # cancel_futures prevents any not-yet-started work from running.
            executor.shutdown(wait=False, cancel_futures=True)
    finally:
        platform.shutdown()

    if errors:
        raise RuntimeError(f"stop_all completed with {len(errors)} error(s): {', '.join(errors)}")
