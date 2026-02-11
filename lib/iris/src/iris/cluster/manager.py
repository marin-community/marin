# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster lifecycle management.

Provides free functions for cluster lifecycle operations:
- connect_cluster(): context manager that starts controller, opens tunnel, yields address
- stop_all(): stops controller and all worker slices
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from iris.cluster.config import IrisConfig, validate_config
from iris.cluster.controller.lifecycle import (
    start_controller,
    stop_controller,
)
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)


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


def stop_all(config: config_pb2.IrisClusterConfig) -> None:
    """Stop controller and all worker slices.

    Best-effort cleanup: attempts every stop/terminate operation even if earlier
    ones fail, and always calls platform.shutdown(). Failures are logged but do
    not prevent subsequent cleanup steps from running.
    """
    iris_config = IrisConfig(config)
    platform = iris_config.platform()
    label_prefix = config.platform.label_prefix or "iris"
    errors: list[str] = []

    try:
        try:
            stop_controller(platform, config)
        except Exception:
            logger.exception("Failed to stop controller")
            errors.append("stop_controller")

        for group_config in config.scale_groups.values():
            try:
                template = group_config.slice_template
                if template.HasField("gcp") and template.gcp.zone:
                    zones = [template.gcp.zone]
                else:
                    zones = ["local"]

                for slice_handle in platform.list_slices(
                    zones=zones,
                    labels={f"{label_prefix}-scale-group": group_config.name},
                ):
                    try:
                        logger.info("Terminating slice %s", slice_handle.slice_id)
                        slice_handle.terminate()
                    except Exception:
                        logger.exception("Failed to terminate slice %s", slice_handle.slice_id)
                        errors.append(f"terminate:{slice_handle.slice_id}")
            except Exception:
                logger.exception("Failed to list/terminate slices for group %s", group_config.name)
                errors.append(f"list_slices:{group_config.name}")
    finally:
        platform.shutdown()

    if errors:
        raise RuntimeError(f"stop_all completed with {len(errors)} error(s): {', '.join(errors)}")
