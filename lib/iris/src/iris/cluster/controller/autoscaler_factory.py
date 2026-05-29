# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Factory for building an :class:`Autoscaler` from proto configuration.

Lives downstream of both :mod:`iris.cluster.config` and
:mod:`iris.cluster.controller.autoscaler` so it can import from each
without dragging autoscaler imports back into config (which would cycle
through ``controller.db`` -> ``controller.projections``).
"""

from __future__ import annotations

import logging

from iris.cluster.config import (
    _scale_groups_to_config,
    _validate_autoscaler_config,
    _validate_scale_group_resources,
)
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.scaling_group import (
    DEFAULT_SCALE_DOWN_RATE_LIMIT,
    DEFAULT_SCALE_UP_RATE_LIMIT,
    ScalingGroup,
)
from iris.cluster.controller.db import ControllerDB
from iris.cluster.providers.protocols import WorkerInfraProvider
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import config_pb2
from iris.time_proto import duration_from_proto

logger = logging.getLogger(__name__)


def create_autoscaler(
    platform: WorkerInfraProvider,
    autoscaler_config: config_pb2.AutoscalerConfig,
    scale_groups: dict[str, config_pb2.ScaleGroupConfig],
    label_prefix: str,
    base_worker_config: config_pb2.WorkerConfig | None = None,
    threads: ThreadContainer | None = None,
    db: ControllerDB | None = None,
) -> Autoscaler:
    """Create autoscaler from WorkerInfraProvider and explicit config.

    Args:
        platform: WorkerInfraProvider instance for creating/discovering slices
        autoscaler_config: Autoscaler settings (already resolved with defaults)
        scale_groups: Map of scale group name to config
        label_prefix: Prefix for labels on managed resources
        base_worker_config: Base worker configuration passed through to platform.create_slice().
            None disables bootstrap (test/local mode).
        threads: Thread container for background threads. Uses global default if not provided.
        db: Optional DB handle for write-through persistence.

    Returns:
        Configured Autoscaler instance

    Raises:
        ValueError: If autoscaler_config has invalid timing values
    """
    threads = threads or get_thread_container()

    _validate_autoscaler_config(autoscaler_config, context="create_autoscaler")
    _validate_scale_group_resources(_scale_groups_to_config(scale_groups))

    scale_down_delay = duration_from_proto(autoscaler_config.scale_down_delay)

    scaling_groups: dict[str, ScalingGroup] = {}
    for name, group_config in scale_groups.items():
        scaling_groups[name] = ScalingGroup(
            config=group_config,
            platform=platform,
            label_prefix=label_prefix,
            idle_threshold=scale_down_delay,
            scale_up_rate_limit=group_config.scale_up_rate_limit or DEFAULT_SCALE_UP_RATE_LIMIT,
            scale_down_rate_limit=group_config.scale_down_rate_limit or DEFAULT_SCALE_DOWN_RATE_LIMIT,
            db=db,
        )
        resources = group_config.resources
        worker_attrs = dict(group_config.worker.attributes) if group_config.HasField("worker") else {}
        slice_template = group_config.slice_template
        cw_instance = slice_template.coreweave.instance_type if slice_template.HasField("coreweave") else ""
        logger.info(
            "Scale group %s: device=%s:%s device_count=%d num_vms=%d buffer=%d max=%d instance=%s worker_attrs=%s",
            name,
            resources.device_type,
            resources.device_variant,
            resources.device_count,
            group_config.num_vms,
            group_config.buffer_slices,
            group_config.max_slices,
            cw_instance or "n/a",
            worker_attrs or "none",
        )

    return Autoscaler.from_config(
        scale_groups=scaling_groups,
        config=autoscaler_config,
        platform=platform,
        base_worker_config=base_worker_config,
        db=db,
    )
