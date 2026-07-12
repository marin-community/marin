# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Factory for building an :class:`Autoscaler` from cluster configuration.

Imports from both :mod:`iris.cluster.config` and the rest of the autoscaler
package. The autoscaler package ``__init__`` does not import this module, so
pulling in ``config`` here never cycles back through the package during its own
initialization (``config`` -> ``controller.backend`` -> ``controller.autoscaler``
resolves against the already-initialized package).
"""

import logging

from finelog.client.log_client import Table

from iris.cluster.config import (
    AutoscalerConfig,
    ScaleGroupConfig,
    WorkerConfig,
    validate_autoscaler_config,
    validate_scale_group_resources,
)
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.scaling_group import (
    DEFAULT_SCALE_DOWN_RATE_LIMIT,
    DEFAULT_SCALE_UP_RATE_LIMIT,
    ScalingGroup,
)
from iris.cluster.platforms.protocols import WorkerInfraProvider

logger = logging.getLogger(__name__)


def create_autoscaler(
    platform: WorkerInfraProvider,
    autoscaler_config: AutoscalerConfig,
    scale_groups: dict[str, ScaleGroupConfig],
    label_prefix: str,
    base_worker_config: WorkerConfig | None = None,
    provisioning_table: Table | None = None,
) -> Autoscaler:
    """Create autoscaler from WorkerInfraProvider and explicit config.

    Args:
        platform: WorkerInfraProvider instance for creating/discovering slices
        autoscaler_config: Autoscaler settings (already resolved with defaults)
        scale_groups: Map of scale group name to config
        label_prefix: Prefix for labels on managed resources
        base_worker_config: Base worker configuration passed through to platform.create_slice().
            None disables bootstrap (test/local mode).
        provisioning_table: finelog ``iris.provisioning`` table for slice-provisioning
            outcomes. None disables emission (test/local mode without finelog).

    Returns:
        Configured Autoscaler instance

    Raises:
        ValueError: If autoscaler_config has invalid timing values
    """
    validate_autoscaler_config(autoscaler_config, context="create_autoscaler")
    validate_scale_group_resources(scale_groups)

    scale_down_delay = autoscaler_config.scale_down_delay

    scaling_groups: dict[str, ScalingGroup] = {}
    for name, group_config in scale_groups.items():
        scaling_groups[name] = ScalingGroup(
            config=group_config,
            platform=platform,
            label_prefix=label_prefix,
            idle_threshold=scale_down_delay,
            scale_up_rate_limit=group_config.scale_up_rate_limit or DEFAULT_SCALE_UP_RATE_LIMIT,
            scale_down_rate_limit=group_config.scale_down_rate_limit or DEFAULT_SCALE_DOWN_RATE_LIMIT,
        )
        resources = group_config.resources
        worker_attrs = dict(group_config.worker.attributes)
        slice_template = group_config.slice_template
        cw_instance = (
            slice_template.coreweave.instance_type
            if slice_template is not None and slice_template.coreweave is not None
            else ""
        )
        logger.info(
            "Scale group %s: device=%s:%s device_count=%d num_vms=%d buffer=%d max=%d instance=%s worker_attrs=%s",
            name,
            resources.device_type if resources is not None else None,
            resources.device_variant if resources is not None else "",
            resources.device_count if resources is not None else 0,
            group_config.num_vms,
            group_config.buffer_slices,
            group_config.max_slices,
            cw_instance or "n/a",
            worker_attrs or "none",
        )

    def make_draining_group(name: str) -> ScalingGroup:
        """Build a scale-to-zero group: ``max_slices=0`` blocks scale-up; idle scaledown drains it."""
        return ScalingGroup(
            config=ScaleGroupConfig(name=name, max_slices=0),
            platform=platform,
            label_prefix=label_prefix,
            idle_threshold=scale_down_delay,
            scale_down_rate_limit=DEFAULT_SCALE_DOWN_RATE_LIMIT,
        )

    return Autoscaler.from_config(
        scale_groups=scaling_groups,
        config=autoscaler_config,
        platform=platform,
        base_worker_config=base_worker_config,
        make_draining_group=make_draining_group,
        provisioning_table=provisioning_table,
    )
