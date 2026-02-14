# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Device environment variable construction for container runtimes.

Builds the JAX/TPU environment variables needed for multi-host distributed
initialization. Used by both Docker and process runtimes.
"""

import logging

from iris.cluster.runtime.types import ContainerConfig

logger = logging.getLogger(__name__)


def build_device_env_vars(config: ContainerConfig) -> dict[str, str]:
    """Build device-specific environment variables for the container.

    When TPU resources are requested, adds JAX/PJRT environment variables
    and TPU metadata from the worker's environment. These environment variables
    enable JAX to properly initialize on TPU devices inside the container.
    """
    env: dict[str, str] = {}

    if not config.resources:
        logger.debug("No resources on container config; skipping device env vars")
        return env

    has_device = config.resources.HasField("device")
    has_tpu = has_device and config.resources.device.HasField("tpu")

    if has_tpu:
        env["JAX_PLATFORMS"] = "tpu,cpu"
        env["PJRT_DEVICE"] = "TPU"
        # Disable JAX's GCE metadata-based TPU cluster detection. Inside Docker
        # the metadata service returns single-host info even on multi-host pods.
        # We set the JAX distributed env vars explicitly below instead.
        env["TPU_SKIP_MDS_QUERY"] = "1"

        # libtpu uses TPU_ACCELERATOR_TYPE to infer topology for pod slices.
        if config.worker_metadata and config.worker_metadata.device.HasField("tpu"):
            tpu_variant = config.worker_metadata.device.tpu.variant
            if tpu_variant:
                env["TPU_ACCELERATOR_TYPE"] = tpu_variant
                # TPU_TYPE is the host-level name set at bootstrap and still
                # used by Iris topology helpers.
                env["TPU_TYPE"] = tpu_variant

        if config.worker_metadata:
            if config.worker_metadata.tpu_name:
                env["TPU_NAME"] = config.worker_metadata.tpu_name
            if config.worker_metadata.tpu_worker_id:
                env["TPU_WORKER_ID"] = config.worker_metadata.tpu_worker_id
                # Alias variables used by some TPU runtimes and tooling.
                env["WORKER_ID"] = config.worker_metadata.tpu_worker_id
                env["CLOUD_TPU_TASK_ID"] = config.worker_metadata.tpu_worker_id
            if config.worker_metadata.tpu_worker_hostnames:
                if not config.worker_metadata.tpu_worker_id:
                    raise ValueError(
                        "TPU worker metadata is incomplete: TPU_WORKER_ID is required "
                        "when TPU_WORKER_HOSTNAMES is set."
                    )
                env["TPU_WORKER_HOSTNAMES"] = config.worker_metadata.tpu_worker_hostnames
                # JAX multi-host coordination: coordinator is worker-0's IP with standard port.
                # JAX reads JAX_COORDINATOR_ADDRESS directly. For num_processes and process_id
                # JAX relies on cluster auto-detection; with TPU_SKIP_MDS_QUERY set,
                # GkeTpuCluster is used which reads TPU_WORKER_HOSTNAMES and TPU_WORKER_ID.
                hostnames = config.worker_metadata.tpu_worker_hostnames.split(",")
                env["JAX_COORDINATOR_ADDRESS"] = f"{hostnames[0]}:8476"
                env["JAX_NUM_PROCESSES"] = str(len(hostnames))
                env["JAX_PROCESS_ID"] = config.worker_metadata.tpu_worker_id
            if config.worker_metadata.tpu_chips_per_host_bounds:
                env["TPU_CHIPS_PER_HOST_BOUNDS"] = config.worker_metadata.tpu_chips_per_host_bounds
            logger.info("TPU device env vars (with metadata): %s", env)
        else:
            logger.warning("TPU device requested but worker_metadata is None; TPU host env vars will be missing")
            logger.info("TPU device env vars (no metadata): %s", env)

    return env
