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

    # N.B. We originally set all of the TPU environment variables explicitly, but this interferes with Jax's
    # automatic Cloud TPU detection. Forcing Jax to do Cloud TPU init is sufficient.
    if has_tpu:
        env["JAX_PLATFORMS"] = "tpu,cpu"
        env["PJRT_DEVICE"] = "TPU"

        # Jax likes to ignore the fact we're on a TPU for some reason.
        env["JAX_FORCE_TPU_INIT"] = "1"

    return env
