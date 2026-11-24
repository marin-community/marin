# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Startup check of the Vultr MI355X launch environment that ``vultr/launch_env.sh`` sets.

Multi-node training on Vultr needs ``NCCL_PROTO=Simple`` (other RCCL protocols let a cross-node all-reduce read
stale inputs, which doubles the loss) and XLA command buffers off (they corrupt device memory). A process started
without them is refused; every other launch setting is logged so a run's log shows what took effect.
"""

import logging
import os
import resource

logger = logging.getLogger(__name__)

REQUIRED_NCCL_PROTO = "Simple"
COMMAND_BUFFERS_OFF = "--xla_gpu_enable_command_buffer="
LOGGED_VARIABLES = (
    "NCCL_PROTO",
    "XLA_FLAGS",
    "HSA_NO_SCRATCH_RECLAIM",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "NCCL_NET_PLUGIN",
    "RAGGED_DOT_IMPL",
)


def check_launch_env(num_processes: int) -> None:
    """Raise if a multi-process run lacks the required RCCL or XLA settings, then log the launch settings."""
    env = os.environ
    if num_processes > 1 and env.get("NCCL_PROTO") != REQUIRED_NCCL_PROTO:
        raise ValueError(
            f"NCCL_PROTO={env.get('NCCL_PROTO')!r}; multi-node runs need {REQUIRED_NCCL_PROTO!r} "
            "(source experiments/amd/june_67b_a2b/vultr/launch_env.sh)"
        )
    if COMMAND_BUFFERS_OFF not in env.get("XLA_FLAGS", "").split():
        raise ValueError(f"XLA_FLAGS must contain {COMMAND_BUFFERS_OFF} (XLA command buffers corrupt memory on ROCm)")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    settings = " ".join(f"{name}={env.get(name)!r}" for name in LOGGED_VARIABLES)
    logger.info("launch env: %s open-file limit %d (hard %d)", settings, soft, hard)
