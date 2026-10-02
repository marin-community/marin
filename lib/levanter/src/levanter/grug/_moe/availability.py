# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Whether this process can run the Grug MoE's GPU kernels.

The checks ask the device rather than ``jax.default_backend()``: a test that stubs the backend to
trace a GPU-only path still runs on CPU devices, which cannot launch these kernels.
"""

import functools
import logging

import jax

logger = logging.getLogger(__name__)

# QuACK's grouped GEMMs are written for SM100 and ship only with the CUDA 13 GPU extra.
_SM100_COMPUTE_CAPABILITY = 10.0


def gpu_device_present() -> bool:
    """Whether this process's default devices are GPUs."""
    return jax.devices()[0].platform == "gpu"


@functools.cache
def quack_grouped_gemm_available() -> bool:
    """Whether QuACK's SM100 grouped GEMMs, used by the ragged expert MLP, can run here."""
    device = jax.devices()[0]
    if device.platform != "gpu" or float(device.compute_capability) < _SM100_COMPUTE_CAPABILITY:
        return False
    try:
        # `sonic_cute` pulls in `quack_moe_cute`, which imports QuACK's varlen entry points at
        # module scope, so this covers a QuACK that is missing or has moved them.
        import levanter.grug._moe.sonic_cute  # noqa: F401,PLC0415
    except ImportError as exc:
        logger.warning(
            "SM100 GPU present but the QuACK grouped-GEMM kernels did not import (%s). "
            "The ragged expert MLP falls back to ragged_dot, which computes the same function "
            "more slowly. Install levanter's `gpu` extra to use them.",
            exc,
        )
        return False
    return True
