# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Whether this process can run the Grug MoE's GPU kernels.

The checks ask the device rather than ``jax.default_backend()``: a test that stubs the backend to
trace a GPU-only path still runs on CPU devices, which cannot launch these kernels.
"""

import jax


def gpu_device_present() -> bool:
    """Whether this process's default devices are GPUs."""
    return jax.devices()[0].platform == "gpu"
