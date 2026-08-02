# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared dtype and hardware constants for QuACK CuTe kernels."""

import jax.numpy as jnp

import cutlass

SM100_FALLBACK_MAX_ACTIVE_CLUSTERS = 148

_JAX_TO_CUTE = {
    jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
    jnp.dtype(jnp.float16): cutlass.Float16,
    jnp.dtype(jnp.float32): cutlass.Float32,
}


def quack_cute_dtype(dtype):
    return _JAX_TO_CUTE[jnp.dtype(dtype)]
