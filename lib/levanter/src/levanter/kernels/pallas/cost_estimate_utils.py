# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


def bytes_for_spec(spec: jax.Array | jax.ShapeDtypeStruct | None) -> int:
    """Returns bytes for an array/shape spec, or 0 for unsupported/None values."""
    if spec is None:
        return 0
    shape = getattr(spec, "shape", None)
    dtype = getattr(spec, "dtype", None)
    if shape is None or dtype is None:
        return 0
    return math.prod(shape) * jnp.dtype(dtype).itemsize


def with_io_bytes_accessed(
    body_cost: pl.CostEstimate,
    *,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate:
    """Copies FLOP/transcendental estimates while setting bytes from kernel IO specs."""
    input_bytes = sum(bytes_for_spec(x) for x in jax.tree.leaves(kernel_inputs_specs))
    output_bytes = sum(bytes_for_spec(x) for x in jax.tree.leaves(kernel_outputs_specs))
    return pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
        remote_bytes_transferred=body_cost.remote_bytes_transferred,
    )
