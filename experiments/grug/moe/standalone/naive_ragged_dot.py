# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax


def naive_ragged_dot(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Apply one expert matrix to each contiguous group of input rows.

    Args:
        lhs: Input rows with shape `[rows, input_dim]`, already grouped by expert.
        rhs: Expert matrices with shape `[experts, input_dim, output_dim]`.
        group_sizes: Number of contiguous input rows assigned to each expert.

    Returns:
        Output rows with shape `[rows, output_dim]` in the same order as `lhs`.

    Raises:
        ValueError: If ranks, dimensions, expert counts, or group sizes are incompatible.
    """
    raise NotImplementedError("Implement naive_ragged_dot using a Python loop and contiguous slices")
