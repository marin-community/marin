# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-first linear helpers for Grug templates."""

from typing import TypeVar

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import DTypeLike

from haliax.quantization import DefaultDotGeneralOp, DotGeneralOp, Fp8DotGeneralOp

T = TypeVar("T")

DEFAULT_FP8_AMAX_HISTORY_LENGTH = 1024

__all__ = [
    "DEFAULT_FP8_AMAX_HISTORY_LENGTH",
    "DotGeneralOp",
    "default_dot_general",
    "enable_fp8_dot_general",
    "linear_last_dim",
]


def default_dot_general() -> DotGeneralOp:
    """Return the default Grug dense-projection dot-general op."""

    return DefaultDotGeneralOp.init()


def linear_last_dim(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    dot_general: DotGeneralOp,
    out_sharding: P | None = None,
) -> jax.Array:
    """Contract the last lhs dim with the first rhs dim.

    Grug modules keep dense weights as raw arrays, so this is the array-first
    counterpart to `haliax.nn.Linear`'s configurable `dot_general` field.
    """

    kwargs = {}
    if out_sharding is not None:
        kwargs["out_sharding"] = out_sharding
    out = dot_general(
        lhs,
        rhs,
        (((lhs.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=lhs.dtype,
        **kwargs,
    )
    if out_sharding is not None:
        out = reshard(out, out_sharding)
    return out


def enable_fp8_dot_general(
    tree: T,
    *,
    amax_history_length: int = DEFAULT_FP8_AMAX_HISTORY_LENGTH,
    compute_dtype: DTypeLike = jnp.bfloat16,
) -> T:
    """Replace default Grug dense-projection dot-general ops with FP8 state."""

    def replace_dot_general(leaf):
        if isinstance(leaf, DefaultDotGeneralOp):
            return Fp8DotGeneralOp.init(amax_history_length=amax_history_length, compute_dtype=compute_dtype)
        return leaf

    return jax.tree_util.tree_map(
        replace_dot_general,
        tree,
        is_leaf=lambda leaf: isinstance(leaf, DefaultDotGeneralOp),
    )
