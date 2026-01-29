# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import BlockSizes
from .reference import (
    linear_softmax_cross_entropy_loss_reference,
    linear_softmax_cross_entropy_loss_streaming,
)


def linear_softmax_cross_entropy_loss_xla(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes | None = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    if block_sizes is None:
        return linear_softmax_cross_entropy_loss_reference(
            x,
            labels,
            w,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
    return linear_softmax_cross_entropy_loss_streaming(
        x,
        labels,
        w,
        block_size=block_sizes.v_block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )


__all__ = ["linear_softmax_cross_entropy_loss_xla"]
