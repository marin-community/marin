# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateless Transformer Engine MXFP8 dense projection."""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import auto_axes


def mxfp8_dense_dot(x: jax.Array, w: jax.Array, *, out_sharding: P) -> jax.Array:
    """Contract ``x[..., K] @ w[K, N]`` with TE's MXFP8 1D dense op."""

    def dot(x: jax.Array, w: jax.Array) -> jax.Array:
        import transformer_engine.jax as te  # type: ignore[import]  # noqa: PLC0415
        from transformer_engine.jax.quantize.quantizer import QuantizerFactory  # type: ignore[import]  # noqa: PLC0415
        from transformer_engine.jax.quantize.scaling_modes import ScalingMode  # type: ignore[import]  # noqa: PLC0415

        quantizer_set = QuantizerFactory.create_set(
            scaling_mode=ScalingMode.MXFP8_1D_SCALING,
            fwd_dtype=jnp.float8_e4m3fn,
            bwd_dtype=jnp.float8_e4m3fn,
            is_2x2x=True,
        )
        return te.dense.dense(
            x,
            w,
            contracting_dims=((x.ndim - 1,), (0,)),
            quantizer_set=quantizer_set,
        )

    # pyrefly: ignore[bad-return, bad-argument-count]  # auto_axes preserves dot's runtime signature
    return auto_axes(dot, out_sharding=out_sharding)(x, w)
