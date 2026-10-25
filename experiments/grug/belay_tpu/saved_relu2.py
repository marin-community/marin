# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The JAX Authors.
"""TPU ReLU² expert MLP with saved preactivations.

The grouped tile schedule and masked accumulation follow JAX Megablox gmm.py
(Apache-2.0). This candidate adds two fused epilogues and an explicit VJP.
The diagnostic selects this candidate explicitly; main training is unchanged.

Contract: rows are sorted by expert; nonnegative int32 group_sizes partition
ALL physical rows. Padding is a trailing suffix, included in the last group.
active_rows excludes that suffix. M/K/N must be multiples of the tile sizes.
No capacity clipping or token dropping is performed by this operator.
"""

from functools import partial

import jax
import jax.numpy as jnp
from haliax.nn.ragged_dot import _megablox_tiling
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm, make_group_metadata, tgmm


@partial(jax.jit, static_argnames=("backward", "tiling", "interpret"))
def _epilogue_gmm(
    lhs,
    rhs,
    groups,
    saved_pre=None,
    *,
    backward=False,
    tiling=_megablox_tiling,
    interpret=False,
):
    """Forward: (ReLU²(lhs @ rhs), pre); backward: (dpre, ReLU²(pre))."""
    m, k = lhs.shape
    n = rhs.shape[1] if backward else rhs.shape[2]
    tm, tk, tn = tiling(m, k, n) if callable(tiling) else tiling
    if m % tm or k % tk or n % tn:
        raise ValueError("This candidate requires whole M/K/N tiles")
    if lhs.dtype != rhs.dtype or lhs.dtype not in (jnp.float32, jnp.bfloat16):
        raise TypeError("GMM inputs must share float32 or bfloat16 dtype")
    if groups.dtype != jnp.int32 or groups.shape != (rhs.shape[0],):
        raise ValueError("Expected one int32 physical group size per expert")
    if backward and (saved_pre.shape != (m, n) or saved_pre.dtype != lhs.dtype):
        raise ValueError("Saved preactivation shape/dtype mismatch")
    metadata, active_tiles = make_group_metadata(
        group_sizes=groups,
        m=m,
        tm=tm,
        start_group=jnp.int32(0),
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )
    dtype = lhs.dtype

    def kernel(meta, a_ref, b_ref, pre_ref, first_ref, second_ref, acc_ref):
        offsets, ids, tiles = meta
        slot, ki = pl.program_id(1), pl.program_id(2)

        @pl.when(ki == 0)
        def initialize():
            acc_ref[...] = jnp.zeros((tm, tn), jnp.float32)
            first_visit = (slot == 0) | (tiles[slot] != tiles[jnp.maximum(slot - 1, 0)])

            @pl.when(first_visit)
            def initialize_outputs():
                first_ref[...] = jnp.zeros((tm, tn), dtype)
                second_ref[...] = jnp.zeros((tm, tn), dtype)

        dims = (((1,), (1,)), ((), ())) if backward else (((1,), (0,)), ((), ()))
        acc_ref[...] += lax.dot_general(
            a_ref[...],
            b_ref[...],
            dimension_numbers=dims,
            preferred_element_type=jnp.float32,
        )

        @pl.when(ki == k // tk - 1)
        def epilogue():
            row = lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + tiles[slot] * tm
            group = ids[slot]
            mask = (row >= offsets[group]) & (row < offsets[group + 1])
            matmul = acc_ref[...].astype(dtype)
            if backward:
                relu = jnp.maximum(pre_ref[...], jnp.zeros((), dtype))
                derivative = (jnp.asarray(2, dtype) * relu).astype(dtype)
                first = (matmul * derivative).astype(dtype)
                second = (relu * relu).astype(dtype)
            else:
                relu = jnp.maximum(matmul, jnp.zeros((), dtype))
                first = (relu * relu).astype(dtype)
                second = matmul
            first_ref[...] = jnp.where(mask, first, first_ref[...])
            second_ref[...] = jnp.where(mask, second, second_ref[...])

    def lhs_index(ni, slot, ki, meta):
        return meta[2][slot], ki

    def rhs_index(ni, slot, ki, meta):
        return (meta[1][slot], ni, ki) if backward else (meta[1][slot], ki, ni)

    def out_index(ni, slot, ki, meta):
        return meta[2][slot], ni

    out_spec = pl.BlockSpec((tm, tn), out_index)
    shape = jax.ShapeDtypeStruct((m, n), dtype)
    return pl.pallas_call(
        kernel,
        out_shape=(shape, shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec((tm, tk), lhs_index),
                pl.BlockSpec((None, tn, tk) if backward else (None, tk, tn), rhs_index),
                out_spec if backward else None,
            ],
            out_specs=(out_spec, out_spec),
            grid=(n // tn, active_tiles, k // tk),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
        name="belay_down_grad_relu2" if backward else "belay_up_relu2_saved_pre",
        interpret=interpret,
    )(metadata, lhs, rhs, saved_pre)


def make_saved_relu2_mlp(*, tiling=_megablox_tiling, interpret=False):
    """Build y = ReLU²(x @ W_up) @ W_down with a six-matmul forward+VJP.

    Weights may be FP32 masters; GEMMs use x.dtype, weight gradients return in
    master dtype. The residual saves preactivation rather than recomputing up.
    group_sizes/active_rows are integer nondifferentiable inputs. Padding NaNs
    in x and dy are masked before GEMMs. Runtime group validity is a caller
    contract; tests/harness explicitly check it before dispatch.
    """
    kwargs = {"tiling": tiling, "interpret": interpret}

    def forward(x, up, down, group_sizes, active_rows):
        if x.ndim != 2 or up.ndim != 3 or down.ndim != 3:
            raise ValueError("Expected x[M,D], up[E,D,H], down[E,H,D]")
        if up.shape != (down.shape[0], x.shape[1], down.shape[1]) or down.shape[2] != x.shape[1]:
            raise ValueError("Expert MLP shapes are inconsistent")
        mask = jnp.arange(x.shape[0])[:, None] < active_rows
        x = jnp.where(mask, x, 0)
        post, pre = _epilogue_gmm(x, up.astype(x.dtype), group_sizes, **kwargs)
        out = gmm(
            post,
            down.astype(x.dtype),
            group_sizes,
            preferred_element_type=x.dtype,
            **kwargs,
        )
        return jnp.where(mask, out, 0), (x, up, down, group_sizes, active_rows, pre)

    @jax.custom_vjp
    def mlp(x, up, down, group_sizes, active_rows):
        return forward(x, up, down, group_sizes, active_rows)[0]

    def backward(residual, dy):
        x, up, down, group_sizes, active_rows, pre = residual
        mask = jnp.arange(x.shape[0])[:, None] < active_rows
        dy = jnp.where(mask, dy, 0)
        dpre, post = _epilogue_gmm(dy, down.astype(x.dtype), group_sizes, pre, backward=True, **kwargs)
        ddown = tgmm(post.T, dy, group_sizes, preferred_element_type=jnp.float32, **kwargs)
        dup = tgmm(x.T, dpre, group_sizes, preferred_element_type=jnp.float32, **kwargs)
        dx = gmm(
            dpre,
            up.astype(x.dtype),
            group_sizes,
            transpose_rhs=True,
            preferred_element_type=x.dtype,
            **kwargs,
        )
        return (
            jnp.where(mask, dx, 0),
            dup.astype(up.dtype),
            ddown.astype(down.dtype),
            None,
            None,
        )

    mlp.defvjp(forward, backward)
    return mlp
