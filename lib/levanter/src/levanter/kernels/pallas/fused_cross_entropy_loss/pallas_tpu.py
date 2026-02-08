# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# This implementation is heavily based on Tokamax's linear softmax
# cross-entropy Pallas Mosaic TPU kernel (Apache-2.0). We adapt it for
# Levanter's API and add optional logsumexp penalty, logit soft-cap, and
# external loss weighting support.

from functools import lru_cache, partial
import math
from typing import Optional

import jax
from jax._src import ad_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import BlockSizes


class PallasUnsupportedError(NotImplementedError):
    """Raised when Pallas kernel cannot be used for given inputs."""


NUM_LANES = 128


def _apply_logit_soft_cap(logits: jax.Array, logit_soft_cap: Optional[float]) -> jax.Array:
    if logit_soft_cap is None:
        return logits
    return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


def _labels_one_hot_emulated(
    labels_adjusted: jax.Array,
    num_classes: int,
    dtype: jnp.dtype,
) -> jax.Array:
    labels_adjusted = labels_adjusted.astype(jnp.int32)
    in_block = (labels_adjusted >= 0) & (labels_adjusted < num_classes)
    safe_labels = jnp.where(in_block, labels_adjusted, -1)
    cols = jnp.arange(num_classes, dtype=labels_adjusted.dtype)[None, :]
    return (cols == safe_labels[:, None]).astype(dtype)


def _validate_inputs(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    block_sizes: BlockSizes,
) -> None:
    if jax.default_backend() != "tpu":
        raise PallasUnsupportedError("Pallas fused cross-entropy requires TPU backend.")
    for name, value in (
        ("b_block_size", block_sizes.b_block_size),
        ("h_block_size", block_sizes.h_block_size),
        ("v_block_size", block_sizes.v_block_size),
    ):
        if value % 128 != 0:
            raise PallasUnsupportedError(f"{name} must be a multiple of 128, got {value}.")
    if x.ndim != 2:
        raise PallasUnsupportedError(f"x must be rank-2 [B, H], got shape {x.shape}.")
    if labels.ndim != 1:
        raise PallasUnsupportedError(f"labels must be rank-1 [B], got shape {labels.shape}.")
    if w.ndim != 2:
        raise PallasUnsupportedError(f"w must be rank-2 [H, V], got shape {w.shape}.")

    if x.shape[0] % block_sizes.b_block_size != 0:
        raise PallasUnsupportedError(
            "Batch dimension must be a multiple of b_block_size: "
            f"B={x.shape[0]}, b_block_size={block_sizes.b_block_size}."
        )
    if labels.shape[0] % block_sizes.b_block_size != 0:
        raise PallasUnsupportedError(
            "Labels dimension must be a multiple of b_block_size: "
            f"B={labels.shape[0]}, b_block_size={block_sizes.b_block_size}."
        )
    if x.shape[1] % block_sizes.h_block_size != 0:
        raise PallasUnsupportedError(
            "Hidden dimension must be a multiple of h_block_size: "
            f"H={x.shape[1]}, h_block_size={block_sizes.h_block_size}."
        )
    if w.shape[0] % block_sizes.h_block_size != 0:
        raise PallasUnsupportedError(
            "Weight hidden dimension must be a multiple of h_block_size: "
            f"H={w.shape[0]}, h_block_size={block_sizes.h_block_size}."
        )
    if jax.default_backend() == "tpu" and x.shape[0] >= 1024:
        device_kind = jax.devices()[0].device_kind.lower()
        if ("tpu v4" in device_kind or "tpu v5" in device_kind or "tpu v7" in device_kind) and (
            block_sizes.b_block_size % 1024 != 0
        ):
            raise PallasUnsupportedError(
                "TPU label layout requires b_block_size to be a multiple of 1024 when B>=1024; "
                f"got b_block_size={block_sizes.b_block_size}."
            )


def _infer_num_tensorcores() -> int:
    if jax.default_backend() != "tpu":
        return 1
    device_kind = jax.devices()[0].device_kind.lower()
    if "tpu v4" in device_kind or ("tpu v5" in device_kind and "v5e" not in device_kind) or "tpu v7" in device_kind:
        return 2
    return 1


def _infer_core_grid(b_dim: int, block_sizes: BlockSizes) -> tuple[int, int]:
    num_cores = _infer_num_tensorcores()
    if num_cores > 1:
        if b_dim % num_cores != 0:
            num_cores = 1
        else:
            b_per_core = b_dim // num_cores
            if b_per_core % block_sizes.b_block_size != 0:
                num_cores = 1
    num_b_blocks_per_core = b_dim // (num_cores * block_sizes.b_block_size)
    return num_cores, num_b_blocks_per_core


def linear_softmax_cross_entropy_loss_forward_pallas_kernel(
    x_ref,
    labels_ref,
    w_ref,
    lse_ref,
    label_logits_ref,
    xw_tiled,
    m_scratch_ref,
    l_scratch_ref,
    label_logits_scratch_ref,
    *,
    v_dim: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
):
    """Forward kernel (streaming logsumexp + label logits; no gathers)."""
    b_index, v_index, h_index = (pl.program_id(i) for i in range(3))
    del b_index
    v_block_size = w_ref.shape[1]
    num_v_blocks, num_h_blocks = (pl.num_programs(i) for i in range(1, 3))

    @pl.when(v_index == num_v_blocks - 1)
    def pad_non_aligned_v_block():
        if v_dim % v_block_size != 0:
            rem = v_dim % v_block_size
            w_ref[:, rem:] = jnp.zeros((w_ref.shape[0], w_ref.shape[1] - rem), dtype=w_ref.dtype)

    @pl.when(jnp.logical_and(v_index == 0, h_index == 0))
    def init_accumulators():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        label_logits_scratch_ref[...] = jnp.zeros_like(label_logits_scratch_ref)

    @pl.when(h_index == 0)
    def init_logits():
        xw_tiled[...] = jnp.zeros_like(xw_tiled)

    xw_tiled[...] += jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=precision,
    )

    @pl.when(h_index == num_h_blocks - 1)
    def accumulate_block():
        logits = xw_tiled[...]
        if dtype is not None:
            logits = logits.astype(dtype)
        logits = _apply_logit_soft_cap(logits, logit_soft_cap)

        labels_adjusted = labels_ref[...] - v_index * v_block_size
        labels_one_hot = _labels_one_hot_emulated(labels_adjusted, v_block_size, logits.dtype)
        label_logits_block = jnp.sum(labels_one_hot * logits, axis=-1)
        label_logits_scratch_ref[...] += label_logits_block[:, None]

        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
        m_curr = jnp.max(logits, axis=-1)[:, None]
        m_next = jnp.maximum(m_prev, m_curr)

        bkv_repeats, rem = divmod(v_block_size, NUM_LANES)
        if rem != 0:
            raise NotImplementedError(f"{v_block_size=} must be a multiple of {NUM_LANES}")

        s_curr = jnp.exp(logits - pltpu.repeat(m_next, bkv_repeats, axis=1))
        l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
        alpha = jnp.exp(m_prev - m_next)
        l_next = l_curr + alpha * l_prev
        m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next

    @pl.when(jnp.logical_and(v_index == num_v_blocks - 1, h_index == num_h_blocks - 1))
    def finalize_loss():
        lse_ref[...] = (jnp.log(l_scratch_ref[...]) + m_scratch_ref[...]).astype(lse_ref.dtype)
        label_logits_ref[...] = label_logits_scratch_ref[...].astype(label_logits_ref.dtype)


@partial(
    jax.jit,
    static_argnames=["block_sizes", "dtype", "logit_soft_cap", "precision"],
)
def linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    """Forward Pallas kernel wrapper (per-example loss + logsumexp)."""
    _validate_inputs(x, labels, w, block_sizes)

    h_dim = x.shape[-1]
    v_dim = w.shape[1]
    b_dim = x.shape[0]

    num_b_blocks = math.ceil(b_dim / block_sizes.b_block_size)
    num_h_blocks = math.ceil(h_dim / block_sizes.h_block_size)
    num_v_blocks = math.ceil(v_dim / block_sizes.v_block_size)

    out_dtype = jnp.dtype(dtype) if dtype is not None else x.dtype

    lse_lanes, label_logits_lanes = pl.pallas_call(
        partial(
            linear_softmax_cross_entropy_loss_forward_pallas_kernel,
            v_dim=v_dim,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        ),
        in_specs=[
            pl.BlockSpec(
                (block_sizes.b_block_size, block_sizes.h_block_size),
                lambda i, j, k: (i, k),
                memory_space=pltpu.VMEM,
            ),  # x
            pl.BlockSpec(
                (block_sizes.b_block_size,),
                lambda i, j, k: (i),
                memory_space=pltpu.VMEM,
            ),  # labels
            pl.BlockSpec(
                (block_sizes.h_block_size, block_sizes.v_block_size),
                lambda i, j, k: (k, j),
                memory_space=pltpu.VMEM,
            ),  # w
        ],
        out_specs=[
            pl.BlockSpec(
                (block_sizes.b_block_size, NUM_LANES),
                lambda i, j, k: (i, 0),
                memory_space=pltpu.VMEM,
            ),  # lse lanes
            pl.BlockSpec(
                (block_sizes.b_block_size, NUM_LANES),
                lambda i, j, k: (i, 0),
                memory_space=pltpu.VMEM,
            ),  # label logits lanes
        ],
        out_shape=[
            jax.ShapeDtypeStruct(shape=(b_dim, NUM_LANES), dtype=out_dtype),
            jax.ShapeDtypeStruct(shape=(b_dim, NUM_LANES), dtype=out_dtype),
        ],
        scratch_shapes=(
            pltpu.VMEM(
                (block_sizes.b_block_size, block_sizes.v_block_size),
                dtype=out_dtype,
            ),  # xw_tiled
            pltpu.VMEM((block_sizes.b_block_size, NUM_LANES), dtype=out_dtype),  # m_scratch
            pltpu.VMEM((block_sizes.b_block_size, NUM_LANES), dtype=out_dtype),  # l_scratch
            pltpu.VMEM((block_sizes.b_block_size, NUM_LANES), dtype=out_dtype),  # label_logits
        ),
        grid=(num_b_blocks, num_v_blocks, num_h_blocks),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        ),
    )(x, labels, w)

    lse = lse_lanes[:, 0]
    label_logits = label_logits_lanes[:, 0]
    loss = lse - label_logits
    return loss, lse


def linear_softmax_cross_entropy_loss_backward_pallas_parallel_kernel(
    x_ref,
    labels_ref,
    w_ref,
    lse_ref,
    dout_loss_ref,
    dout_lse_ref,
    x_grad_hbm_ref,
    w_grad_hbm_ref,
    xw_scratch_ref,
    x_grad_tile_ref,
    w_grad_tile_ref,
    x_read_sem,
    w_read_sem,
    x_write_sem,
    w_write_sem,
    *,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
):
    """Backward kernel with an explicit core grid axis (per-core w_grad partials)."""
    core_index, b_index, v_index, stage_index, h_index = (pl.program_id(i) for i in range(5))
    num_v_blocks = pl.num_programs(2)
    num_b_blocks_per_core = pl.num_programs(1)
    b_block_size, h_block_size = x_ref.shape
    v_block_size = w_ref.shape[1]
    v_dim = w_grad_hbm_ref.shape[-1]

    # Zero the tail if V isn't aligned so the last tile is safe.
    @pl.when(v_index == num_v_blocks - 1)
    def pad_non_aligned_v_block():
        if v_dim % v_block_size != 0:
            rem = v_dim % v_block_size
            w_ref[:, rem:] = jnp.zeros((w_ref.shape[0], w_ref.shape[1] - rem), dtype=w_ref.dtype)

    # Stage 0: build logits for this (B,V) tile by accumulating over H.
    @pl.when(jnp.logical_and(stage_index == 0, h_index == 0))
    def init_logits():
        xw_scratch_ref[...] = jax.lax.dot_general(
            x_ref[...],
            w_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision,
        )

    @pl.when(jnp.logical_and(stage_index == 0, h_index != 0))
    def accumulate_logits():
        xw_scratch_ref[...] += jax.lax.dot_general(
            x_ref[...],
            w_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision,
        )

    cur_v_block_size = jnp.minimum(v_dim - v_block_size * v_index, v_block_size)
    cur_v_block_size = (jnp.ceil(cur_v_block_size / 128) * 128).astype(jnp.int32)
    cur_v_block_size = pl.multiple_of(cur_v_block_size, 128)

    b_index_global = core_index * num_b_blocks_per_core + b_index
    x_grad_slice = x_grad_hbm_ref.at[
        pl.ds(b_index_global * b_block_size, b_block_size),
        pl.ds(h_index * h_block_size, h_block_size),
    ]
    w_grad_slice = w_grad_hbm_ref.at[
        core_index,
        pl.ds(h_index * h_block_size, h_block_size),
        pl.ds(v_index * v_block_size, cur_v_block_size),
    ]
    w_grad_tile_slice = w_grad_tile_ref.at[:, pl.ds(0, cur_v_block_size)]

    x_write_future = pltpu.make_async_copy(x_grad_tile_ref, x_grad_slice, sem=x_write_sem)
    w_write_future = pltpu.make_async_copy(w_grad_tile_slice, w_grad_slice, sem=w_write_sem)
    x_read_future = pltpu.make_async_copy(x_grad_slice, x_grad_tile_ref, sem=x_read_sem)
    w_read_future = pltpu.make_async_copy(w_grad_slice, w_grad_tile_slice, sem=w_read_sem)

    # Stage 1: async DMA reads for gradient tiles.
    @pl.when(jnp.logical_and(stage_index == 1, v_index != 0))
    def x_read():
        x_read_future.start()

    @pl.when(jnp.logical_and(stage_index == 1, b_index != 0))
    def w_read():
        w_read_future.start()

    # Compute the softmax-gradient term for this V tile (once per H=0).
    @pl.when(jnp.logical_and(stage_index == 1, h_index == 0))
    def compute_s():
        labels_adjusted = labels_ref[...] - v_index * v_block_size
        labels_one_hot = _labels_one_hot_emulated(labels_adjusted, v_block_size, x_ref.dtype)
        logits = xw_scratch_ref[...]
        if dtype is not None:
            logits = logits.astype(dtype)
        if logit_soft_cap is not None:
            tanh_arg = logits / logit_soft_cap
            tanh_val = jnp.tanh(tanh_arg)
            logits = tanh_val * logit_soft_cap
            cap_deriv = 1.0 - tanh_val**2
        else:
            cap_deriv = 1.0

        logits = logits - lse_ref[...][:, None]
        probs = jnp.exp(logits)
        dout_loss = dout_loss_ref[...]
        dout_lse = dout_lse_ref[...]
        delta = (dout_loss[:, None] + dout_lse[:, None]) * probs - dout_loss[:, None] * labels_one_hot
        xw_scratch_ref[...] = delta * cap_deriv

    @pl.when(jnp.logical_and(stage_index == 1, v_index == 0))
    def init_x_grad():
        x_grad_tile_ref[...] = jax.lax.dot_general(
            xw_scratch_ref[...],
            w_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision,
        )
        x_write_future.start()

    @pl.when(jnp.logical_and(stage_index == 1, v_index != 0))
    def accumulate_x_grad():
        res = jax.lax.dot_general(
            xw_scratch_ref[...],
            w_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision,
        )
        x_read_future.wait()
        x_grad_tile_ref[...] += res
        x_write_future.start()

    @pl.when(stage_index == 1)
    def wait_async_x_writes():
        x_write_future.wait()

    @pl.when(jnp.logical_and(stage_index == 1, b_index == 0))
    def init_w_grad():
        w_grad_tile_ref[...] = jax.lax.dot_general(
            x_ref[...],
            xw_scratch_ref[...],
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision,
        )
        w_write_future.start()

    @pl.when(jnp.logical_and(stage_index == 1, b_index != 0))
    def accumulate_w_grad():
        res = jax.lax.dot_general(
            x_ref[...],
            xw_scratch_ref[...],
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=precision,
        )
        w_read_future.wait()
        w_grad_tile_ref[...] += res
        w_write_future.start()

    @pl.when(stage_index == 1)
    def wait_async_writes():
        w_write_future.wait()


@partial(
    jax.jit,
    static_argnames=["block_sizes", "dtype", "logit_soft_cap", "precision"],
)
def _linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined(
    dout_loss: Float[Array, "B"],
    dout_lse: Float[Array, "B"],
    lse: Float[Array, "B"],
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
) -> tuple[Float[Array, "B H"], Float[Array, "H V"]]:
    """Backward Pallas kernel wrapper (combined dx/dw)."""
    _validate_inputs(x, labels, w, block_sizes)

    v_dim = w.shape[1]
    b_dim = x.shape[0]
    h_dim = x.shape[-1]
    num_v_blocks = math.ceil(v_dim / block_sizes.v_block_size)
    num_h_blocks = math.ceil(h_dim / block_sizes.h_block_size)
    num_stages = 2
    num_cores, num_b_blocks_per_core = _infer_core_grid(b_dim, block_sizes)
    x_grad, w_grad_partial = pl.pallas_call(
        partial(
            linear_softmax_cross_entropy_loss_backward_pallas_parallel_kernel,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        ),
        grid=(num_cores, num_b_blocks_per_core, num_v_blocks, num_stages, num_h_blocks),
        in_specs=[
            pl.BlockSpec(  # x
                (block_sizes.b_block_size, block_sizes.h_block_size),
                lambda c, i, j, s, k: (c * num_b_blocks_per_core + i, k),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # labels
                (block_sizes.b_block_size,),
                lambda c, i, j, s, k: (c * num_b_blocks_per_core + i),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # w
                (block_sizes.h_block_size, block_sizes.v_block_size),
                lambda c, i, j, s, k: (k, j),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # lse
                (block_sizes.b_block_size,),
                lambda c, i, j, s, k: (c * num_b_blocks_per_core + i,),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # dout_loss
                (block_sizes.b_block_size,),
                lambda c, i, j, s, k: (c * num_b_blocks_per_core + i,),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # dout_lse
                (block_sizes.b_block_size,),
                lambda c, i, j, s, k: (c * num_b_blocks_per_core + i,),
                memory_space=pltpu.VMEM,
            ),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),  # x_grad
            pl.BlockSpec(memory_space=pltpu.HBM),  # w_grad_partial
        ],
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, dtype=jnp.float32),
            jax.ShapeDtypeStruct((num_cores,) + w.shape, dtype=jnp.float32),
        ],
        scratch_shapes=(
            pltpu.VMEM((block_sizes.b_block_size, block_sizes.v_block_size), dtype=jnp.float32),  # xw_scratch
            pltpu.VMEM((block_sizes.b_block_size, block_sizes.h_block_size), dtype=jnp.float32),  # x_grad_tile
            pltpu.VMEM((block_sizes.h_block_size, block_sizes.v_block_size), dtype=jnp.float32),  # w_grad_tile
            pltpu.SemaphoreType.DMA,  # x_read_sem
            pltpu.SemaphoreType.DMA,  # w_read_sem
            pltpu.SemaphoreType.DMA,  # x_write_sem
            pltpu.SemaphoreType.DMA,  # w_write_sem
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary", "arbitrary", "arbitrary"),
        ),
    )(x, labels, w, lse, dout_loss, dout_lse)
    w_grad = jnp.sum(w_grad_partial, axis=0)
    return x_grad, w_grad


@partial(
    jax.jit,
    static_argnames=["block_sizes", "dtype", "logit_soft_cap", "precision"],
)
def linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
    dout_loss: Float[Array, "B"],
    dout_lse: Float[Array, "B"],
    lse: Float[Array, "B"],
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
) -> tuple[Float[Array, "B H"], Float[Array, "H V"]]:
    return _linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined(
        dout_loss,
        dout_lse,
        lse,
        x,
        labels,
        w,
        block_sizes=block_sizes,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )


def _zeros_like_if_needed(ct, like):
    if isinstance(ct, ad_util.Zero):
        return jnp.zeros_like(like)
    return ct


@lru_cache(maxsize=None)
def _make_custom_vjp(
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
):
    block_sizes = BlockSizes(
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
    )

    def _forward(x: jax.Array, labels: jax.Array, w: jax.Array):
        return linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
            x,
            labels,
            w,
            block_sizes=block_sizes,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )

    @jax.custom_vjp
    def _fn(x: jax.Array, labels: jax.Array, w: jax.Array):
        return _forward(x, labels, w)

    def _fn_fwd(x: jax.Array, labels: jax.Array, w: jax.Array):
        loss, lse = _forward(x, labels, w)
        return (loss, lse), (x, labels, w, lse)

    def _fn_bwd(residuals, ct):
        x, labels, w, lse = residuals
        dout_loss, dout_lse = ct
        dout_loss = _zeros_like_if_needed(dout_loss, lse)
        dout_lse = _zeros_like_if_needed(dout_lse, lse)

        x_grad, w_grad = linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
            dout_loss,
            dout_lse,
            lse,
            x,
            labels,
            w,
            block_sizes=block_sizes,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
        labels_grad = jnp.zeros_like(labels)
        return x_grad, labels_grad, w_grad

    _fn.defvjp(_fn_fwd, _fn_bwd)
    return _fn


def linear_softmax_cross_entropy_loss_pallas(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    """Pallas implementation returning (loss, lse) per example."""
    fn = _make_custom_vjp(
        block_sizes.b_block_size,
        block_sizes.h_block_size,
        block_sizes.v_block_size,
        dtype,
        logit_soft_cap,
        precision,
    )
    return fn(x, labels, w)


__all__ = [
    "PallasUnsupportedError",
    "linear_softmax_cross_entropy_loss_pallas",
]
