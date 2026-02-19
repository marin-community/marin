# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import BlockSizes
from .reference import linear_softmax_cross_entropy_loss_reference
from .xla import linear_softmax_cross_entropy_loss_xla


_GB10_WEIGHT_TILE_BYTES_LIMIT = 101_376
_GB10_MAX_H_TILES = 512
_GB10_FULL_MATMUL_MAX_OUTPUT_ELEMENTS = 67_108_864
_GB10_XLA_STREAMING_V_BLOCK_BATCH_1K = 2048
_GB10_XLA_STREAMING_V_BLOCK_BATCH_4K = 3072
_GB10_XLA_STREAMING_V_BLOCK_BATCH_8K = 3072
_GB10_CUSTOM_BWD_V_BLOCK_BATCH_1K = 6144
_GB10_CUSTOM_BWD_V_BLOCK_BATCH_2K_PLUS = 7168


def _apply_logit_soft_cap(logits: jax.Array, logit_soft_cap: Optional[float]) -> jax.Array:
    if logit_soft_cap is None:
        return logits
    return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _device_kind() -> str:
    if not jax.devices():
        return ""
    return jax.devices()[0].device_kind.lower()


def _max_weight_tile_bytes_for_device(device_kind: str) -> Optional[int]:
    if "gb10" in device_kind:
        return _GB10_WEIGHT_TILE_BYTES_LIMIT
    return None


def _max_h_tiles_for_device(device_kind: str) -> Optional[int]:
    if "gb10" in device_kind:
        return _GB10_MAX_H_TILES
    return None


def _should_use_gb10_full_matmul_fallback(
    x: Float[Array, "B H"],
    w: Float[Array, "H V"],
) -> bool:
    device_kind = _device_kind()
    if "gb10" not in device_kind:
        return False
    if x.dtype != jnp.bfloat16 or w.dtype != jnp.bfloat16:
        return False
    b_dim = x.shape[0]
    v_dim = w.shape[1]
    output_elements = b_dim * v_dim
    return output_elements <= _GB10_FULL_MATMUL_MAX_OUTPUT_ELEMENTS


def _gb10_xla_fallback_block_sizes(
    *,
    b_dim: int,
    v_dim: int,
) -> BlockSizes | None:
    if v_dim < 65536:
        return None
    if b_dim >= 8192:
        return BlockSizes(v_block_size=_GB10_XLA_STREAMING_V_BLOCK_BATCH_8K)
    if b_dim >= 4096:
        return BlockSizes(v_block_size=_GB10_XLA_STREAMING_V_BLOCK_BATCH_4K)
    if b_dim >= 1024:
        return BlockSizes(v_block_size=_GB10_XLA_STREAMING_V_BLOCK_BATCH_1K)
    return None


class PallasUnsupportedError(NotImplementedError):
    """Raised when the GPU fused cross-entropy backend cannot be used."""


def _validate_inputs(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    block_sizes: BlockSizes,
) -> None:
    if jax.default_backend() != "gpu":
        raise PallasUnsupportedError("Pallas fused cross-entropy requires GPU backend.")

    if x.ndim != 2:
        raise PallasUnsupportedError(f"x must be rank-2 [B, H], got shape {x.shape}.")
    if labels.ndim != 1:
        raise PallasUnsupportedError(f"labels must be rank-1 [B], got shape {labels.shape}.")
    if w.ndim != 2:
        raise PallasUnsupportedError(f"w must be rank-2 [H, V], got shape {w.shape}.")
    if x.shape[0] != labels.shape[0]:
        raise PallasUnsupportedError(f"Batch mismatch: x has B={x.shape[0]}, labels has B={labels.shape[0]}.")
    if x.shape[1] != w.shape[0]:
        raise PallasUnsupportedError(f"Hidden mismatch: x has H={x.shape[1]}, w has H={w.shape[0]}.")

    if block_sizes.b_block_size <= 0:
        raise PallasUnsupportedError(f"b_block_size must be positive, got {block_sizes.b_block_size}.")
    if block_sizes.h_block_size <= 0:
        raise PallasUnsupportedError(f"h_block_size must be positive, got {block_sizes.h_block_size}.")
    if block_sizes.v_block_size <= 0:
        raise PallasUnsupportedError(f"v_block_size must be positive, got {block_sizes.v_block_size}.")
    if block_sizes.b_block_size < 16:
        raise PallasUnsupportedError("b_block_size must be at least 16 on GPU.")
    if block_sizes.h_block_size < 16:
        raise PallasUnsupportedError("h_block_size must be at least 16 on GPU.")
    if block_sizes.v_block_size < 16:
        raise PallasUnsupportedError("v_block_size must be at least 16 on GPU.")
    if block_sizes.b_block_size % 16 != 0:
        raise PallasUnsupportedError("b_block_size must be a multiple of 16 on GPU.")
    if block_sizes.h_block_size % 16 != 0:
        raise PallasUnsupportedError("h_block_size must be a multiple of 16 on GPU.")

    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise PallasUnsupportedError(f"labels must be integer dtype, got {labels.dtype}.")


def _zero_pad(x: jax.Array, *, axis: int, multiple: int, pad_value: float | int) -> jax.Array:
    if multiple <= 0:
        raise PallasUnsupportedError(f"Padding multiple must be positive, got {multiple}.")
    dim = x.shape[axis]
    pad = (-dim) % multiple
    if pad == 0:
        return x

    pad_spec = [(0, 0)] * x.ndim
    pad_spec[axis] = (0, pad)
    return jnp.pad(x, pad_spec, constant_values=pad_value)


def _validate_launch_feasibility(
    *,
    w_dtype: jnp.dtype,
    h_block_size: int,
    v_block_size: int,
    num_h_blocks: int,
) -> None:
    if not _is_power_of_two(h_block_size):
        raise PallasUnsupportedError(
            "h_block_size must be a power of 2 for current GPU Triton lowering; " f"got h_block_size={h_block_size}."
        )
    if not _is_power_of_two(v_block_size):
        raise PallasUnsupportedError(
            "v_block_size must be a power of 2 for current GPU Triton lowering; " f"got v_block_size={v_block_size}."
        )

    device_kind = _device_kind()
    max_weight_tile_bytes = _max_weight_tile_bytes_for_device(device_kind)
    if max_weight_tile_bytes is not None:
        requested = h_block_size * v_block_size * jnp.dtype(w_dtype).itemsize
        if requested > max_weight_tile_bytes:
            raise PallasUnsupportedError(
                "Requested weight tile exceeds GPU shared-memory budget for this device: "
                f"requested={requested} bytes, limit={max_weight_tile_bytes} bytes, "
                f"h_block_size={h_block_size}, v_block_size={v_block_size}."
            )

    max_h_tiles = _max_h_tiles_for_device(device_kind)
    if max_h_tiles is not None and num_h_blocks > max_h_tiles:
        raise PallasUnsupportedError(
            "Kernel launch would require too many hidden-dimension dot tiles for this GPU and is likely to trigger "
            "pathological compile time. "
            f"num_h_blocks={num_h_blocks}, limit={max_h_tiles}, h_block_size={h_block_size}."
        )


def _forward_pallas_gpu_tile_kernel(
    x_ref,
    labels_ref,
    w_ref,
    out_m_ref,
    out_l_ref,
    out_label_ref,
    *,
    v_dim: int,
    h_block_size: int,
    num_h_blocks: int,
    v_block_size: int,
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
):
    """Compute per-(B block, V block) partials for streaming softmax reduction."""
    b_block_size = x_ref.shape[0]
    v_block_index = pl.program_id(1)
    v_start = v_block_index * v_block_size

    labels = labels_ref[...]
    in_block = labels >= 0

    logits = jnp.zeros((b_block_size, v_block_size), dtype=jnp.float32)
    for h_block_index in range(num_h_blocks):
        h_start = h_block_index * h_block_size
        h_end = h_start + h_block_size
        x_block = x_ref[:, h_start:h_end]
        w_block = w_ref[h_start:h_end, :]
        logits = logits + jax.lax.dot_general(
            x_block,
            w_block,
            (((1,), (0,)), ((), ())),
            precision=precision,
        )

    logits = _apply_logit_soft_cap(logits, logit_soft_cap)

    vocab_indices = v_start + jnp.arange(v_block_size, dtype=jnp.int32)
    in_v_block = vocab_indices < v_dim
    safe_logits = jnp.where(in_v_block[None, :], logits, -jnp.inf)

    tile_m = jnp.max(safe_logits, axis=-1)
    exp_shifted = jnp.where(in_v_block[None, :], jnp.exp(safe_logits - tile_m[:, None]), 0.0)
    tile_l = jnp.sum(exp_shifted, axis=-1)

    tile_m = jnp.where(in_block, tile_m, -jnp.inf)
    tile_l = jnp.where(in_block, tile_l, 0.0)

    label_offsets = labels.astype(jnp.int32) - jnp.int32(v_start)
    label_in_block = in_block & (label_offsets >= 0) & (label_offsets < v_block_size)
    safe_label_offsets = jnp.clip(label_offsets, 0, v_block_size - 1)
    cols = jnp.arange(v_block_size, dtype=jnp.int32)[None, :]
    label_one_hot = (cols == safe_label_offsets[:, None]).astype(logits.dtype)
    tile_label_logits = jnp.sum(logits * label_one_hot, axis=-1)
    tile_label_logits = jnp.where(label_in_block, tile_label_logits, -jnp.inf)

    out_m_ref[0, 0, :] = tile_m.astype(out_m_ref.dtype)
    out_l_ref[0, 0, :] = tile_l.astype(out_l_ref.dtype)
    out_label_ref[0, 0, :] = tile_label_logits.astype(out_label_ref.dtype)


@partial(
    jax.jit,
    static_argnames=[
        "b_block_size",
        "h_block_size",
        "v_dim",
        "num_b_blocks",
        "num_v_blocks",
        "num_h_blocks",
        "v_block_size",
        "logit_soft_cap",
        "precision",
    ],
)
def _linear_softmax_cross_entropy_loss_pallas_gpu_tiled(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    b_block_size: int,
    v_dim: int,
    h_block_size: int,
    num_b_blocks: int,
    num_v_blocks: int,
    num_h_blocks: int,
    v_block_size: int,
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "NB NV BB"], Float[Array, "NB NV BB"], Float[Array, "NB NV BB"]]:
    """Phase 1: parallel grid over (B-block, V-block) producing reduction partials."""
    h_pad = x.shape[1]

    return pl.pallas_call(
        partial(
            _forward_pallas_gpu_tile_kernel,
            v_dim=v_dim,
            h_block_size=h_block_size,
            num_h_blocks=num_h_blocks,
            v_block_size=v_block_size,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        ),
        in_specs=[
            pl.BlockSpec((b_block_size, h_pad), lambda b_index, v_index: (b_index, 0)),
            pl.BlockSpec((b_block_size,), lambda b_index, v_index: (b_index,)),
            pl.BlockSpec((h_pad, v_block_size), lambda b_index, v_index: (0, v_index)),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, b_block_size), lambda b_index, v_index: (b_index, v_index, 0)),
            pl.BlockSpec((1, 1, b_block_size), lambda b_index, v_index: (b_index, v_index, 0)),
            pl.BlockSpec((1, 1, b_block_size), lambda b_index, v_index: (b_index, v_index, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((num_b_blocks, num_v_blocks, b_block_size), dtype=jnp.float32),
            jax.ShapeDtypeStruct((num_b_blocks, num_v_blocks, b_block_size), dtype=jnp.float32),
            jax.ShapeDtypeStruct((num_b_blocks, num_v_blocks, b_block_size), dtype=jnp.float32),
        ],
        grid=(num_b_blocks, num_v_blocks),
    )(x, labels, w)


def _reduce_partials_flash_style(
    partial_m: Float[Array, "NB NV BB"],
    partial_l: Float[Array, "NB NV BB"],
    partial_label: Float[Array, "NB NV BB"],
) -> tuple[Float[Array, "NB BB"], Float[Array, "NB BB"]]:
    """FlashAttention-style associative reduction over vocab tiles."""

    def merge_fn(lhs, rhs):
        m_l, l_l, label_l = lhs
        m_r, l_r, label_r = rhs
        m = jnp.maximum(m_l, m_r)
        l = l_l * jnp.exp(m_l - m) + l_r * jnp.exp(m_r - m)
        label = jnp.maximum(label_l, label_r)
        return m, l, label

    m_scan, l_scan, label_scan = jax.lax.associative_scan(merge_fn, (partial_m, partial_l, partial_label), axis=1)
    m = m_scan[:, -1, :]
    l = l_scan[:, -1, :]
    label_logits = label_scan[:, -1, :]

    lse = jnp.log(l) + m
    loss = lse - label_logits
    return loss, lse


@partial(jax.jit, static_argnames=["block_sizes", "dtype", "logit_soft_cap", "precision"])
def _linear_softmax_cross_entropy_loss_pallas_gpu_impl(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes | None = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    """GPU Pallas implementation returning per-example loss and logsumexp."""
    device_kind = _device_kind()
    is_gb10_bf16 = "gb10" in device_kind and x.dtype == jnp.bfloat16 and w.dtype == jnp.bfloat16

    if _should_use_gb10_full_matmul_fallback(x, w):
        return linear_softmax_cross_entropy_loss_reference(
            x,
            labels,
            w,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
    if is_gb10_bf16:
        # On GB10 BF16 we intentionally keep forward on the XLA streaming path and attach
        # the custom backward via custom_vjp. This means forward loss/lse parity against
        # explicit XLA calls is expected to be exact by construction.
        xla_block_sizes = _gb10_xla_fallback_block_sizes(
            b_dim=x.shape[0],
            v_dim=w.shape[1],
        )
        return linear_softmax_cross_entropy_loss_xla(
            x,
            labels,
            w,
            block_sizes=xla_block_sizes,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )

    if block_sizes is None:
        block_sizes = BlockSizes.get_default()

    _validate_inputs(x, labels, w, block_sizes)

    b_dim, _ = x.shape
    v_dim = w.shape[1]
    b_block = block_sizes.b_block_size
    h_block = block_sizes.h_block_size
    v_block = block_sizes.v_block_size

    x_pad = _zero_pad(_zero_pad(x, axis=1, multiple=h_block, pad_value=0.0), axis=0, multiple=b_block, pad_value=0.0)
    labels_pad = _zero_pad(labels.astype(jnp.int32), axis=0, multiple=b_block, pad_value=-1)
    w_pad = _zero_pad(_zero_pad(w, axis=0, multiple=h_block, pad_value=0.0), axis=1, multiple=v_block, pad_value=0.0)

    b_pad = x_pad.shape[0]
    v_pad = w_pad.shape[1]
    h_pad = w_pad.shape[0]
    num_b_blocks = b_pad // b_block
    num_v_blocks = _ceil_div(v_pad, v_block)
    num_h_blocks = h_pad // h_block

    _validate_launch_feasibility(
        w_dtype=w_pad.dtype,
        h_block_size=h_block,
        v_block_size=v_block,
        num_h_blocks=num_h_blocks,
    )

    partial_m, partial_l, partial_label_logits = _linear_softmax_cross_entropy_loss_pallas_gpu_tiled(
        x_pad,
        labels_pad,
        w_pad,
        b_block_size=b_block,
        v_dim=v_dim,
        h_block_size=h_block,
        num_b_blocks=num_b_blocks,
        num_v_blocks=num_v_blocks,
        num_h_blocks=num_h_blocks,
        v_block_size=v_block,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )

    out_dtype = jnp.dtype(dtype) if dtype is not None else x.dtype
    loss_blocks, lse_blocks = _reduce_partials_flash_style(
        partial_m,
        partial_l,
        partial_label_logits,
    )
    out_loss = loss_blocks.astype(out_dtype).reshape((b_pad,))
    out_lse = lse_blocks.astype(out_dtype).reshape((b_pad,))

    return out_loss[:b_dim], out_lse[:b_dim]


def _gb10_custom_backward_v_block_size(
    x: Float[Array, "B H"],
    w: Float[Array, "H V"],
) -> int | None:
    device_kind = _device_kind()
    is_gb10_bf16 = "gb10" in device_kind and x.dtype == jnp.bfloat16 and w.dtype == jnp.bfloat16
    if not is_gb10_bf16:
        return None
    if _should_use_gb10_full_matmul_fallback(x, w):
        return None
    if w.shape[1] < 65536:
        return None
    if x.shape[0] >= 8192:
        return _GB10_CUSTOM_BWD_V_BLOCK_BATCH_2K_PLUS
    if x.shape[0] >= 2048:
        return _GB10_CUSTOM_BWD_V_BLOCK_BATCH_2K_PLUS
    if x.shape[0] >= 1024:
        return _GB10_CUSTOM_BWD_V_BLOCK_BATCH_1K
    return None


@partial(jax.jit, static_argnames=["v_block_size", "logit_soft_cap", "precision"])
def _backward_streaming_from_lse(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    lse: Float[Array, "B"],
    g_loss: Float[Array, "B"],
    g_lse: Float[Array, "B"],
    *,
    v_block_size: int,
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "B H"], Float[Array, "H V"]]:
    """Streaming backward pass over vocab blocks using saved LSE."""
    b_dim, h_dim = x.shape
    v_dim = w.shape[1]

    v_pad = _ceil_div(v_dim, v_block_size) * v_block_size
    if v_pad == v_dim:
        w_pad = w
    else:
        w_pad = _zero_pad(w, axis=1, multiple=v_block_size, pad_value=0.0)
    num_v_blocks = v_pad // v_block_size

    labels_i32 = labels.astype(jnp.int32)
    lse_f32 = lse.astype(jnp.float32)
    g_loss_f32 = g_loss.astype(jnp.float32)
    g_lse_f32 = g_lse.astype(jnp.float32)
    g_softmax = g_loss_f32 + g_lse_f32
    v_offsets = jnp.arange(v_block_size, dtype=jnp.int32)

    grad_x_init = jnp.zeros((b_dim, h_dim), dtype=jnp.float32)
    grad_w_init = jnp.zeros((h_dim, v_pad), dtype=jnp.float32)

    def body(v_block_index, carry):
        grad_x, grad_w = carry
        v_start = v_block_index * v_block_size
        w_block = jax.lax.dynamic_slice(w_pad, (0, v_start), (h_dim, v_block_size))

        logits_raw = jax.lax.dot_general(
            x,
            w_block,
            (((1,), (0,)), ((), ())),
            precision=precision,
        ).astype(jnp.float32)
        logits = _apply_logit_soft_cap(logits_raw, logit_soft_cap)

        in_vocab = (v_start + v_offsets) < v_dim
        probs = jnp.where(in_vocab[None, :], jnp.exp(logits - lse_f32[:, None]), 0.0)
        dlogits = probs * g_softmax[:, None]

        in_block = (labels_i32 >= v_start) & (labels_i32 < v_start + v_block_size)
        safe_idx = jnp.clip(labels_i32 - v_start, 0, v_block_size - 1)
        label_one_hot = jax.nn.one_hot(safe_idx, v_block_size, dtype=jnp.float32)
        label_one_hot = label_one_hot * in_block[:, None].astype(jnp.float32)
        dlogits = dlogits - label_one_hot * g_loss_f32[:, None]

        if logit_soft_cap is not None:
            cap = jnp.asarray(logit_soft_cap, dtype=jnp.float32)
            soft_cap_grad = 1.0 - jnp.square(logits / cap)
            dlogits = dlogits * soft_cap_grad

        dlogits_for_matmul = dlogits.astype(x.dtype)
        grad_x_block = jax.lax.dot_general(
            dlogits_for_matmul,
            w_block,
            (((1,), (1,)), ((), ())),
            precision=precision,
        ).astype(jnp.float32)
        grad_w_block = jax.lax.dot_general(
            x,
            dlogits_for_matmul,
            (((0,), (0,)), ((), ())),
            precision=precision,
        ).astype(jnp.float32)

        grad_x = grad_x + grad_x_block
        grad_w = jax.lax.dynamic_update_slice(grad_w, grad_w_block, (0, v_start))
        return grad_x, grad_w

    grad_x, grad_w = jax.lax.fori_loop(0, num_v_blocks, body, (grad_x_init, grad_w_init))
    grad_w = grad_w[:, :v_dim]
    return grad_x.astype(x.dtype), grad_w.astype(w.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _linear_softmax_cross_entropy_loss_pallas_gpu_with_custom_backward(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    block_sizes: BlockSizes | None,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    return _linear_softmax_cross_entropy_loss_pallas_gpu_impl(
        x,
        labels,
        w,
        block_sizes=block_sizes,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )


def _linear_softmax_cross_entropy_loss_pallas_gpu_with_custom_backward_fwd(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    block_sizes: BlockSizes | None,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
):
    loss, lse = _linear_softmax_cross_entropy_loss_pallas_gpu_impl(
        x,
        labels,
        w,
        block_sizes=block_sizes,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )
    backward_v_block = _gb10_custom_backward_v_block_size(x, w)
    return (loss, lse), (x, labels, w, lse, backward_v_block)


def _linear_softmax_cross_entropy_loss_pallas_gpu_with_custom_backward_bwd(
    block_sizes: BlockSizes | None,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    residuals,
    output_cotangent,
):
    x, labels, w, lse, backward_v_block = residuals
    g_loss, g_lse = output_cotangent

    if backward_v_block is not None:
        grad_x, grad_w = _backward_streaming_from_lse(
            x,
            labels,
            w,
            lse,
            g_loss,
            g_lse,
            v_block_size=backward_v_block,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
    else:
        _, vjp_fn = jax.vjp(
            lambda x_, w_: _linear_softmax_cross_entropy_loss_pallas_gpu_impl(
                x_,
                labels,
                w_,
                block_sizes=block_sizes,
                dtype=dtype,
                logit_soft_cap=logit_soft_cap,
                precision=precision,
            ),
            x,
            w,
        )
        grad_x, grad_w = vjp_fn((g_loss, g_lse))

    return grad_x, None, grad_w


_linear_softmax_cross_entropy_loss_pallas_gpu_with_custom_backward.defvjp(
    _linear_softmax_cross_entropy_loss_pallas_gpu_with_custom_backward_fwd,
    _linear_softmax_cross_entropy_loss_pallas_gpu_with_custom_backward_bwd,
)


@partial(jax.jit, static_argnames=["block_sizes", "dtype", "logit_soft_cap", "precision"])
def linear_softmax_cross_entropy_loss_pallas_gpu(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes | None = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    return _linear_softmax_cross_entropy_loss_pallas_gpu_with_custom_backward(
        x,
        labels,
        w,
        block_sizes,
        dtype,
        logit_soft_cap,
        precision,
    )


__all__ = ["linear_softmax_cross_entropy_loss_pallas_gpu", "PallasUnsupportedError"]
