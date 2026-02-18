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


_GB10_WEIGHT_TILE_BYTES_LIMIT = 101_376
_GB10_MAX_DOT_TILES = 512


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


def _max_dot_tiles_for_device(device_kind: str) -> Optional[int]:
    if "gb10" in device_kind:
        return _GB10_MAX_DOT_TILES
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
        raise PallasUnsupportedError(
            f"Batch mismatch: x has B={x.shape[0]}, labels has B={labels.shape[0]}."
        )
    if x.shape[1] != w.shape[0]:
        raise PallasUnsupportedError(
            f"Hidden mismatch: x has H={x.shape[1]}, w has H={w.shape[0]}."
        )

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
    num_v_blocks: int,
) -> None:
    if not _is_power_of_two(h_block_size):
        raise PallasUnsupportedError(
            "h_block_size must be a power of 2 for current GPU Triton lowering; "
            f"got h_block_size={h_block_size}."
        )
    if not _is_power_of_two(v_block_size):
        raise PallasUnsupportedError(
            "v_block_size must be a power of 2 for current GPU Triton lowering; "
            f"got v_block_size={v_block_size}."
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

    max_dot_tiles = _max_dot_tiles_for_device(device_kind)
    if max_dot_tiles is not None:
        dot_tiles = num_h_blocks * num_v_blocks
        if dot_tiles > max_dot_tiles:
            raise PallasUnsupportedError(
                "Kernel launch would require too many static dot tiles for this GPU and is likely to trigger "
                "pathological compile time. "
                f"dot_tiles={dot_tiles}, limit={max_dot_tiles}, "
                f"num_h_blocks={num_h_blocks}, num_v_blocks={num_v_blocks}."
            )


def _forward_pallas_gpu_kernel(
    x_ref,
    labels_ref,
    w_ref,
    out_loss_ref,
    out_lse_ref,
    *,
    v_dim: int,
    h_block_size: int,
    num_v_blocks: int,
    num_h_blocks: int,
    v_block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
):
    """Forward kernel that streams logits over V blocks per batch block."""
    m = jnp.full((x_ref.shape[0],), -jnp.inf, dtype=jnp.float32)
    l = jnp.zeros((x_ref.shape[0],), dtype=jnp.float32)
    label_logits = jnp.full((x_ref.shape[0],), -jnp.inf, dtype=jnp.float32)
    labels = labels_ref[...]

    for v_idx in range(num_v_blocks):
        v_start = v_idx * v_block_size
        logits = jnp.zeros((x_ref.shape[0], v_block_size), dtype=jnp.float32)
        for h_idx in range(num_h_blocks):
            h_start = h_idx * h_block_size
            x_block = x_ref[:, h_start : h_start + h_block_size]
            w_block = w_ref[h_start : h_start + h_block_size, v_start : v_start + v_block_size]
            logits = logits + jax.lax.dot_general(
                x_block,
                w_block,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
                precision=precision,
            )

        v_offsets = jnp.arange(v_block_size)
        in_v_block = (v_start + v_offsets) < v_dim
        logits = jnp.where(in_v_block[None, :], logits, -jnp.inf)

        if dtype is not None:
            logits = logits.astype(dtype)
        logits = _apply_logit_soft_cap(logits, logit_soft_cap)

        block_m = jnp.max(logits, axis=-1)
        m_next = jnp.maximum(m, block_m)
        block_l = jnp.sum(jnp.exp(logits - m_next[:, None]), axis=-1)
        alpha = jnp.exp(m - m_next)
        l = alpha * l + block_l
        m = m_next

        in_block = jnp.logical_and(labels >= v_start, labels < v_start + v_block_size)
        safe_local_idx = jnp.where(in_block, labels - v_start, 0)
        idx = jnp.arange(v_block_size, dtype=labels.dtype)[None, :]
        label_one_hot = jnp.logical_and(idx == safe_local_idx[:, None], in_block[:, None])
        safe_logits = jnp.where(in_v_block[None, :], logits, 0.0)
        logits_for_label = jnp.where(in_block[:, None], safe_logits, 0.0)
        block_label_logits = jnp.sum(logits_for_label * label_one_hot.astype(logits.dtype), axis=-1)
        block_label_logits = jnp.where(in_block, block_label_logits, -jnp.inf)
        label_logits = jnp.where(in_block, block_label_logits, label_logits)

    out_lse_ref[...] = (jnp.log(l) + m).astype(out_lse_ref.dtype)
    out_loss_ref[...] = (jnp.log(l) + m - label_logits).astype(out_loss_ref.dtype)


@partial(
    jax.jit,
    static_argnames=[
        "h_block_size",
        "v_dim",
        "num_v_blocks",
        "num_h_blocks",
        "v_block_size",
        "dtype",
        "logit_soft_cap",
        "precision",
    ],
)
def _linear_softmax_cross_entropy_loss_pallas_gpu_single_block(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    v_dim: int,
    h_block_size: int,
    num_v_blocks: int,
    num_h_blocks: int,
    v_block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
):
    """Single-batch-block GPU kernel launch."""
    b_dim, h_pad = x.shape
    v_pad = w.shape[1]
    out_dtype = jnp.dtype(dtype) if dtype is not None else x.dtype
    out_loss, out_lse = pl.pallas_call(
        partial(
            _forward_pallas_gpu_kernel,
            v_dim=v_dim,
            h_block_size=h_block_size,
            num_v_blocks=num_v_blocks,
            num_h_blocks=num_h_blocks,
            v_block_size=v_block_size,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        ),
        in_specs=[
            pl.BlockSpec((b_dim, h_pad), lambda _: (0, 0)),
            pl.BlockSpec((b_dim,), lambda _: (0,)),
            pl.BlockSpec((h_pad, v_pad), lambda _: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((b_dim,), lambda _: (0,)),
            pl.BlockSpec((b_dim,), lambda _: (0,)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((b_dim,), dtype=out_dtype),
            jax.ShapeDtypeStruct((b_dim,), dtype=out_dtype),
        ],
        grid=(1,),
    )(x, labels, w)
    return out_loss, out_lse


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
    """GPU Pallas implementation returning per-example loss and logsumexp."""
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
        num_v_blocks=num_v_blocks,
    )

    if num_b_blocks == 1:
        out_loss, out_lse = _linear_softmax_cross_entropy_loss_pallas_gpu_single_block(
            x_pad,
            labels_pad,
            w_pad,
            v_dim=v_dim,
            h_block_size=h_block,
            num_v_blocks=num_v_blocks,
            num_h_blocks=num_h_blocks,
            v_block_size=v_block,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
    else:
        x_chunks = x_pad.reshape((num_b_blocks, b_block, h_pad))
        labels_chunks = labels_pad.reshape((num_b_blocks, b_block))
        single_block_fn = partial(
            _linear_softmax_cross_entropy_loss_pallas_gpu_single_block,
            h_block_size=h_block,
            v_dim=v_dim,
            num_v_blocks=num_v_blocks,
            num_h_blocks=num_h_blocks,
            v_block_size=v_block,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
        out_loss, out_lse = jax.vmap(
            single_block_fn,
            in_axes=(0, 0, None),
            out_axes=(0, 0),
        )(
            x_chunks,
            labels_chunks,
            w_pad,
        )
        out_loss = out_loss.reshape((b_pad,))
        out_lse = out_lse.reshape((b_pad,))

    return out_loss[:b_dim], out_lse[:b_dim]


__all__ = ["linear_softmax_cross_entropy_loss_pallas_gpu", "PallasUnsupportedError"]
