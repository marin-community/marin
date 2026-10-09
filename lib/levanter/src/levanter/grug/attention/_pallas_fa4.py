# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Causal self-attention with a Pallas Triton forward and the FA4/CuTe backward (``gpu_pallas_fa4``).

On H100 our FA4 port's forward is the 128-thread segmented kernel, which runs well below its native SM90 backward;
the Pallas Triton softmax forward (``levanter.kernels.pallas.parallax.pallas_gpu.softmax_forward``) is about twice as
fast at head dim 128. This backend pairs the two under one ``custom_vjp``: the forward produces ``(out, lse)`` in the
FlashAttention convention and the FA4 backward consumes them. Supports causal masks with segment ids (no sliding
window), batch/head sharding only.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float

from levanter.grug.attention._core import AttentionMask
from levanter.grug.attention._fa4_cute import _partitioned_dims, _segmented_kernel_config, _self_attention_lower_bounds
from levanter.grug.attention._fa4_cute_backend import segmented_flash_attention_backward
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig
from levanter.kernels.pallas.parallax.pallas_gpu import softmax_forward
from levanter.sharding import partition_spec_of

FORWARD_BLOCK_Q = 128  # 4.68 ms vs 5.61 ms (64) at B 30, L 4096, H 16/8, D 128 on H100
FORWARD_BLOCK_K = 64
FORWARD_NUM_WARPS = 4


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _attention(q, k, v, segment_ids, lower_bounds, valid, scale: float, kernel_config: Flash4CuteKernelConfig):
    return _forward(q, k, v, segment_ids, scale)[0]


def _forward(q, k, v, segment_ids, scale):
    return softmax_forward(
        q,
        k,
        v,
        segment_ids,
        scale=scale,
        block_q=FORWARD_BLOCK_Q,
        block_k=FORWARD_BLOCK_K,
        num_warps=FORWARD_NUM_WARPS,
        interpret=False,
    )


def _attention_fwd(q, k, v, segment_ids, lower_bounds, valid, scale, kernel_config):
    out, lse = _forward(q, k, v, segment_ids, scale)
    return out, (q, k, v, out, lse, segment_ids, lower_bounds, valid)


def _attention_bwd(scale, kernel_config, residuals, dout):
    q, k, v, out, lse, segment_ids, lower_bounds, valid = residuals
    dq, dk, dv = segmented_flash_attention_backward(
        q,
        k,
        v,
        out,
        dout.astype(q.dtype),
        lse,
        lower_bounds,
        valid,
        softmax_scale=scale,
        kernel_config=kernel_config,
        q_offset=jnp.zeros((1,), dtype=jnp.int32),
    )[:3]
    return dq, dk, dv, None, None, None


_attention.defvjp(_attention_fwd, _attention_bwd)


def gpu_pallas_fa4_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask,
) -> Float[Array, "B Q Hq D"]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_pallas_fa4_attention requires the JAX GPU backend.")
    if q.dtype not in (jnp.bfloat16, jnp.float16):
        raise TypeError(f"gpu_pallas_fa4_attention supports only bf16/fp16, got {q.dtype}")
    if not isinstance(mask, AttentionMask) or not mask.is_causal or mask.sliding_window is not None:
        raise NotImplementedError("gpu_pallas_fa4_attention supports causal AttentionMasks without a sliding window")
    if q.shape[1] != k.shape[1]:
        raise ValueError(f"self-attention requires q_len == k_len, got q={q.shape}, k={k.shape}")
    lower_bounds, valid = _self_attention_lower_bounds(q, k, mask, backend_name="gpu_pallas_fa4_attention")
    segment_ids = None if mask.segment_ids is None else mask.segment_ids[0].astype(jnp.int32)
    scale = 1.0 / math.sqrt(q.shape[-1])
    kernel_config = _segmented_kernel_config(q.shape[-1])
    if segment_ids is not None:
        segment_ids = jnp.broadcast_to(segment_ids, (q.shape[0], q.shape[1]))

    def local(q, k, v, lower_bounds, valid, *seg):
        return _attention(q, k, v, seg[0] if seg else None, lower_bounds, valid, scale, kernel_config)

    seg_args = () if segment_ids is None else (segment_ids,)
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return local(q, k, v, lower_bounds, valid, *seg_args)
    q_dims = _partitioned_dims(q, mesh)
    if q_dims[1] or q_dims[3]:
        raise NotImplementedError(
            f"gpu_pallas_fa4_attention supports batch/head sharding only, got {partition_spec_of(q)}"
        )
    q_spec = tuple(partition_spec_of(q))
    out_spec = P(*q_spec, *((None,) * (q.ndim - len(q_spec))))
    metadata = P(out_spec[0], None)
    lower_bounds, valid = reshard(lower_bounds, metadata), reshard(valid, metadata)
    seg_args = tuple(reshard(s, metadata) for s in seg_args)
    return shard_map(local, mesh=mesh, out_specs=out_spec, check_vma=False)(q, k, v, lower_bounds, valid, *seg_args)
