# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Causal self-attention with a Pallas Triton forward and the FA4/CuTe backward (``gpu_pallas_fa4``).

On H100 our FA4 port's forward is the 128-thread segmented kernel, which runs well below its native SM90 backward;
the Pallas Triton softmax forward (``levanter.kernels.pallas.parallax.pallas_gpu.softmax_forward``) is about twice as
fast at head dim 128. This backend pairs the two under one ``custom_vjp``: the forward produces ``(out, lse)`` in the
FlashAttention convention and the FA4 backward consumes them, with the same per-query key bounds (causal masks,
segments, sliding windows) and the optional banded Inkling bias, whose gradient the FA4 backward computes in-kernel.
The backward config is ``gpu_fa4_cute``'s (on SM100 the native SM100 backward where validated, else the segmented
kernel). Batch/head sharding only.
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
from levanter.grug.attention._inkling_relpos import rel_extent_of_band
from levanter.kernels.pallas.parallax.pallas_gpu import softmax_forward
from levanter.sharding import partition_spec_of

FORWARD_BLOCK_Q = 128  # 4.68 ms vs 5.61 ms (64) at B 30, L 4096, H 16/8, D 128 on H100
FORWARD_BLOCK_K = 64
FORWARD_NUM_WARPS = 4


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def _attention(
    q, k, v, lower_bounds, valid, rel_bias, scale: float, kernel_config: Flash4CuteKernelConfig, plain_causal: bool
):
    return _forward(q, k, v, lower_bounds, valid, rel_bias, scale, plain_causal)[0]


def _forward(q, k, v, lower_bounds, valid, rel_bias, scale, plain_causal):
    # Plain causal masks skip the per-query bound test (about 30% of the forward at B 30, L 4096, H 16/8).
    return softmax_forward(
        q,
        k,
        v,
        None if plain_causal else lower_bounds,
        None if plain_causal else valid,
        rel_bias,
        scale=scale,
        block_q=FORWARD_BLOCK_Q,
        block_k=FORWARD_BLOCK_K,
        num_warps=FORWARD_NUM_WARPS,
        interpret=False,
    )


def _attention_fwd(q, k, v, lower_bounds, valid, rel_bias, scale, kernel_config, plain_causal):
    out, lse = _forward(q, k, v, lower_bounds, valid, rel_bias, scale, plain_causal)
    return out, (q, k, v, out, lse, lower_bounds, valid, rel_bias)


def _attention_bwd(scale, kernel_config, plain_causal, residuals, dout):
    q, k, v, out, lse, lower_bounds, valid, rel_bias = residuals
    grads = segmented_flash_attention_backward(
        q,
        k,
        v,
        out,
        dout.astype(q.dtype),
        lse,
        lower_bounds,
        valid,
        rel_bias,
        softmax_scale=scale,
        kernel_config=kernel_config,
        q_offset=jnp.zeros((1,), dtype=jnp.int32),
    )
    if rel_bias is None:
        dq, dk, dv = grads
        return dq, dk, dv, None, None, None
    dq, dk, dv, d_rel_bias = grads
    return dq, dk, dv, None, None, d_rel_bias.astype(rel_bias.dtype)


_attention.defvjp(_attention_fwd, _attention_bwd)


def gpu_pallas_fa4_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask,
    *,
    rel_bias: Float[Array, "B Hq Q W"] | None = None,
) -> Float[Array, "B Q Hq D"]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_pallas_fa4_attention requires the JAX GPU backend.")
    if q.dtype not in (jnp.bfloat16, jnp.float16):
        raise TypeError(f"gpu_pallas_fa4_attention supports only bf16/fp16, got {q.dtype}")
    if not isinstance(mask, AttentionMask) or not mask.is_causal:
        raise NotImplementedError("gpu_pallas_fa4_attention supports causal AttentionMasks only")
    if q.shape[1] != k.shape[1]:
        raise ValueError(f"self-attention requires q_len == k_len, got q={q.shape}, k={k.shape}")
    if mask.fa4_bounds is not None:
        lower_bounds, valid = mask.fa4_bounds
    else:
        lower_bounds, valid = _self_attention_lower_bounds(q, k, mask, backend_name="gpu_pallas_fa4_attention")
    # Segments and sliding windows live in the per-query key lower bounds; only a plain causal mask skips them.
    plain_causal = mask.segment_ids is None and mask.fa4_bounds is None and mask.sliding_window is None
    scale = 1.0 / math.sqrt(q.shape[-1])
    kernel_config = _segmented_kernel_config(q.shape[-1], None if rel_bias is None else rel_extent_of_band(rel_bias))

    def local(q, k, v, lower_bounds, valid, *bias):
        return _attention(q, k, v, lower_bounds, valid, bias[0] if bias else None, scale, kernel_config, plain_causal)

    bias_args = () if rel_bias is None else (rel_bias,)
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return local(q, k, v, lower_bounds, valid, *bias_args)
    q_dims = _partitioned_dims(q, mesh)
    if q_dims[1] or q_dims[3]:
        raise NotImplementedError(
            f"gpu_pallas_fa4_attention supports batch/head sharding only, got {partition_spec_of(q)}"
        )
    q_spec = tuple(partition_spec_of(q))
    out_spec = P(*q_spec, *((None,) * (q.ndim - len(q_spec))))
    metadata = P(out_spec[0], None)
    lower_bounds, valid = reshard(lower_bounds, metadata), reshard(valid, metadata)
    # rel_bias is [B, Hq, S, W]: batch and heads sharded like q's, sequence and band columns unsharded.
    bias_args = tuple(reshard(b, P(out_spec[0], out_spec[2], None, None)) for b in bias_args)
    return shard_map(local, mesh=mesh, out_specs=out_spec, check_vma=False)(q, k, v, lower_bounds, valid, *bias_args)
