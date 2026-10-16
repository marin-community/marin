# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Grug's ``gpu_aiter_flash`` attention: AMD AITER's flash attention through JAX-AITER (https://github.com/ROCm/jax-aiter).

JAX-AITER wraps AITER's CK and assembly flash-attention kernels in XLA FFI calls with a ``custom_vjp``. It is not a
Marin dependency: it ships ``gfx950``-only wheels for the TheRock ROCm stack (Ubuntu 24.04, glibc 2.39), so install
it into the environment that runs the job. AITER's assembly backward covers causal attention; sliding-window layers
fall back to its CK backward.

Supported masks: causal, optionally with a sliding window. Packed segment IDs would need JAX-AITER's varlen entry
point and are rejected.
"""

import importlib

from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh
from jaxtyping import Array, Float

from levanter.grug.attention._core import AttentionMask
from levanter.sharding import partition_spec_of, partitioned_dims

# The non-deterministic backward accumulates dQ with atomics, as FlashAttention's default training path does.
DETERMINISTIC_BACKWARD = False


def _window_size(mask: AttentionMask) -> tuple[int, int]:
    """FlashAttention's (left, right) window: ``sliding_window=W`` keeps keys ``q - (W - 1) <= k <= q``.

    Without a window this is ``(-1, -1)``, not ``(-1, 0)``: ``causal=True`` already bounds keys on the right, and
    JAX-AITER treats any right bound >= 0 as a sliding window, which rules out AITER's assembly backward.
    """
    if mask.sliding_window is None:
        return (-1, -1)
    if mask.sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
    return (mask.sliding_window - 1, 0)


def aiter_flash_attention(
    q: Float[Array, "B S Hq D"],
    k: Float[Array, "B S Hkv D"],
    v: Float[Array, "B S Hkv D"],
    mask: AttentionMask,
) -> Float[Array, "B S Hq D"]:
    """Causal, optionally windowed, self-attention through JAX-AITER's ``flash_attn_func``.

    The softmax scale is ``1 / sqrt(head_dim)``, as in ``reference_attention``. Under a mesh the kernel runs per shard
    inside ``shard_map``; batch and heads may be sharded, sequence and head_dim may not.
    """
    if not mask.is_causal:
        raise NotImplementedError("gpu_aiter_flash supports causal self-attention only")
    if mask.segment_ids is not None:
        raise NotImplementedError("gpu_aiter_flash does not support segment ids")
    if mask.fa4_bounds is not None:
        raise NotImplementedError("gpu_aiter_flash does not read FA4/CuTe precomputed bounds")
    flash_attn_func = importlib.import_module("jax_aiter.mha").flash_attn_func
    window = _window_size(mask)

    def run(q, k, v):
        # flash_attn_func returns a tuple even when no LSE or probabilities are requested.
        (out,) = flash_attn_func(q, k, v, causal=True, window_size=window, deterministic=DETERMINISTIC_BACKWARD)
        return out

    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return run(q, k, v)
    q_dims = partitioned_dims(q, mesh)
    if q_dims[1] or q_dims[3]:
        raise ValueError(f"gpu_aiter_flash needs unsharded sequence and head_dim, got {partition_spec_of(q)}")
    for name, x in (("k", k), ("v", v)):
        if partitioned_dims(x, mesh) != q_dims:
            raise ValueError(
                f"gpu_aiter_flash needs {name} sharded like q, got q={partition_spec_of(q)} "
                f"{name}={partition_spec_of(x)}"
            )
    if not any(q_dims):
        return run(q, k, v)
    q_spec = tuple(partition_spec_of(q))
    q_spec = P(*q_spec, *((None,) * (q.ndim - len(q_spec))))

    @shard_map(mesh=mesh, out_specs=q_spec, check_vma=False)
    def _local(q_local, k_local, v_local):
        return run(q_local, k_local, v_local)

    return _local(q, k, v)
