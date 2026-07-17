# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""TransformerEngine NCCL_EP expert-parallel Grug MoE backend (issue #7331).

Unlike the ring/a2a backends (whole computation inside one shard_map), NCCL_EP
runs dispatch/combine as global-view TE FFI primitives with their own sharding
rules and custom VJPs; only the grouped expert FFN runs inside shard_map. This
mirrors TE's own MoE block: dispatch -> shard_map(grouped FFN) -> weighted
hadamard -> combine (the combine FFI is unweighted).

Process-global requirements (caller responsibility, see the standalone bench):
- one process per GPU (``jax.local_device_count() == 1``),
- ``te.jax.ep.ep_bootstrap`` called once per process inside the active mesh and
  a ``global_shard_guard(MeshResource(fsdp_resource="data", ep_resource="expert"))``,
- ``configure_nccl_ep`` called after bootstrap to record the layer config,
- TE imported before the JAX CUDA client exists (FFI handler registration).

TE constraint: exactly ONE dp/fsdp mesh axis outside ``expert`` — run with
``replica_axis_size=1`` (single model copy; FSDP spans ``data``).
"""

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P

from levanter.grug._moe.ep_common import _quack_expert_mlp_fn

try:
    from transformer_engine.jax.ep import EpLayerConfig, ep_combine, ep_dispatch

    _TE_IMPORT_ERROR = None
except ImportError as _e:  # optional dep: transformer-engine with NCCL_EP
    EpLayerConfig = ep_combine = ep_dispatch = None
    _TE_IMPORT_ERROR = _e

# NCCL EP HT-mode TMA alignment for per-expert dispatch segments (matches TE's
# own MoE block and the TE test suite).
_DISPATCH_ALIGNMENT = 16

_LAYER_CFG = None
_RECV_CAPACITY = None


def configure_nccl_ep(top_k: int, recv_capacity_per_rank: int) -> None:
    """Record the per-layer EP config after ``ep_bootstrap``.

    One shared ``EpLayerConfig`` serves every layer (TE's per-step cache keys on
    handle_mem, not on the config object). Module-level because the TE EP
    backend is itself a process-global singleton.
    """
    global _LAYER_CFG, _RECV_CAPACITY
    if EpLayerConfig is None:
        raise ModuleNotFoundError(
            "moe_implementation='nccl_ep' requires a transformer-engine build with NCCL_EP"
        ) from _TE_IMPORT_ERROR
    _LAYER_CFG = EpLayerConfig(top_k=top_k, dispatch_output_per_expert_alignment=_DISPATCH_ALIGNMENT)
    _RECV_CAPACITY = int(recv_capacity_per_rank)


def _moe_mlp_ep_nccl(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh,
    batch_spec,
) -> tuple[jax.Array, jax.Array]:
    """Global-view NCCL_EP MoE: TE dispatch -> shard_map(QuACK FFN) -> TE combine.

    ``x`` [T, H] and routing tensors are sharded over the batch axes; expert
    weights arrive fully materialized per expert shard (P("expert", ...)).
    Returns ``(out, dropped)``; drops beyond ``recv_capacity`` are not counted
    per-step by TE, so ``dropped`` is a constant 0 placeholder.
    """
    if _LAYER_CFG is None:
        raise RuntimeError("configure_nccl_ep() must be called after ep_bootstrap before tracing")
    cfg = _LAYER_CFG
    recv_capacity = _RECV_CAPACITY

    topk_idx = selected_experts.astype(jnp.int32)
    recv_tokens, recv_w, handle_mem, token_counts = ep_dispatch(
        cfg, topk_idx, x, combine_weights.astype(jnp.float32), recv_capacity
    )

    def _local_ffn(recv_tokens_l, token_counts_l, w13_l, w2_l):
        x_dispatch = recv_tokens_l.reshape(recv_tokens_l.shape[-2], recv_tokens_l.shape[-1])
        group_sizes = token_counts_l.reshape(-1).astype(jnp.int32)
        expert_mlp_fn = _quack_expert_mlp_fn(w13_l, w2_l, implementation="nccl_ep")
        out = expert_mlp_fn(x_dispatch, group_sizes)
        return out.reshape(recv_tokens_l.shape)

    lead = P(batch_spec[0], None, None)  # (outer, ep) leading axis of EP-output tensors
    counts_spec = P(batch_spec[0], None)
    w_spec = P("expert", None, None)
    ffn = shard_map(
        _local_ffn,
        mesh=mesh,
        in_specs=(lead, counts_spec, w_spec, w_spec),
        out_specs=lead,
        check_vma=False,
    )
    expert_out = ffn(recv_tokens, token_counts, w_up_gate, w_down)

    # ep_combine is unweighted: apply the routing weights (zero-masked — padded
    # slots carry weight 0) before the scatter-sum; grad w.r.t. combine_weights
    # flows through this hadamard, not the FFI.
    mask = (recv_w != 0).astype(jnp.float32)[..., None]
    weighted = (expert_out.astype(jnp.float32) * recv_w[..., None] * mask).astype(expert_out.dtype)
    out = ep_combine(cfg, handle_mem, token_counts, weighted, tuple(x.shape[:-1]))
    out = jax.lax.with_sharding_constraint(out, jax.sharding.NamedSharding(mesh, batch_spec))

    dropped = jnp.zeros((), dtype=jnp.int32)
    return out.astype(x.dtype), dropped
