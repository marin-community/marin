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
_CHUNK_TOKENS = 0


def configure_nccl_ep(top_k: int, recv_capacity_per_rank: int, chunk_tokens_per_rank: int = 0) -> None:
    """Record the per-layer EP config after ``ep_bootstrap``.

    One shared ``EpLayerConfig`` serves every layer (TE's per-step cache keys on
    handle_mem, not on the config object). Module-level because the TE EP
    backend is itself a process-global singleton.

    ``chunk_tokens_per_rank`` > 0 splits each rank's token stream into
    fixed-size chunks and loops dispatch → FFN → combine per chunk
    (``lax.scan``). ``recv_capacity_per_rank`` and the bootstrap's
    ``max_tokens_per_rank`` must then be sized for ONE chunk, decoupling EP
    buffer memory from the global batch (the no-drop capacity wall, NCCLEP-006).
    """
    global _LAYER_CFG, _RECV_CAPACITY, _CHUNK_TOKENS
    if EpLayerConfig is None:
        raise ModuleNotFoundError(
            "moe_implementation='nccl_ep' requires a transformer-engine build with NCCL_EP"
        ) from _TE_IMPORT_ERROR
    _LAYER_CFG = EpLayerConfig(top_k=top_k, dispatch_output_per_expert_alignment=_DISPATCH_ALIGNMENT)
    _RECV_CAPACITY = int(recv_capacity_per_rank)
    _CHUNK_TOKENS = int(chunk_tokens_per_rank)


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

    lead = P(batch_spec[0], None, None)  # (outer, ep) leading axis of EP-output tensors
    lead2 = P(batch_spec[0], None)
    w_spec = P("expert", None, None)

    def _local_ffn(recv_tokens_l, token_counts_l, w13_l, w2_l):
        x_dispatch = recv_tokens_l.reshape(recv_tokens_l.shape[-2], recv_tokens_l.shape[-1])
        group_sizes = token_counts_l.reshape(-1).astype(jnp.int32)
        # The QuACK seam extends the last group over the capacity tail so every
        # output row is defined — zero the tail first: those rows are
        # uninitialized dispatch buffer, and garbage (Inf/NaN bit patterns)
        # poisons the wgrad accumulation even under zero cotangents.
        total = jnp.sum(group_sizes)
        row_ids = jnp.arange(x_dispatch.shape[0], dtype=jnp.int32)
        x_dispatch = jnp.where((row_ids < total)[:, None], x_dispatch, 0)
        # extend_tail_group=False: the no-drop capacity tail is up to
        # (ep-1)/ep of the buffer; running GEMMs over it multiplies FFN work.
        # Unwritten output rows are where-selected away before ep_combine.
        expert_mlp_fn = _quack_expert_mlp_fn(
            w13_l, w2_l, implementation="nccl_ep", extend_tail_group=False
        )
        out = expert_mlp_fn(x_dispatch, group_sizes)
        return out.reshape(recv_tokens_l.shape)

    lead_axes = batch_spec[0]
    shard_axis_names = (lead_axes,) if isinstance(lead_axes, str) else tuple(lead_axes)
    shards = 1
    for name in shard_axis_names:
        shards *= mesh.shape[name]

    def _one_chunk(x_b, topk_idx_b, combine_w_b, w13_b, w2_b):
        recv_tokens, recv_w, handle_mem, token_counts = ep_dispatch(
            cfg, topk_idx_b, x_b, combine_w_b, recv_capacity
        )
        # Pin the EP-output tensors to the (outer, ep) layout (same move as
        # TE's own multi-process test) so the FFN shard_map sees the specs.
        recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, lead)
        recv_w = jax.lax.with_sharding_constraint(recv_w, lead2)
        token_counts = jax.lax.with_sharding_constraint(token_counts, lead2)

        # Inside auto_axes the context mesh is the auto-typed view of `mesh`;
        # shard_map must receive that view, not the outer Explicit object.
        ffn = shard_map(
            _local_ffn,
            mesh=jax.sharding.get_abstract_mesh(),
            in_specs=(lead, lead2, w_spec, w_spec),
            out_specs=lead,
            check_vma=False,
        )
        expert_out = ffn(recv_tokens, token_counts, w13_b, w2_b)

        # ep_combine is unweighted: apply the routing weights before the
        # scatter-sum; grad w.r.t. combine_weights flows through this
        # hadamard, not the FFI. `where` (not a 0-mask multiply): padded slots
        # can hold garbage NaN/Inf and 0*NaN = NaN, in both fwd and the VJP.
        w_slot = recv_w[..., None].astype(expert_out.dtype)
        weighted = jnp.where(w_slot != 0, expert_out * w_slot, jnp.zeros((), expert_out.dtype))
        weighted = jax.lax.with_sharding_constraint(weighted, lead)
        out = ep_combine(cfg, handle_mem, token_counts, weighted, tuple(x_b.shape[:-1]))
        return out.astype(x_b.dtype)

    def _body(x_b, topk_idx_b, combine_w_b, w13_b, w2_b):
        tokens_per_rank = x_b.shape[0] // shards
        if _CHUNK_TOKENS == 0 or _CHUNK_TOKENS >= tokens_per_rank:
            return _one_chunk(x_b, topk_idx_b, combine_w_b, w13_b, w2_b)
        if tokens_per_rank % _CHUNK_TOKENS != 0:
            raise ValueError(
                f"chunk_tokens_per_rank={_CHUNK_TOKENS} must divide per-rank tokens {tokens_per_rank}"
            )
        num_chunks = tokens_per_rank // _CHUNK_TOKENS

        # Rank-local chunking: [T, ...] -> [K, shards, C, ...] where the
        # `shards` dim stays pinned to the batch mesh axes, so the reshape and
        # moveaxis shuffle only rank-local rows (no cross-rank resharding).
        def to_chunks(t):
            t = t.reshape(shards, num_chunks, _CHUNK_TOKENS, *t.shape[1:])
            t = jax.lax.with_sharding_constraint(t, P(lead_axes, *(None,) * (t.ndim - 1)))
            t = jnp.moveaxis(t, 1, 0)
            return jax.lax.with_sharding_constraint(t, P(None, lead_axes, *(None,) * (t.ndim - 2)))

        chunk_spec = P(lead_axes, None, None)

        # Checkpoint the chunk body: without it, scan saves each chunk's
        # capacity-sized tensors as bwd residuals and K × (ep × C × top_k)
        # rows = the SAME total as one unchunked dispatch — the b1024 memory
        # wall reappears (NCCLEP-007: 164.93 GiB OOM at C=16384). Recompute
        # dispatch+FFN per chunk in the bwd instead; only the [C,·] chunk
        # inputs are stacked. prevent_cse=False: scan iteration boundaries
        # already prevent the CSE remat guards against.
        chunk_fn = jax.checkpoint(_one_chunk, prevent_cse=False)

        def scan_fn(carry, chunk):
            x_c, idx_c, w_c = (t.reshape(shards * _CHUNK_TOKENS, t.shape[-1]) for t in chunk)
            x_c = jax.lax.with_sharding_constraint(x_c, P(lead_axes, None))
            out_c = chunk_fn(x_c, idx_c, w_c, w13_b, w2_b)
            out_c = out_c.reshape(shards, _CHUNK_TOKENS, out_c.shape[-1])
            return carry, jax.lax.with_sharding_constraint(out_c, chunk_spec)

        _, outs = jax.lax.scan(scan_fn, 0, (to_chunks(x_b), to_chunks(topk_idx_b), to_chunks(combine_w_b)))
        outs = jnp.moveaxis(outs, 0, 1)  # [shards, K, C, H]
        outs = jax.lax.with_sharding_constraint(outs, P(lead_axes, None, None, None))
        return outs.reshape(x_b.shape[0], x_b.shape[-1])

    # The bench mesh types every axis Explicit, where sharding constraints are
    # asserts and TE's partitioning rules (written for auto-sharding) type the
    # FFI outputs replicated. Run the whole EP region under auto axes and pin
    # only the final output back to the explicit batch spec.
    out = jax.sharding.auto_axes(
        _body, axes=tuple(mesh.axis_names), out_sharding=batch_spec
    )(x, selected_experts.astype(jnp.int32), combine_weights.astype(jnp.float32), w_up_gate, w_down)

    dropped = jnp.zeros((), dtype=jnp.int32)
    return out, dropped
