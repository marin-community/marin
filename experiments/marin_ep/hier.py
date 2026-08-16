# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-hop hierarchical transport plans: ragged internode + fused intranode.

EP64 runs as 16 processes x 4 GPUs. The fused Mosaic kernel only reaches
process-local peers (MEP-024), so dispatch becomes two hops:

- Hop A (internode): source ``(n_s, g)`` sends each row to the same-local-rank
  peer ``(n_d, g)`` on the expert owner's node, via ``ragged_all_to_all`` over
  the "node" sub-axis. The compacted send buffer is global-expert-major, and
  experts are blocked by owner device, so per-destination-node slices are
  already contiguous.
- Hop B (intranode): the staging buffer at ``(n_d, g)`` — per-source-node
  chunks, each expert-major over ``n_d``'s experts — scatters into the owners'
  expert-major pools with the fused ``put_segments`` kernel over the "gpu"
  sub-axis. One plan entry per (destination gpu, node-local expert, source
  node): ``nodes * local_experts`` entries per destination, uniform.

The final pool at ``(n_d, g_d)`` keeps the single-hop region layout (one
compacted block per local expert, sized by the waterfilled ``kept``); only the
within-expert source order changes, from global-device order to
``(source gpu, source node)`` order. Combine transposes both hops, so rows
return to exactly their compacted send positions.

Index conventions: global device ``d = n * G + g``; global expert ``e`` is
owned by device ``e // El``; node of expert ``e`` is ``e // (El * G)``.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int
from levanter.grug._moe.marin_ep_transport import SegmentPlan, _rotated


def hop_a_node_counts(accepted: Int[Array, "S E"], *, nodes: int, gpus: int) -> Int[Array, "S Nodes"]:
    """Rows each source device sends to each destination node (hop A sizes)."""
    num_experts = accepted.shape[1]
    node_of_expert = jnp.arange(num_experts) // (num_experts // nodes)
    return jax.ops.segment_sum(accepted.T, node_of_expert, num_segments=nodes).T


def hop_a_params(
    accepted: Int[Array, "S E"],
    node_id: jax.Array,
    gpu_id: jax.Array,
    *,
    nodes: int,
    gpus: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``ragged_all_to_all`` offsets/sizes for hop A over the "node" axis.

    Sends: my compacted expert-major buffer, one contiguous slice per
    destination node. Receive layout at ``(n_d, g)``: source-node-order
    concatenation of every same-rank source's slice for ``n_d``.
    """
    counts = hop_a_node_counts(accepted, nodes=nodes, gpus=gpus)  # [S, Nodes]
    me = node_id * gpus + gpu_id
    my_counts = counts[me]  # [Nodes]
    input_offsets = jnp.cumsum(my_counts) - my_counts
    send_sizes = my_counts
    # Receivers with my local rank, in node order; my slice lands after
    # every lower-numbered source node's slice.
    same_rank = counts.reshape(nodes, gpus, nodes)[:, gpu_id, :]  # [Nodes_src, Nodes_dst]
    recv_before = jnp.cumsum(same_rank, axis=0) - same_rank  # rows ahead of source n_s at dest n_d
    output_offsets = recv_before[node_id, :]
    recv_sizes = same_rank[:, node_id]
    return (
        input_offsets.astype(jnp.int32),
        send_sizes.astype(jnp.int32),
        output_offsets.astype(jnp.int32),
        recv_sizes.astype(jnp.int32),
    )


def stage_rows(
    accepted: Int[Array, "S E"], node_id: jax.Array, gpu_id: jax.Array, *, nodes: int, gpus: int
) -> jax.Array:
    """Total staging-buffer rows at ``(node_id, gpu_id)`` after hop A."""
    counts = hop_a_node_counts(accepted, nodes=nodes, gpus=gpus)
    return jnp.sum(counts.reshape(nodes, gpus, nodes)[:, gpu_id, node_id], dtype=jnp.int32)


def _node_expert_ids(node_id: jax.Array, *, nodes: int, gpus: int, local_experts: int) -> jax.Array:
    """Global expert ids owned by ``node_id``, in (gpu, local expert) order."""
    per_node = gpus * local_experts
    return node_id * per_node + jnp.arange(per_node, dtype=jnp.int32)


def hier_dispatch_segments(
    accepted: Int[Array, "S E"],
    region: Int[Array, " E"],
    node_id: jax.Array,
    gpu_id: jax.Array,
    *,
    nodes: int,
    gpus: int,
    local_experts: int,
) -> SegmentPlan:
    """Hop-B plan: my staging buffer -> intranode owners' pool regions.

    ``region`` holds each expert's start inside its owner's pool (global
    ``[E]``, same as the single-hop layout). Entry ``(g_d, e_local, n_s)``
    copies source node ``n_s``'s rows for expert ``(node_id, g_d, e_local)``
    from my staging buffer into ``g_d``'s pool at the (my gpu, ``n_s``)
    source rank.
    """
    per_node = gpus * local_experts
    node_experts = _node_expert_ids(node_id, nodes=nodes, gpus=gpus, local_experts=local_experts)  # [per_node]
    # accepted rows from source (n_s, my gpu) for each of my node's experts.
    acc_src = accepted.reshape(nodes, gpus, -1)[:, gpu_id, :][:, node_experts]  # [Nodes_src, per_node]

    # Staging layout: node-order chunks; inside chunk n_s, expert-major over
    # my node's experts.
    chunk_rows = jnp.sum(acc_src, axis=1)  # [Nodes_src]
    chunk_start = jnp.cumsum(chunk_rows) - chunk_rows
    within = jnp.cumsum(acc_src, axis=1) - acc_src  # [Nodes_src, per_node]
    src_lo = chunk_start[:, None] + within  # [Nodes_src, per_node]

    # Pool destination: region[e] + rows ahead of (my gpu, n_s) in the
    # (source gpu, source node) order.
    acc_by_gpu = accepted.reshape(nodes, gpus, -1)[:, :, node_experts]  # [Nodes_src, G_src, per_node]
    gpu_totals = jnp.sum(acc_by_gpu, axis=0)  # [G_src, per_node]
    before_gpu = (jnp.cumsum(gpu_totals, axis=0) - gpu_totals)[gpu_id]  # [per_node]
    before_node = jnp.cumsum(acc_by_gpu[:, gpu_id, :], axis=0) - acc_by_gpu[:, gpu_id, :]  # [Nodes_src, per_node]
    dst_lo = region[node_experts][None, :] + before_gpu[None, :] + before_node  # [Nodes_src, per_node]

    dest_order = _rotated(gpu_id, gpus)  # [G]
    # Entries for dest g_d: its local_experts columns, all source nodes.
    col = dest_order[:, None, None] * local_experts + jnp.arange(local_experts)[None, None, :]  # [G,1,El]
    node_idx = jnp.arange(nodes, dtype=jnp.int32)[None, :, None]  # [1,Nodes,1]
    gather = (node_idx * per_node + col).reshape(-1)  # [G*Nodes*El] flat (Nodes, per_node) index
    entries = nodes * local_experts
    return SegmentPlan(
        dest_ids=dest_order,
        entry_start=jnp.arange(gpus + 1, dtype=jnp.int32) * entries,
        src_lo=src_lo.reshape(-1)[gather].astype(jnp.int32),
        dst_lo=dst_lo.reshape(-1)[gather].astype(jnp.int32),
        rows=acc_src.reshape(-1)[gather].astype(jnp.int32),
    )


def hier_combine_segments(
    accepted: Int[Array, "S E"],
    region: Int[Array, " E"],
    node_id: jax.Array,
    gpu_id: jax.Array,
    *,
    nodes: int,
    gpus: int,
    local_experts: int,
) -> SegmentPlan:
    """Hop-B' plan: my expert-output pool rows -> intranode staging buffers.

    Exact transpose of ``hier_dispatch_segments``: for my expert
    ``(node_id, my gpu, e_local)``, the rows contributed by staging device
    ``(node_id, g_s)`` on behalf of source node ``n_s`` return to ``g_s``'s
    staging buffer at the position hop A deposited them.
    """
    per_node = gpus * local_experts
    node_experts = _node_expert_ids(node_id, nodes=nodes, gpus=gpus, local_experts=local_experts)
    acc_by_gpu = accepted.reshape(nodes, gpus, -1)[:, :, node_experts]  # [Nodes_src, G_src, per_node]
    my_bank = gpu_id * local_experts + jnp.arange(local_experts, dtype=jnp.int32)  # cols of my experts

    # Pool offsets of the (g_s, n_s) block for my experts.
    gpu_totals = jnp.sum(acc_by_gpu, axis=0)  # [G_src, per_node]
    before_gpu = jnp.cumsum(gpu_totals, axis=0) - gpu_totals  # [G_src, per_node]
    before_node = jnp.cumsum(acc_by_gpu, axis=0) - acc_by_gpu  # [Nodes_src, G_src, per_node]
    pool_lo = region[node_experts][None, None, :] + before_gpu[None, :, :] + before_node  # [Nodes,G,per_node]

    # Staging offsets on device (node_id, g_s): chunk/segment starts.
    chunk_rows = jnp.sum(acc_by_gpu, axis=2)  # [Nodes_src, G_src]
    chunk_start = jnp.cumsum(chunk_rows, axis=0) - chunk_rows  # [Nodes_src, G_src]
    within = jnp.cumsum(acc_by_gpu, axis=2) - acc_by_gpu  # [Nodes_src, G_src, per_node]
    stage_lo = chunk_start[:, :, None] + within  # [Nodes_src, G_src, per_node]

    dest_order = _rotated(gpu_id, gpus)
    entries = nodes * local_experts
    # Entries for dest staging gpu g_s: my experts x all source nodes.
    for_my = acc_by_gpu[:, :, my_bank]  # [Nodes_src, G_src, El]
    pool_my = pool_lo[:, :, my_bank]
    stage_my = stage_lo[:, :, my_bank]
    src_parts = pool_my.transpose(1, 0, 2)[dest_order]  # [G, Nodes, El]
    dst_parts = stage_my.transpose(1, 0, 2)[dest_order]
    row_parts = for_my.transpose(1, 0, 2)[dest_order]
    return SegmentPlan(
        dest_ids=dest_order,
        entry_start=jnp.arange(gpus + 1, dtype=jnp.int32) * entries,
        src_lo=src_parts.reshape(-1).astype(jnp.int32),
        dst_lo=dst_parts.reshape(-1).astype(jnp.int32),
        rows=row_parts.reshape(-1).astype(jnp.int32),
    )
