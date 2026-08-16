# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-hop hierarchical transport plans vs a direct NumPy reference.

Exercises the levanter builders on the host: hop A (flat-axis ragged layout
from `_shard_a2a_params(hier_flat_counts(...))`), hop B
(`hier_dispatch_segments` executed by the shared plan interpreter), then the
reverse path (`hier_combine_segments` + hop-A transpose). The pool must match
a from-definition reference and the round trip must restore every kept row.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
from levanter.grug._moe.ep_common import _shard_a2a_params
from levanter.grug._moe.marin_ep_transport import (
    hier_combine_segments,
    hier_dispatch_segments,
    hier_flat_counts,
)

from experiments.marin_ep.planref import execute_plans

NODES = 3
GPUS = 4
LOCAL_EXPERTS = 2
DEVICES = NODES * GPUS
EXPERTS = DEVICES * LOCAL_EXPERTS
HIDDEN = 8
MAX_ROWS = 5


def _setup(seed: int = 3):
    rng = np.random.default_rng(seed)
    accepted = rng.integers(0, MAX_ROWS, size=(DEVICES, EXPERTS)).astype(np.int32)
    accepted[0, 0] = 0
    accepted[2, 5] = 0
    kept = accepted.sum(axis=0)
    kept_by_owner = kept.reshape(DEVICES, LOCAL_EXPERTS)
    region = (np.cumsum(kept_by_owner, axis=1) - kept_by_owner).reshape(EXPERTS).astype(np.int32)
    pool_rows = int(kept_by_owner.sum(axis=1).max())

    # Compacted expert-major send buffers with globally unique row tags.
    sends, seg_starts = [], []
    for d in range(DEVICES):
        seg = np.cumsum(accepted[d]) - accepted[d]
        n = int(accepted[d].sum())
        buf = np.zeros((max(n, 1), HIDDEN), np.float32)
        buf[:n] = d * 1e6 + np.arange(n)[:, None] + np.arange(HIDDEN)[None, :] / 1e3
        sends.append(buf[:n] if n else np.zeros((0, HIDDEN), np.float32))
        seg_starts.append(seg)
    return accepted, region, pool_rows, sends, seg_starts


def _expert_slice(sends, seg_starts, accepted, d: int, e: int) -> np.ndarray:
    lo = seg_starts[d][e]
    return sends[d][lo : lo + accepted[d, e]]


def _emulate_hop_a(accepted, sends, seg_starts):
    """Staging buffers keyed by flat device id, per the flat ragged layout."""
    stagings = {}
    for n_d in range(NODES):
        for g in range(GPUS):
            chunks = []
            for n_s in range(NODES):
                d = n_s * GPUS + g
                node_experts = range(n_d * GPUS * LOCAL_EXPERTS, (n_d + 1) * GPUS * LOCAL_EXPERTS)
                rows = [_expert_slice(sends, seg_starts, accepted, d, e) for e in node_experts]
                chunks.append(np.concatenate(rows, axis=0) if rows else np.zeros((0, HIDDEN), np.float32))
            stagings[n_d * GPUS + g] = np.concatenate(chunks, axis=0)
    return stagings


def test_flat_hop_a_params_match_emulated_layout():
    accepted, _, _, sends, seg_starts = _setup()
    stagings = _emulate_hop_a(accepted, sends, seg_starts)
    counts_flat = np.asarray(hier_flat_counts(jnp.asarray(accepted), nodes=NODES, gpus=GPUS))
    # Different-rank cells are zero; row sums preserve every accepted row.
    for s in range(DEVICES):
        for d in range(DEVICES):
            if s % GPUS != d % GPUS:
                assert counts_flat[s, d] == 0
    np.testing.assert_array_equal(counts_flat.sum(axis=1), accepted.sum(axis=1))

    for s in range(DEVICES):
        in_off, send_sz, out_off, _recv_sz = (
            np.asarray(a) for a in _shard_a2a_params(jnp.asarray(counts_flat), jnp.int32(s))
        )
        for d in range(DEVICES):
            got = stagings[d][out_off[d] : out_off[d] + send_sz[d]]
            want = sends[s][in_off[d] : in_off[d] + send_sz[d]]
            np.testing.assert_array_equal(got, want, err_msg=f"src {s} -> dst {d}")


def _node_local(plan, node: int):
    """Rebase a flat-dest plan onto node-local output indices for the interpreter."""
    return dataclasses.replace(plan, dest_ids=plan.dest_ids - jnp.int32(node * GPUS))


def test_two_hop_dispatch_matches_reference_pool_and_round_trips():
    accepted, region, pool_rows, sends, seg_starts = _setup()
    stagings = _emulate_hop_a(accepted, sends, seg_starts)
    acc = jnp.asarray(accepted)
    reg = jnp.asarray(region)

    pools = {}
    for n_d in range(NODES):
        plans = [
            _node_local(
                hier_dispatch_segments(
                    acc, reg, jnp.int32(n_d * GPUS + g), nodes=NODES, gpus=GPUS, local_experts=LOCAL_EXPERTS
                ),
                n_d,
            )
            for g in range(GPUS)
        ]
        outs = execute_plans(plans, [stagings[n_d * GPUS + g] for g in range(GPUS)], pool_rows)
        for g in range(GPUS):
            pools[n_d * GPUS + g] = outs[g]

    # Reference pool: per expert, (source gpu, source node) order.
    for owner in range(DEVICES):
        expect_rows = []
        for e_local in range(LOCAL_EXPERTS):
            e = owner * LOCAL_EXPERTS + e_local
            for g_s in range(GPUS):
                for n_s in range(NODES):
                    expect_rows.append(_expert_slice(sends, seg_starts, accepted, n_s * GPUS + g_s, e))
        want = np.concatenate(expect_rows, axis=0)
        np.testing.assert_array_equal(pools[owner][: want.shape[0]], want, err_msg=f"pool {owner}")

    # Combine: pool -> staging (hop-B transpose), then hop-A transpose -> sources.
    counts_flat = np.asarray(hier_flat_counts(acc, nodes=NODES, gpus=GPUS))
    for n_d in range(NODES):
        plans = [
            _node_local(
                hier_combine_segments(
                    acc, reg, jnp.int32(n_d * GPUS + g), nodes=NODES, gpus=GPUS, local_experts=LOCAL_EXPERTS
                ),
                n_d,
            )
            for g in range(GPUS)
        ]
        stage_sizes = {g: stagings[n_d * GPUS + g].shape[0] for g in range(GPUS)}
        outs = execute_plans(plans, [pools[n_d * GPUS + g] for g in range(GPUS)], max(stage_sizes.values()) or 1)
        for g in range(GPUS):
            np.testing.assert_array_equal(
                outs[g][: stage_sizes[g]], stagings[n_d * GPUS + g], err_msg=f"staging return ({n_d},{g})"
            )

    # Hop-A transpose: staging chunks return to per-node slices of each source.
    for s in range(DEVICES):
        _in_off, send_sz, out_off, _ = (np.asarray(a) for a in _shard_a2a_params(jnp.asarray(counts_flat), jnp.int32(s)))
        rebuilt = [stagings[d][out_off[d] : out_off[d] + send_sz[d]] for d in range(DEVICES) if send_sz[d]]
        np.testing.assert_array_equal(np.concatenate(rebuilt, axis=0), sends[s], err_msg=f"round trip {s}")
