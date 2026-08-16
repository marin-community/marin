# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-hop hierarchical transport plans vs a direct NumPy reference.

Emulates the full pipeline on the host: hop A (internode ragged layout,
placed with ``hop_a_params`` offsets), hop B (``hier_dispatch_segments``
executed by the shared plan interpreter), then the reverse path
(``hier_combine_segments`` + hop-A transpose). The pool must match a
from-definition reference and the round trip must restore every kept row.
"""

import jax.numpy as jnp
import numpy as np

from experiments.marin_ep.hier import (
    hier_combine_segments,
    hier_dispatch_segments,
    hop_a_node_counts,
    hop_a_params,
    stage_rows,
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
    """Return staging buffers keyed by (node, gpu), built per the hop-A layout."""
    node_counts = np.asarray(hop_a_node_counts(jnp.asarray(accepted), nodes=NODES, gpus=GPUS))
    stagings = {}
    for n_d in range(NODES):
        for g in range(GPUS):
            chunks = []
            for n_s in range(NODES):
                d = n_s * GPUS + g
                node_experts = range(n_d * GPUS * LOCAL_EXPERTS, (n_d + 1) * GPUS * LOCAL_EXPERTS)
                rows = [_expert_slice(sends, seg_starts, accepted, d, e) for e in node_experts]
                chunks.append(np.concatenate(rows, axis=0) if rows else np.zeros((0, HIDDEN), np.float32))
            stagings[(n_d, g)] = np.concatenate(chunks, axis=0)
    return node_counts, stagings


def test_hop_a_params_match_emulated_layout():
    accepted, _, _, sends, seg_starts = _setup()
    node_counts, stagings = _emulate_hop_a(accepted, sends, seg_starts)
    acc = jnp.asarray(accepted)
    for n_s in range(NODES):
        for g in range(GPUS):
            d = n_s * GPUS + g
            in_off, send_sz, out_off, recv_sz = (
                np.asarray(a) for a in hop_a_params(acc, jnp.int32(n_s), jnp.int32(g), nodes=NODES, gpus=GPUS)
            )
            np.testing.assert_array_equal(send_sz, node_counts[d])
            # My slice for dest node n_d starts where my expert-major buffer
            # reaches n_d's expert block, and lands after lower source nodes.
            for n_d in range(NODES):
                lo = seg_starts[d][n_d * GPUS * LOCAL_EXPERTS]
                assert in_off[n_d] == lo, (n_s, g, n_d)
                got = stagings[(n_d, g)][out_off[n_d] : out_off[n_d] + send_sz[n_d]]
                want = sends[d][in_off[n_d] : in_off[n_d] + send_sz[n_d]]
                np.testing.assert_array_equal(got, want)
            assert int(send_sz.sum()) == int(accepted[d].sum())
            # recv_sizes: rows I receive from each same-rank source node.
            my_stage = stagings[(n_s, g)]
            assert int(recv_sz.sum()) == my_stage.shape[0]
            assert int(np.asarray(stage_rows(acc, jnp.int32(n_s), jnp.int32(g), nodes=NODES, gpus=GPUS))) == (
                stagings[(n_s, g)].shape[0]
            )


def test_two_hop_dispatch_matches_reference_pool_and_round_trips():
    accepted, region, pool_rows, sends, seg_starts = _setup()
    _, stagings = _emulate_hop_a(accepted, sends, seg_starts)
    acc = jnp.asarray(accepted)
    reg = jnp.asarray(region)

    pools = {}
    for n_d in range(NODES):
        plans = [
            hier_dispatch_segments(
                acc, reg, jnp.int32(n_d), jnp.int32(g), nodes=NODES, gpus=GPUS, local_experts=LOCAL_EXPERTS
            )
            for g in range(GPUS)
        ]
        node_stagings = [stagings[(n_d, g)] for g in range(GPUS)]
        outs = execute_plans(plans, node_stagings, pool_rows)
        for g in range(GPUS):
            pools[(n_d, g)] = outs[g]

    # Reference pool: per expert, (source gpu, source node) order.
    for n_d in range(NODES):
        for g_d in range(GPUS):
            owner = n_d * GPUS + g_d
            expect_rows = []
            for e_local in range(LOCAL_EXPERTS):
                e = owner * LOCAL_EXPERTS + e_local
                for g_s in range(GPUS):
                    for n_s in range(NODES):
                        expect_rows.append(_expert_slice(sends, seg_starts, accepted, n_s * GPUS + g_s, e))
            want = np.concatenate(expect_rows, axis=0)
            np.testing.assert_array_equal(pools[(n_d, g_d)][: want.shape[0]], want, err_msg=f"pool ({n_d},{g_d})")

    # Combine: pool -> staging (hop-B transpose), then hop-A transpose -> sources.
    for n_d in range(NODES):
        plans = [
            hier_combine_segments(
                acc, reg, jnp.int32(n_d), jnp.int32(g), nodes=NODES, gpus=GPUS, local_experts=LOCAL_EXPERTS
            )
            for g in range(GPUS)
        ]
        node_pools = [pools[(n_d, g)] for g in range(GPUS)]
        stage_sizes = {g: stagings[(n_d, g)].shape[0] for g in range(GPUS)}
        outs = execute_plans(plans, node_pools, max(stage_sizes.values()) or 1)
        for g in range(GPUS):
            np.testing.assert_array_equal(
                outs[g][: stage_sizes[g]], stagings[(n_d, g)], err_msg=f"staging return ({n_d},{g})"
            )

    # Hop-A transpose: staging chunks return to per-node slices of each source.
    for n_s in range(NODES):
        for g in range(GPUS):
            d = n_s * GPUS + g
            rebuilt = []
            for n_d in range(NODES):
                in_off, send_sz, out_off, recv_sz = (
                    np.asarray(a) for a in hop_a_params(acc, jnp.int32(n_s), jnp.int32(g), nodes=NODES, gpus=GPUS)
                )
                rebuilt.append(stagings[(n_d, g)][out_off[n_d] : out_off[n_d] + send_sz[n_d]])
            np.testing.assert_array_equal(np.concatenate(rebuilt, axis=0), sends[d], err_msg=f"round trip {d}")
