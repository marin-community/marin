"""Simulate ragged_all_to_all against _expert_granular_a2a_params on CPU.

Checks, for a small multi-shard MoE with forced drops:
  1. the simulated dispatch produces each receiver's buffer expert-major with
     sender order within experts and valid prefixes matching clipped sizes;
  2. an identity "MLP" plus the simulated return a2a reconstructs exactly the
     old path's expand-from-keep-mask result (valid prefix rows at unclipped
     offsets, zeros at dropped rows).
"""

import numpy as np
import jax.numpy as jnp

from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _expert_granular_a2a_params,
    _expert_prefix_keep_mask,
)

rng = np.random.default_rng(0)
S, EL, T, K, H = 4, 3, 10, 2, 5
E = S * EL
ASSIGN = T * K
CAPACITY = int(0.7 * ASSIGN)  # force drops
SPLITS = 2

selected = rng.integers(0, E, size=(S, T, K))
x = rng.normal(size=(S, ASSIGN, H)).astype(np.float32)  # pre-"repeat" rows, one per assignment

sorted_idx = [np.argsort(selected[s].reshape(-1), kind="stable") for s in range(S)]
sorted_x = [x[s][sorted_idx[s]] for s in range(S)]
group_sizes = np.stack([np.bincount(selected[s].reshape(-1), minlength=E) for s in range(S)]).astype(np.int32)

clipped = np.asarray(
    _clip_receiver_group_sizes(jnp.asarray(group_sizes), local_expert_size=EL, receiver_capacity=CAPACITY)
)

params = [
    _expert_granular_a2a_params(
        jnp.asarray(group_sizes),
        jnp.asarray(clipped),
        jnp.asarray(s),
        local_expert_size=EL,
        splits_per_group=SPLITS,
    )
    for s in range(S)
]


def simulate_a2a(operands, outputs, get_params):
    """outputs mutated in place; get_params(s) -> (in_off, send, out_off, recv)."""
    for s in range(S):
        in_off, send, out_off, _ = (np.asarray(a) for a in get_params(s))
        spd = len(in_off) // S
        for u in range(len(in_off)):
            dst = u // spd
            n = send[u]
            outputs[dst][out_off[u] : out_off[u] + n] = operands[s][in_off[u] : in_off[u] + n]


recv = [np.zeros((CAPACITY, H), np.float32) for _ in range(S)]
simulate_a2a(sorted_x, recv, lambda s: params[s][0])

# Reference receiver buffers: expert-major, sender-major within expert, valid prefix per group.
for r in range(S):
    expected_rows = []
    for e in range(EL):
        g = r * EL + e
        for s in range(S):
            start = int(np.cumsum(group_sizes[s])[g] - group_sizes[s][g])
            expected_rows.append(sorted_x[s][start : start + clipped[s, g]])
    expected = np.concatenate(expected_rows, axis=0)
    np.testing.assert_array_equal(recv[r][: len(expected)], expected)
    np.testing.assert_array_equal(recv[r][len(expected) :], 0)
    assert len(expected) == clipped[:, r * EL : (r + 1) * EL].sum() <= CAPACITY

# Identity MLP; simulate the return direction.
ret = [np.zeros((ASSIGN, H), np.float32) for _ in range(S)]
simulate_a2a(recv, ret, lambda s: params[s][1])

# Reference: old path's expand_from_keep_mask semantics.
for s in range(S):
    keep = np.asarray(
        _expert_prefix_keep_mask(jnp.asarray(group_sizes[s]), jnp.asarray(clipped[s]), total_size=ASSIGN)
    )
    expected = np.where(keep[:, None], sorted_x[s], 0)
    np.testing.assert_array_equal(ret[s], expected)

drop_fraction = 1 - clipped.sum() / group_sizes.sum()
print(f"OK: dispatch layout, roundtrip, and drop semantics match (drop fraction {drop_fraction:.3f})")


# --- chunked variant: masked clip per chunk, full sender starts, chained returns ---
CHUNKS = 3  # EL=3 -> 1 expert per chunk, the most aggressive split
CHUNK_EXPERTS = EL // CHUNKS
CHUNK_CAP = int(np.ceil(CAPACITY / CHUNKS))
chunk_of_expert = (np.arange(E) % EL) // CHUNK_EXPERTS

ret2 = [np.zeros((ASSIGN, H), np.float32) for _ in range(S)]
accepted = np.zeros((S, E), np.int32)
for g in range(CHUNKS):
    masked = np.where(chunk_of_expert[None, :] == g, group_sizes, 0)
    clipped_g = np.asarray(
        _clip_receiver_group_sizes(jnp.asarray(masked), local_expert_size=EL, receiver_capacity=CHUNK_CAP)
    )
    accepted += clipped_g
    params_g = [
        _expert_granular_a2a_params(
            jnp.asarray(group_sizes),
            jnp.asarray(clipped_g),
            jnp.asarray(s),
            local_expert_size=EL,
            splits_per_group=SPLITS,
        )
        for s in range(S)
    ]
    recv_g = [np.zeros((CHUNK_CAP, H), np.float32) for _ in range(S)]
    simulate_a2a(sorted_x, recv_g, lambda s: params_g[s][0])
    # chunk receiver packs its experts from offset 0, sender-major within expert
    for r in range(S):
        rows = []
        for e in range(EL):
            gexp = r * EL + e
            if chunk_of_expert[gexp] != g:
                continue
            for s in range(S):
                start = int(np.cumsum(group_sizes[s])[gexp] - group_sizes[s][gexp])
                rows.append(sorted_x[s][start : start + clipped_g[s, gexp]])
        expected = np.concatenate(rows, axis=0) if rows else np.zeros((0, H), np.float32)
        np.testing.assert_array_equal(recv_g[r][: len(expected)], expected)
    simulate_a2a(recv_g, ret2, lambda s: params_g[s][1])

for s in range(S):
    keep = np.asarray(
        _expert_prefix_keep_mask(jnp.asarray(group_sizes[s]), jnp.asarray(accepted[s]), total_size=ASSIGN)
    )
    np.testing.assert_array_equal(ret2[s], np.where(keep[:, None], sorted_x[s], 0))

chunk_drop = 1 - accepted.sum() / group_sizes.sum()
print(f"OK chunked: {CHUNKS} chunks roundtrip matches per-chunk keep semantics (drop fraction {chunk_drop:.3f})")
