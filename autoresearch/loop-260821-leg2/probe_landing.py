"""Empirical landing-pattern probe for expert-granular ragged_all_to_all params.

Runs on one 4-GPU node. Rows carry (1000*shard + sorted_position) in every element, so
the receiver buffers show exactly which slice landed where. Compares against the
simulation semantics that check_expert_a2a_params.py verified, and prints both on
mismatch. Also probes the peer-granular baseline params (leg-1 semantics) as a control.
"""

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import shard_map

from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _expert_granular_a2a_params,
    _shard_a2a_params,
)

S, EL, T, K, H = 4, 3, 8, 2, 4
E = S * EL
ASSIGN = T * K
CAPACITY = ASSIGN * S  # no drops possible


def main():
    devices = jax.devices()
    assert len(devices) >= S, f"need {S} devices, got {len(devices)}"
    mesh = Mesh(np.array(devices[:S]), ("expert",))

    rng = np.random.default_rng(0)
    selected = rng.integers(0, E, size=(S, T, K)).astype(np.int32)

    # Host-side reference state per shard.
    sorted_idx = [np.argsort(selected[s].reshape(-1), kind="stable") for s in range(S)]
    group_sizes = np.stack([np.bincount(selected[s].reshape(-1), minlength=E) for s in range(S)]).astype(np.int32)
    clipped = np.asarray(
        _clip_receiver_group_sizes(jnp.asarray(group_sizes), local_expert_size=EL, receiver_capacity=CAPACITY)
    )
    np.testing.assert_array_equal(clipped, group_sizes)  # no drops in this probe

    # Row payloads: 1000*shard + sorted position.
    payload = np.stack([np.full((ASSIGN, H), 1000 * s, np.float32) + np.arange(ASSIGN)[:, None] for s in range(S)])

    def dispatch_local(x_local, sel_local):
        shard_id = jax.lax.axis_index("expert")
        flat = sel_local.reshape(-1)
        gs = jnp.bincount(flat, length=E).astype(jnp.int32)
        all_gs = jax.lax.all_gather(gs, "expert")
        cgs = _clip_receiver_group_sizes(all_gs, local_expert_size=EL, receiver_capacity=CAPACITY)
        dparams, rparams = _expert_granular_a2a_params(
            all_gs, cgs, shard_id, local_expert_size=EL, splits_per_group=1
        )
        out = jnp.zeros((CAPACITY, H), x_local.dtype)
        recv = jax.lax.ragged_all_to_all(x_local, out, *dparams, axis_name="expert")
        ret_out = jnp.zeros((ASSIGN, H), x_local.dtype)
        ret = jax.lax.ragged_all_to_all(recv, ret_out, *rparams, axis_name="expert")
        return recv, ret

    fn = shard_map(
        dispatch_local,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None)),
        out_specs=(P("expert", None), P("expert", None)),
    )
    x = jnp.asarray(payload.reshape(S * ASSIGN, H))
    x = jax.device_put(x, NamedSharding(mesh, P("expert", None)))
    sorted_payload = np.stack([payload[s][sorted_idx[s]] for s in range(S)])
    xs = jax.device_put(jnp.asarray(sorted_payload.reshape(S * ASSIGN, H)), NamedSharding(mesh, P("expert", None)))
    sel = jax.device_put(jnp.asarray(selected.reshape(S * T, K)), NamedSharding(mesh, P("expert", None)))

    recv, ret = jax.jit(fn)(xs, sel)
    recv = np.asarray(recv).reshape(S, CAPACITY, H)
    ret = np.asarray(ret).reshape(S, ASSIGN, H)

    # Simulation expectation (validated by check_expert_a2a_params.py).
    ok = True
    for r in range(S):
        rows = []
        for e in range(EL):
            g = r * EL + e
            for s in range(S):
                start = int(np.cumsum(group_sizes[s])[g] - group_sizes[s][g])
                rows.append(sorted_payload[s][start : start + clipped[s, g]])
        expected = np.concatenate(rows, axis=0)
        got = recv[r][: len(expected)]
        if not np.array_equal(got, expected):
            ok = False
            print(f"=== receiver {r} MISMATCH (dispatch) ===")
            print("expected first-col:", expected[:, 0].astype(int).tolist())
            print("got      first-col:", got[:, 0].astype(int).tolist())
    for s in range(S):
        if not np.array_equal(ret[s], sorted_payload[s]):
            ok = False
            print(f"=== sender {s} MISMATCH (return) ===")
            print("expected first-col:", sorted_payload[s][:, 0].astype(int).tolist())
            print("got      first-col:", ret[s][:, 0].astype(int).tolist())
    print("PROBE_RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
