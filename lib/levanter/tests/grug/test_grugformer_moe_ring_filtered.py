# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ring-filtered expert-parallel MoE against ring and a dense reference, with sender and receiver drops."""

import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from levanter.grug._moe.ep_ring_filtered import destination_hit_fraction
from levanter.grug.grug_moe import moe_mlp
from levanter.testing.devices import use_test_devices


@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    ("transport_capacity_factor", "capacity_factor", "padding"),
    [
        (8.0, 8.0, "none"),
        (0.5, 8.0, "none"),
        (8.0, 0.75, "none"),
        (0.5, 0.3, "uneven"),
        (0.5, 0.3, "all"),
    ],
    ids=["no_drops", "sender_drops", "receiver_drops", "both_drops_uneven_padding", "all_padding"],
)
@pytest.mark.multidevice
def test_ring_filtered_matches_ring_and_dense_with_drops(
    transport_capacity_factor: float, capacity_factor: float, padding: str
):
    """On 4 expert shards, ring_filtered matches a dense reference that applies its sender capacity and then ring's
    receiver capacity, and matches ring whenever no sender drops. Gradients include the combine weights, which
    carry the router gradient. The uneven padding leaves shard 0 with no valid tokens."""
    with use_test_devices(4) as devices, jax.default_matmul_precision("highest"):
        shards, tokens_per_shard, hidden, inter, experts, topk = 4, 48, 8, 6, 32, 4

        tokens = shards * tokens_per_shard
        local_experts = experts // shards
        mesh = Mesh(
            np.asarray(devices).reshape(1, shards, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        keys = jax.random.split(jax.random.key(0), 7)
        x = jax.random.normal(keys[0], (tokens, hidden))
        selected = jnp.argsort(jax.random.uniform(keys[1], (tokens, experts)), axis=1)[:, :topk].astype(jnp.int32)
        combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, topk)), axis=-1)
        w_up_gate = jax.random.normal(keys[3], (experts, hidden, 2 * inter)) / math.sqrt(hidden)
        w_down = jax.random.normal(keys[4], (experts, inter, hidden)) / math.sqrt(inter)
        cotangent = jax.random.normal(keys[5], (tokens, hidden))
        if padding == "uneven":
            # Shard s keeps about s / 3 of its tokens.
            keep_fraction = np.repeat(np.arange(shards) / (shards - 1), tokens_per_shard)
            token_valid = np.asarray(jax.random.uniform(keys[6], (tokens,))) < keep_fraction
        else:
            token_valid = np.full(tokens, padding == "none")

        # Sender capacity: per (source, destination) shard pair, the first C valid tokens that select an expert
        # on the destination keep their assignments there, with C scaled by the source's valid tokens.
        sel_np = np.asarray(selected)
        keep = np.repeat(token_valid[:, None], topk, axis=1)
        hit_fraction = destination_hit_fraction(num_experts=experts, local_experts=local_experts, topk=topk)
        pair_capacity = min(tokens_per_shard, math.ceil(transport_capacity_factor * hit_fraction * tokens_per_shard))
        for source in range(shards):
            first = source * tokens_per_shard
            source_valid = int(token_valid[first : first + tokens_per_shard].sum())
            capacity = min(pair_capacity, max(1, math.ceil(transport_capacity_factor * hit_fraction * source_valid)))
            for dest in range(shards):
                seen = 0
                for t in range(first, first + tokens_per_shard):
                    on_dest = sel_np[t] // local_experts == dest
                    if token_valid[t] and on_dest.any():
                        if seen >= capacity:
                            keep[t, on_dest] = False
                        seen += 1
        valid_assignments = int(token_valid.sum()) * topk
        expected_sender_dropped = valid_assignments - int(keep.sum())

        # Ring's receiver capacity: each destination runs at most L of the assignments it receives, in (expert,
        # token) order, with L set by the valid demand before sender drops.
        receiver_capacity = min(
            max(local_experts, math.ceil(capacity_factor * valid_assignments / shards)),
            max(local_experts, math.ceil(capacity_factor * tokens_per_shard * topk)),
        )
        expected_receiver_dropped = 0
        for dest in range(shards):
            accepted = 0
            for expert in range(dest * local_experts, (dest + 1) * local_experts):
                for t, k in zip(*np.nonzero(keep & (sel_np == expert))):
                    if accepted < receiver_capacity:
                        accepted += 1
                    else:
                        keep[t, k] = False
                        expected_receiver_dropped += 1
        keep = jnp.asarray(keep, jnp.float32)

        def dense(x, cw, w_up_gate, w_down):
            hidden_act = jnp.einsum("th,tkhi->tki", x, w_up_gate[selected])
            gate, up = jnp.split(hidden_act, [inter], axis=-1)
            out = jnp.einsum("tki,tkih->tkh", jax.nn.silu(gate) * up, w_down[selected])
            return jnp.einsum("tkh,tk->th", out, cw * keep)

        batch = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        args = (
            jax.device_put(x, batch),
            jax.device_put(combine_weights, batch),
            jax.device_put(w_up_gate, expert_sharding),
            jax.device_put(w_down, expert_sharding),
        )
        selected_sharded = jax.device_put(selected, batch)
        token_valid_sharded = jax.device_put(jnp.asarray(token_valid), NamedSharding(mesh, P(("data", "expert"))))
        cotangent_sharded = jax.device_put(cotangent, batch)

        def backend(implementation):
            def loss(x, cw, w_up_gate, w_down):
                out, counts = moe_mlp(
                    x,
                    selected_sharded,
                    cw,
                    w_up_gate,
                    w_down,
                    token_valid=token_valid_sharded,
                    activation=jax.nn.silu,
                    implementation=implementation,
                    mesh=mesh,
                    capacity_factor=capacity_factor,
                    pooled_transport_capacity_factor=transport_capacity_factor,
                    report_capacity_overflow=True,
                )
                return jnp.sum(out * cotangent_sharded), (out, counts)

            with jax.set_mesh(mesh):
                grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
                (_, (out, counts)), grads = grad_fn(*args)
            return out, counts, grads

        expected = dense(x, combine_weights, w_up_gate, w_down)
        expected_grads = jax.grad(lambda *a: jnp.sum(dense(*a) * cotangent), argnums=(0, 1, 2, 3))(
            x, combine_weights, w_up_gate, w_down
        )
        actual, counts, actual_grads = backend("ring_filtered")
        references = [(expected, expected_grads)]
        if expected_sender_dropped == 0:
            ring_out, ring_counts, ring_grads = backend("ring")
            assert int(ring_counts.dropped) == expected_receiver_dropped
            references.append((ring_out, ring_grads))

        for ref_out, ref_grads in references:
            np.testing.assert_allclose(np.asarray(actual), np.asarray(ref_out), rtol=1e-5, atol=1e-5)
            for a, e in zip(actual_grads, ref_grads, strict=True):
                np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=1e-5, atol=1e-5)
        assert int(counts.sender_dropped) == expected_sender_dropped
        assert int(counts.receiver_dropped) == expected_receiver_dropped
        # Each case drops what its parameters ask for; with no valid tokens nothing can drop.
        assert (expected_sender_dropped > 0) == (padding != "all" and transport_capacity_factor < 1.0)
        assert (expected_receiver_dropped > 0) == (padding != "all" and capacity_factor < 1.0)
