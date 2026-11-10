# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The Pallas-Triton routing-metadata kernels of ``ring_filtered`` against a sequential reference and the XLA path.

Off GPU the kernels run in the Pallas interpreter; on TPU the option is rejected, so these tests skip there.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from levanter.grug._moe.pallas_routing_metadata import expert_rows, send_slots
from levanter.grug.grug_moe import moe_mlp
from levanter.testing.devices import use_test_devices

pytestmark = pytest.mark.skipif(jax.default_backend() == "tpu", reason="Pallas-Triton kernels need a GPU or CPU")


def _bits(x: jax.Array) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def _send_reference(selected, valid, cap, *, local_experts, destinations, pair_capacity):
    """One pass over the tokens per destination, in token order."""
    tokens, topk = selected.shape
    send_rows = destinations * pair_capacity
    dest = selected // local_experts
    slot = np.full((tokens, destinations), send_rows, np.int32)
    source = np.full((send_rows,), tokens, np.int32)
    for d in range(destinations):
        rank = 0
        for t in range(tokens):
            if valid[t] and (dest[t] == d).any():
                if rank < cap:
                    slot[t, d] = d * pair_capacity + rank
                    source[d * pair_capacity + rank] = t
                rank += 1
    kept = int(np.sum((np.take_along_axis(slot, dest, axis=1) != send_rows) & valid[:, None]))
    return slot, source, kept


def _expert_reference(ids, cap, *, local_experts, physical_capacity):
    """Stable sort of the assignments by local expert; capacity filled in expert order."""
    assignments = ids.shape[0]
    counts = np.bincount(ids[ids >= 0], minlength=local_experts)
    starts = np.cumsum(counts) - counts
    accepted = np.minimum(counts, np.maximum(cap - starts, 0))
    picked = np.full((physical_capacity,), assignments, np.int32)
    position = np.full((assignments,), physical_capacity, np.int32)
    seen = np.zeros(local_experts, np.int64)
    for a in range(assignments):
        e = ids[a]
        if e < 0:
            continue
        if seen[e] < accepted[e]:
            row = starts[e] + seen[e]
            picked[row] = a
            position[a] = row
        seen[e] += 1
    # Accepted rows are contiguous from row 0: a capped expert leaves no gap because capacity is a prefix.
    assert np.all(picked[: accepted.sum()] < assignments)
    return accepted.astype(np.int32), int(counts.sum() - accepted.sum()), picked, position


def _routing(key, tokens, *, experts, topk):
    k_sel, k_valid = jax.random.split(key)
    # Skewed expert choice so some destinations and experts overflow their capacity.
    logits = jax.random.gumbel(k_sel, (tokens, experts)) + jnp.linspace(1.5, 0.0, experts)
    selected = np.asarray(jax.lax.top_k(logits, topk)[1], np.int32)
    valid = np.asarray(jax.random.bernoulli(k_valid, 0.9, (tokens,)))
    return selected, valid


@pytest.mark.parametrize(
    ("tokens", "cap", "pair_capacity"),
    [(3000, 400, 500), (700, 700, 700), (64, 3, 8)],
    ids=["three_blocks_sender_drops", "one_block_no_drops", "tiny_cap"],
)
def test_send_slots_match_reference(tokens, cap, pair_capacity):
    local_experts, destinations, topk = 4, 8, 4
    selected, valid = _routing(jax.random.key(tokens), tokens, experts=local_experts * destinations, topk=topk)
    maps = jax.jit(
        lambda s, v, c: send_slots(
            s,
            v,
            c,
            local_experts=local_experts,
            destinations=destinations,
            pair_capacity=pair_capacity,
        )
    )(selected, valid, jnp.int32(cap))
    slot, source, kept = _send_reference(
        selected, valid, cap, local_experts=local_experts, destinations=destinations, pair_capacity=pair_capacity
    )
    np.testing.assert_array_equal(np.asarray(maps.slot), slot)
    np.testing.assert_array_equal(np.asarray(maps.source), source)
    assert int(maps.kept_assignments) == kept


@pytest.mark.parametrize(
    ("pair_capacity", "cap", "physical_capacity"),
    [(150, 900, 2400), (150, 5000, 4800), (40, 64, 64)],
    ids=["five_blocks_receiver_drops", "no_drops_all_rows", "one_block_tight"],
)
def test_expert_rows_match_reference(pair_capacity, cap, physical_capacity):
    destinations, topk, local_experts = 8, 4, 32
    rng = np.random.default_rng(pair_capacity + cap)
    rows = destinations * pair_capacity
    # Some send rows are empty (-1 ids), and experts are skewed so the low ones fill the capacity first.
    ids = np.minimum(rng.geometric(0.08, (rows, topk)) - 1, local_experts - 1).astype(np.int32)
    ids[rng.random((rows, topk)) < 0.15] = -1
    weights = rng.random((rows, topk)).astype(np.float32)
    meta = np.concatenate([weights, ids.astype(np.float32)], axis=1).reshape(destinations, pair_capacity, 2 * topk)
    meta = np.concatenate([meta, np.zeros((destinations, 1, 2 * topk), np.float32)], axis=1)
    out = jax.jit(lambda m, c: expert_rows(m, c, local_experts=local_experts, physical_capacity=physical_capacity))(
        meta, jnp.int32(cap)
    )
    group_sizes, dropped, picked, position = _expert_reference(
        ids.reshape(-1), min(cap, physical_capacity), local_experts=local_experts, physical_capacity=physical_capacity
    )
    np.testing.assert_array_equal(np.asarray(out.group_sizes), group_sizes)
    assert int(out.dropped) == dropped
    np.testing.assert_array_equal(np.asarray(out.picked), picked)
    np.testing.assert_array_equal(np.asarray(out.position), position)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.multidevice
def test_ring_filtered_pallas_routing_metadata_matches_xla_bitwise(dtype):
    """On 4 expert shards with sender and receiver drops and invalid tokens, ring_filtered with the Pallas routing maps
    gives the outputs, gradients and drop counts of the XLA maps, bit for bit."""
    with use_test_devices(4) as devices:
        shards, tokens_per_shard, hidden, inter, experts, topk = 4, 64, 32, 16, 64, 4
        tokens = shards * tokens_per_shard
        mesh = Mesh(
            np.asarray(devices).reshape(1, shards, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        keys = jax.random.split(jax.random.key(0), 7)
        x = jax.random.normal(keys[0], (tokens, hidden)).astype(dtype)
        logits = jax.random.gumbel(keys[1], (tokens, experts)) + jnp.linspace(1.0, 0.0, experts)
        selected = jax.lax.top_k(logits, topk)[1].astype(jnp.int32)
        combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, topk)), axis=-1)
        w_up_gate = (jax.random.normal(keys[3], (experts, hidden, 2 * inter)) / math.sqrt(hidden)).astype(dtype)
        w_down = (jax.random.normal(keys[4], (experts, inter, hidden)) / math.sqrt(inter)).astype(dtype)
        cotangent = jax.random.normal(keys[5], (tokens, hidden)).astype(dtype)
        keep_fraction = np.repeat(0.3 + 0.7 * np.arange(shards) / (shards - 1), tokens_per_shard)
        token_valid = np.asarray(jax.random.uniform(keys[6], (tokens,))) < keep_fraction

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

        def run(routing_metadata):
            def loss(x, cw, w_up_gate, w_down):
                out, counts = moe_mlp(
                    x,
                    selected_sharded,
                    cw,
                    w_up_gate,
                    w_down,
                    token_valid=token_valid_sharded,
                    activation=jax.nn.silu,
                    implementation="ring_filtered",
                    mesh=mesh,
                    capacity_factor=0.4,
                    pooled_transport_capacity_factor=0.6,
                    report_capacity_overflow=True,
                    routing_metadata=routing_metadata,
                )
                return jnp.sum((out * cotangent_sharded).astype(jnp.float32)), (out, counts)

            with jax.set_mesh(mesh):
                grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
                jaxpr = str(jax.make_jaxpr(grad_fn)(*args))
                (_, (out, counts)), grads = grad_fn(*args)
            # The option must select the kernels, or the comparison below would hold trivially.
            assert ("pallas_call" in jaxpr) == (routing_metadata == "pallas")
            return out, counts, grads

        xla_out, xla_counts, xla_grads = run("xla")
        out, counts, grads = run("pallas")
        assert int(xla_counts.sender_dropped) > 0 and int(xla_counts.receiver_dropped) > 0
        assert int(counts.sender_dropped) == int(xla_counts.sender_dropped)
        assert int(counts.receiver_dropped) == int(xla_counts.receiver_dropped)
        assert int(counts.padding_skipped) == int(xla_counts.padding_skipped)
        np.testing.assert_array_equal(_bits(out), _bits(xla_out))
        for a, e in zip(grads, xla_grads, strict=True):
            assert np.any(np.asarray(e, np.float32) != 0)
            np.testing.assert_array_equal(_bits(a), _bits(e))


def test_routing_metadata_kernels_match_reference_at_june_shapes_on_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("compiles the Triton kernels; needs a GPU")
    # One June layer-microbatch: 32768 tokens per GPU, top-4 of 256 experts over 8 shards, C = 14988 send rows per
    # destination, 131072 expert rows.
    local_experts, destinations, topk, pair_capacity = 32, 8, 4, 14988
    selected, valid = _routing(jax.random.key(3), 32768, experts=local_experts * destinations, topk=topk)
    maps = jax.jit(
        lambda s, v: send_slots(
            s,
            v,
            jnp.int32(13000),
            local_experts=local_experts,
            destinations=destinations,
            pair_capacity=pair_capacity,
        )
    )(selected, valid)
    slot, source, kept = _send_reference(
        selected, valid, 13000, local_experts=local_experts, destinations=destinations, pair_capacity=pair_capacity
    )
    np.testing.assert_array_equal(np.asarray(maps.slot), slot)
    np.testing.assert_array_equal(np.asarray(maps.source), source)
    assert int(maps.kept_assignments) == kept

    rng = np.random.default_rng(4)
    rows = destinations * pair_capacity
    ids = np.minimum(rng.geometric(0.05, (rows, topk)) - 1, local_experts - 1).astype(np.int32)
    ids[rng.random((rows, topk)) < 0.1] = -1
    meta = np.concatenate([rng.random((rows, topk)).astype(np.float32), ids.astype(np.float32)], axis=1)
    meta = np.concatenate(
        [meta.reshape(destinations, pair_capacity, 2 * topk), np.zeros((destinations, 1, 2 * topk), np.float32)], 1
    )
    out = jax.jit(lambda m: expert_rows(m, jnp.int32(120000), local_experts=local_experts, physical_capacity=131072))(
        meta
    )
    group_sizes, dropped, picked, position = _expert_reference(
        ids.reshape(-1), 120000, local_experts=local_experts, physical_capacity=131072
    )
    np.testing.assert_array_equal(np.asarray(out.group_sizes), group_sizes)
    assert int(out.dropped) == dropped
    np.testing.assert_array_equal(np.asarray(out.picked), picked)
    np.testing.assert_array_equal(np.asarray(out.position), position)
