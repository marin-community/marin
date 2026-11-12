# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Pallas row gathers with the XLA reference."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from levanter.grug._moe.ep_ring import _combine_rows, _dispatch_rows, _gather_sum_slots, _take_valid_rows
from levanter.grug._moe.pallas_gather_sum import gather_sum_rows, gather_valid_rows
from levanter.grug.grug_moe import moe_mlp
from levanter.testing.devices import use_test_devices

pytestmark = pytest.mark.skipif(jax.default_backend() == "tpu", reason="Pallas-Triton kernels need a GPU or CPU")


def _bits(x: jax.Array) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def _rows(key, shape, dtype):
    rows = jax.random.normal(key, shape, jnp.float32)
    # Include signed zeros to check the gather-sum's accumulation order.
    rows = rows.at[::7].set(-0.0).at[1::11].set(0.0)
    return rows.astype(dtype)


def _slots(key, num_out, topk, num_src):
    # Each source row read at most once, as in the dispatch backward, with sentinels at and past num_src.
    perm = jax.random.permutation(key, num_out * topk)
    slots = jnp.where(perm < num_src, perm, num_src + perm % 3).reshape(num_out, topk)
    return slots.astype(jnp.int32)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize(
    ("num_src", "num_out", "topk", "hidden"),
    [(96, 70, 4, 64), (40, 33, 1, 48), (300, 129, 3, 1024)],
    ids=["k4_tail_rows", "k1_narrow", "k3_two_col_blocks"],
)
def test_gather_sum_rows_matches_xla_bitwise(num_src, num_out, topk, hidden, dtype):
    k_rows, k_slots = jax.random.split(jax.random.key(0))
    rows = _rows(k_rows, (num_src, hidden), dtype)
    slots = _slots(k_slots, num_out, topk, num_src)
    expected = jax.jit(_gather_sum_slots)(rows, slots)
    actual = jax.jit(gather_sum_rows)(rows, slots)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_gather_valid_rows_matches_xla_bitwise(dtype):
    k_x, k_idx, k_valid = jax.random.split(jax.random.key(1), 3)
    x = _rows(k_x, (50, 80), dtype)
    index = jax.random.randint(k_idx, (77,), 0, 50)
    valid = jax.random.bernoulli(k_valid, 0.7, (77,))
    expected = jax.jit(_take_valid_rows)(x, index, valid)
    actual = jax.jit(gather_valid_rows)(x, index, valid)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def test_pallas_backward_falls_back_to_xla_for_untileable_widths():
    # 24 columns have no power-of-two divisor >= 16, so both backward gathers use XLA under gather_sum="pallas".
    k_x, k_idx, k_valid, k_slots = jax.random.split(jax.random.key(3), 4)
    x = _rows(k_x, (20, 24), jnp.float32)
    slot_token = jax.random.randint(k_idx, (30,), 0, 20)
    slot_valid = jax.random.bernoulli(k_valid, 0.7, (30,))
    slots = _slots(k_slots, 20, 2, 30)

    def grads(backward):
        def loss(x, rows):
            dispatched = _dispatch_rows(x, slot_token, slot_valid, slots, backward)
            combined = _combine_rows(rows, slot_token, slot_valid, slots, backward)
            return jnp.sum(jnp.sin(dispatched)) + jnp.sum(jnp.cos(combined))

        fn = jax.jit(jax.grad(loss, argnums=(0, 1)))
        return fn(x, x[slot_token]), str(jax.make_jaxpr(fn)(x, x[slot_token]))

    expected, _ = grads("xla")
    actual, jaxpr = grads("pallas")
    assert "pallas_call" not in jaxpr
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(_bits(a), _bits(e))


@pytest.mark.timeout(600)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.multidevice
def test_ring_filtered_pallas_gather_sum_matches_xla_bitwise(dtype):
    with use_test_devices(4) as devices:
        shards, tokens_per_shard, hidden, inter, experts, topk = 4, 32, 32, 16, 32, 4
        tokens = shards * tokens_per_shard
        mesh = Mesh(
            np.asarray(devices).reshape(1, shards, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        keys = jax.random.split(jax.random.key(0), 7)
        x = jax.random.normal(keys[0], (tokens, hidden)).astype(dtype)
        selected = jnp.argsort(jax.random.uniform(keys[1], (tokens, experts)), axis=1)[:, :topk].astype(jnp.int32)
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

        def run(gather_sum):
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
                    capacity_factor=0.3,
                    pooled_transport_capacity_factor=0.5,
                    report_capacity_overflow=True,
                    gather_sum=gather_sum,
                )
                return jnp.sum((out * cotangent_sharded).astype(jnp.float32)), (out, counts)

            with jax.set_mesh(mesh):
                grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
                jaxpr = str(jax.make_jaxpr(grad_fn)(*args))
                (_, (out, counts)), grads = grad_fn(*args)
            # Only the Pallas path has kernels, so the bitwise comparison below cannot pass trivially.
            assert ("pallas_call" in jaxpr) == (gather_sum == "pallas")
            return out, counts, grads

        xla_out, xla_counts, xla_grads = run("xla")
        out, counts, grads = run("pallas")
        assert int(xla_counts.sender_dropped) > 0 and int(xla_counts.receiver_dropped) > 0
        assert int(counts.sender_dropped) == int(xla_counts.sender_dropped)
        assert int(counts.receiver_dropped) == int(xla_counts.receiver_dropped)
        np.testing.assert_array_equal(_bits(out), _bits(xla_out))
        for a, e in zip(grads, xla_grads, strict=True):
            assert np.any(np.asarray(e, np.float32) != 0)
            np.testing.assert_array_equal(_bits(a), _bits(e))


@pytest.mark.parametrize("topk", [1, 4])
def test_pallas_gathers_match_xla_bitwise_at_june_shapes_on_gpu(topk):
    if jax.default_backend() != "gpu":
        pytest.skip("compiles the Triton kernel; needs a GPU")
    # The dispatch backward's shape at 1/8 scale: 14988 received rows from 16384 expert rows, hidden 2560.
    k_rows, k_slots, k_idx, k_valid = jax.random.split(jax.random.key(2), 4)
    rows = _rows(k_rows, (16384, 2560), jnp.bfloat16)
    slots = _slots(k_slots, 14988, topk, 16384)
    np.testing.assert_array_equal(
        _bits(jax.jit(gather_sum_rows)(rows, slots)), _bits(jax.jit(_gather_sum_slots)(rows, slots))
    )
    index = jax.random.randint(k_idx, (14988,), 0, 16384)
    valid = jax.random.bernoulli(k_valid, 0.8, (14988,))
    np.testing.assert_array_equal(
        _bits(jax.jit(gather_valid_rows)(rows, index, valid)), _bits(jax.jit(_take_valid_rows)(rows, index, valid))
    )
