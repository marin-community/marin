# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the grouped fixed-capacity all-to-all MoE backend.

The contract is narrow and checkable: with capacity large enough that neither backend drops,
``fixed_grouped_a2a`` must reproduce ``fixed_all_to_all`` exactly, forward and backward. Below that
it must drop strictly less for the same buffer, which is the whole reason it exists.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, set_mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from levanter.grug._moe.ep_fixed_all_to_all import _moe_mlp_ep_fixed_a2a_local
from levanter.grug._moe.ep_fixed_grouped_all_to_all import _moe_mlp_ep_fixed_grouped_a2a_local

EP = 4
LOCAL_EXPERTS = 3
NUM_EXPERTS = EP * LOCAL_EXPERTS
TOKENS = 64
TOPK = 2
HIDDEN = 8
INTERMEDIATE = 16


def _mesh():
    return Mesh(
        np.asarray(jax.devices()[:EP]).reshape(1, 1, EP, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _inputs(seed, skew):
    """Routing with a deliberate hot/cold split, so grouping has imbalance to absorb."""
    key = jax.random.PRNGKey(seed)
    k_x, k_r, k_w13, k_w2 = jax.random.split(key, 4)
    x = jax.random.normal(k_x, (TOKENS, HIDDEN), jnp.float32)
    logits = jax.random.normal(k_r, (TOKENS, NUM_EXPERTS), jnp.float32)
    # Bias the first expert of every device so its cell overflows while its neighbours idle.
    logits = logits.at[:, ::LOCAL_EXPERTS].add(skew)
    experts = jnp.argsort(-logits, axis=-1)[:, :TOPK].astype(jnp.int32)
    weights = jnp.ones((TOKENS, TOPK), jnp.float32) / TOPK
    w13 = jax.random.normal(k_w13, (NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE), jnp.float32) * 0.1
    w2 = jax.random.normal(k_w2, (NUM_EXPERTS, INTERMEDIATE, HIDDEN), jnp.float32) * 0.1
    return x, experts, weights, w13, w2


def _run(fn, capacity_factor, inputs):
    x, experts, weights, w13, w2 = inputs
    mesh = _mesh()

    def local(x_l, e_l, cw_l, w13_l, w2_l):
        return fn(
            x_l,
            e_l,
            cw_l,
            w13_l,
            w2_l,
            activation_fn=jax.nn.silu,
            num_experts=NUM_EXPERTS,
            capacity_factor=capacity_factor,
        )

    sharded = shard_map(
        local,
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None), P("expert", None, None), P("expert", None, None)),
        out_specs=(P("expert", None), P()),
        check_rep=False,
    )

    row = NamedSharding(mesh, P("expert", None))
    bank = NamedSharding(mesh, P("expert", None, None))

    with set_mesh(mesh):
        x = jax.device_put(x, row)
        experts_s = jax.device_put(experts, row)
        weights_s = jax.device_put(weights, row)
        w13 = jax.device_put(w13, bank)
        w2 = jax.device_put(w2, bank)

        def loss_and_out(x_, w13_, w2_):
            out, dropped = sharded(x_, experts_s, weights_s, w13_, w2_)
            return jnp.sum(out**2), (out, dropped)

        (loss, (out, dropped)), grads = jax.value_and_grad(loss_and_out, argnums=(0, 1, 2), has_aux=True)(x, w13, w2)
    return out, int(dropped), grads


@pytest.mark.skipif(len(jax.devices()) < EP, reason=f"needs {EP} devices")
def test_grouped_matches_fixed_when_nothing_drops():
    # Capacity per cell must exceed what a single expert can attract, or the two backends make
    # different routing decisions and the comparison stops being about numerics. num_experts x the
    # mean is the worst case: every assignment on a shard landing on one expert.
    inputs = _inputs(seed=0, skew=3.0)
    drop_free = float(NUM_EXPERTS)
    out_fixed, drop_fixed, grad_fixed = _run(_moe_mlp_ep_fixed_a2a_local, drop_free, inputs)
    out_grouped, drop_grouped, grad_grouped = _run(_moe_mlp_ep_fixed_grouped_a2a_local, drop_free, inputs)

    assert drop_fixed == 0 and drop_grouped == 0
    np.testing.assert_allclose(out_grouped, out_fixed, rtol=1e-5, atol=1e-5)
    for g_new, g_ref in zip(grad_grouped, grad_fixed):
        np.testing.assert_allclose(g_new, g_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(len(jax.devices()) < EP, reason=f"needs {EP} devices")
def test_grouped_drops_less_than_fixed_at_the_same_buffer():
    # Same capacity factor means the same bytes in both backends; pooling is what buys the
    # difference. A skewed router is the case grouping is meant to absorb.
    inputs = _inputs(seed=1, skew=4.0)
    _, drop_fixed, _ = _run(_moe_mlp_ep_fixed_a2a_local, 1.0, inputs)
    _, drop_grouped, _ = _run(_moe_mlp_ep_fixed_grouped_a2a_local, 1.0, inputs)

    assert drop_grouped < drop_fixed, f"grouped {drop_grouped} should drop less than fixed {drop_fixed}"


@pytest.mark.skipif(len(jax.devices()) < EP, reason=f"needs {EP} devices")
def test_grouped_conserves_tokens_it_keeps():
    # A kept assignment must reach exactly one expert: dropping the router skew entirely should
    # leave nothing dropped and every output row non-zero.
    inputs = _inputs(seed=2, skew=0.0)
    out, dropped, _ = _run(_moe_mlp_ep_fixed_grouped_a2a_local, float(NUM_EXPERTS), inputs)
    assert dropped == 0
    assert jnp.all(jnp.any(out != 0, axis=-1))
