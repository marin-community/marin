# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert-parallel MoE backends against dense and ring references across shards."""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from levanter.grug._moe.common import _TRANSPORT_CAPACITY_MOE_IMPLEMENTATIONS
from levanter.grug.grug_moe import MoeImplementation, moe_mlp
from levanter.testing.devices import use_test_devices


@pytest.mark.parametrize(
    "implementation",
    ["ring", "ring_gather_combine", "ring_filtered", "fixed_all_to_all", "fixed_pooled_wave_all_to_all"],
)
@pytest.mark.parametrize(
    "token_valid",
    [[True, True, True, True], [True, False, True, True]],
    ids=["all_valid", "padded"],
)
@pytest.mark.multidevice
def test_portable_ep_backends_match_dense_cross_shard_value_and_gradients(
    implementation: MoeImplementation,
    token_valid: list[bool],
):
    with use_test_devices(4) as devices, jax.default_matmul_precision("highest"):
        mesh = Mesh(
            np.asarray(devices).reshape(2, 2, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        x = jax.random.normal(jax.random.key(0), (4, 4))
        selected_experts = jnp.asarray(
            [[2, 3], [4, 5], [6, 7], [0, 1]],
            dtype=jnp.int32,
        )
        combine_weights = jax.nn.softmax(jax.random.normal(jax.random.key(1), (4, 2)), axis=-1)
        token_valid = jnp.asarray(token_valid)
        w_up_gate = jax.random.normal(jax.random.key(2), (8, 4, 6))
        w_down = jax.random.normal(jax.random.key(3), (8, 3, 4))
        cotangent = jax.random.normal(jax.random.key(4), (4, 4))

        def dense_output(x, w_up_gate, w_down):
            selected_w13 = w_up_gate[selected_experts]
            hidden = jnp.einsum("th,tkhi->tki", x, selected_w13)
            gate, up = jnp.split(hidden, [3], axis=-1)
            expert_output = jnp.einsum(
                "tki,tkih->tkh",
                jax.nn.silu(gate) * up,
                w_down[selected_experts],
            )
            return jnp.einsum("tkh,tk->th", expert_output, combine_weights * token_valid[:, None])

        expected = jax.jit(dense_output)(x, w_up_gate, w_down)
        expected_gradients = jax.jit(
            jax.grad(
                lambda x, w_up_gate, w_down: jnp.sum(dense_output(x, w_up_gate, w_down) * cotangent),
                argnums=(0, 1, 2),
            )
        )(x, w_up_gate, w_down)

        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        token_sharding = NamedSharding(mesh, P(("data", "expert")))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.device_put(x, batch_sharding)
        selected_experts = jax.device_put(selected_experts, batch_sharding)
        combine_weights = jax.device_put(combine_weights, batch_sharding)
        token_valid = jax.device_put(token_valid, token_sharding)
        w_up_gate = jax.device_put(w_up_gate, expert_sharding)
        w_down = jax.device_put(w_down, expert_sharding)
        cotangent = jax.device_put(cotangent, batch_sharding)

        extra = {}
        if implementation in _TRANSPORT_CAPACITY_MOE_IMPLEMENTATIONS:
            extra["pooled_transport_capacity_factor"] = 4.0

        def backend_output(x, w_up_gate, w_down):
            return moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                token_valid=token_valid,
                activation=jax.nn.silu,
                implementation=implementation,
                mesh=mesh,
                capacity_factor=4.0,
                report_capacity_overflow=True,
                **extra,
            )

        with jax.set_mesh(mesh):
            actual, overflow = jax.jit(backend_output)(x, w_up_gate, w_down)
            actual_gradients = jax.jit(
                jax.grad(
                    lambda x, w_up_gate, w_down: jnp.sum(backend_output(x, w_up_gate, w_down)[0] * cotangent),
                    argnums=(0, 1, 2),
                )
            )(x, w_up_gate, w_down)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual_gradient),
                np.asarray(expected_gradient),
                rtol=1e-5,
                atol=1e-5,
            )
        assert int(overflow.dropped) == 0
        assert int(overflow.padding_skipped) == int(jnp.sum(~token_valid)) * 2


@pytest.mark.parametrize(
    "token_valid",
    [[True] * 16, [True, False, True, True] * 4],
    ids=["all_valid", "padded"],
)
@pytest.mark.multidevice
def test_ring_gather_combine_matches_scatter_combine_with_drops(token_valid: list[bool]):
    """`ring_gather_combine` gives `ring`'s values, drops and gradients without token-buffer scatter-adds."""
    with use_test_devices(4) as devices:
        mesh = Mesh(
            np.asarray(devices).reshape(2, 2, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        tokens, hidden, inter, experts, topk = 16, 10, 6, 8, 3
        keys = jax.random.split(jax.random.key(0), 6)
        x = jax.random.normal(keys[0], (tokens, hidden))
        # Skewed routing so some shards run out of capacity.
        logits = jax.random.normal(keys[1], (tokens, experts)) + jnp.linspace(2.0, 0.0, experts)
        selected_experts = jax.lax.top_k(logits, topk)[1].astype(jnp.int32)
        combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, topk)), axis=-1)
        token_valid = jnp.asarray(token_valid)
        w_up_gate = jax.random.normal(keys[3], (experts, hidden, 2 * inter))
        w_down = jax.random.normal(keys[4], (experts, inter, hidden))
        cotangent = jax.random.normal(keys[5], (tokens, hidden))

        batch = NamedSharding(mesh, P(("data", "expert"), None))
        expert = NamedSharding(mesh, P("expert", None, None))
        x, selected_experts, combine_weights, cotangent = (
            jax.device_put(a, batch) for a in (x, selected_experts, combine_weights, cotangent)
        )
        token_valid = jax.device_put(token_valid, NamedSharding(mesh, P(("data", "expert"))))
        w_up_gate, w_down = jax.device_put(w_up_gate, expert), jax.device_put(w_down, expert)

        def loss(implementation, x, combine_weights, w_up_gate, w_down):
            out, counts = moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                token_valid=token_valid,
                activation=jax.nn.silu,
                implementation=implementation,
                mesh=mesh,
                capacity_factor=0.5,
                report_capacity_overflow=True,
            )
            return jnp.sum(out * cotangent), counts.dropped

        def token_row_scatter_adds(jaxpr):
            # Scatter-adds that write [rows, hidden] token buffers, searched through nested jaxprs.
            count = 0
            for eqn in jaxpr.eqns:
                shape = eqn.outvars[0].aval.shape if eqn.outvars else ()
                count += eqn.primitive.name == "scatter-add" and len(shape) == 2 and shape[1] == hidden
                for param in eqn.params.values():
                    for sub in param if isinstance(param, (tuple, list)) else (param,):
                        sub = getattr(sub, "jaxpr", sub)
                        if hasattr(sub, "eqns"):
                            count += token_row_scatter_adds(sub)
            return count

        results = {}
        for implementation in ("ring", "ring_gather_combine"):
            with jax.set_mesh(mesh):
                grad_fn = jax.value_and_grad(
                    functools.partial(loss, implementation), argnums=(0, 1, 2, 3), has_aux=True
                )
                args = (x, combine_weights, w_up_gate, w_down)
                scatter_adds = token_row_scatter_adds(jax.make_jaxpr(grad_fn)(*args).jaxpr)
                results[implementation] = (grad_fn(*args), scatter_adds)

        ((ring_value, ring_dropped), ring_grads), ring_scatter_adds = results["ring"]
        ((gather_value, gather_dropped), gather_grads), gather_scatter_adds = results["ring_gather_combine"]
        assert ring_scatter_adds > 0, ring_scatter_adds
        assert gather_scatter_adds == 0, gather_scatter_adds
        assert int(ring_dropped) > 0, int(ring_dropped)
        assert int(gather_dropped) == int(ring_dropped)
        np.testing.assert_allclose(np.asarray(gather_value), np.asarray(ring_value), rtol=1e-5, atol=1e-5)
        for gather_grad, ring_grad in zip(gather_grads, ring_grads, strict=True):
            np.testing.assert_allclose(np.asarray(gather_grad), np.asarray(ring_grad), rtol=1e-5, atol=1e-5)
