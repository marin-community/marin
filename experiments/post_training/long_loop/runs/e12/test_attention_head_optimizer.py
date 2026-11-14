# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.post_training.long_loop.runs.e12.attention_head_optimizer import (
    E12AttentionHeadConfig,
    attention_head_mask,
    trainable_path,
)


@pytest.fixture(autouse=True)
def cpu_mesh():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    with jax.set_mesh(mesh):
        yield


def toy_params():
    matrix = jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4) / 10
    return {
        "output_proj": matrix,
        "token_embed": matrix,
        "stacked_blocks": {
            "stacked": {
                "attn": {**{k: matrix[None] for k in ("w_q", "w_k", "w_v", "w_o")}, "attn_gate": matrix},
                "mlp": {"router": matrix, "router_bias": matrix, "expert_mlp": {"w_up": matrix[None, None]}},
                "rms_attn": {"weight": matrix},
            }
        },
    }


def test_exact_projection_names():
    assert trainable_path("output_proj")
    assert trainable_path("stacked_blocks.stacked.attn.w_q")
    for path in (
        "token_embed",
        "stacked_blocks.stacked.attn.attn_gate",
        "stacked_blocks.stacked.rms_attn.weight",
        "stacked_blocks.stacked.mlp.router",
        "stacked_blocks.stacked.mlp.expert_mlp.w_up",
        "fake_output_proj",
        "stacked_blocks.stacked.attn.w_q_extra",
    ):
        assert not trainable_path(path)


@pytest.mark.parametrize("jit", [False, True])
def test_frozen_leaves_stay_exact_across_nonzero_steps(jit):
    params = toy_params()
    initial = jax.tree.map(np.asarray, params)
    mask = attention_head_mask(params)
    optimizer = E12AttentionHeadConfig(learning_rate=2e-5, adam_lr=2e-5, warmup=0.0).build(25)
    state = optimizer.init(params)

    def step(params, state):
        grads = jax.tree.map(lambda x: jnp.sin(x * 3) + 0.3, params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, updates

    run = jax.jit(step) if jit else step
    for _ in range(5):
        params, state, updates = run(params, state)
        for before, after, update, active in zip(
            jax.tree.leaves(initial),
            jax.tree.leaves(params),
            jax.tree.leaves(updates),
            jax.tree.leaves(mask),
            strict=True,
        ):
            if not active:
                np.testing.assert_array_equal(before, after)
                np.testing.assert_array_equal(np.zeros_like(before), update)
    assert all(
        not np.array_equal(a, b)
        for a, b, active in zip(jax.tree.leaves(initial), jax.tree.leaves(params), jax.tree.leaves(mask), strict=True)
        if active
    )
    # Frozen expert bank is rank four. No optimizer array may have its shape.
    assert not any(getattr(x, "ndim", 0) == 4 for x in jax.tree.leaves(state))


def test_masked_alone_would_leak_gradients():
    params = toy_params()
    mask = attention_head_mask(params)
    optimizer = optax.masked(optax.adamw(1e-3, weight_decay=0.1), mask)
    updates, _ = optimizer.update(jax.tree.map(jnp.ones_like, params), optimizer.init(params), params)
    assert np.any(np.asarray(updates["token_embed"]) != 0)
    # The explicit complementary zero transform blocks passthrough, including decay.
    safe = optax.chain(optimizer, optax.masked(optax.set_to_zero(), jax.tree.map(lambda b: not b, mask)))
    state = safe.init(params)
    before = np.asarray(params["token_embed"])
    for _ in range(5):
        updates, state = safe.update(jax.tree.map(jnp.ones_like, params), state, params)
        params = optax.apply_updates(params, updates)
        np.testing.assert_array_equal(before, np.asarray(params["token_embed"]))
