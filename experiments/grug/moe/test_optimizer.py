# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig

_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN, _VOCAB = 2, 8, 16, 32
_UINT_DTYPE = {"bfloat16": jnp.uint16, "float32": jnp.uint32}


def _moe_params(dtype):
    """A minimal tree with one leaf per optimizer group of both MoE configs."""

    def normal(shape, scale, key):
        return (jax.random.normal(key, shape) * scale).astype(dtype)

    k = jax.random.split(jax.random.PRNGKey(0), 12)
    return {
        "token_embed": normal((_VOCAB, _HIDDEN), 0.02, k[0]),
        "output_proj": normal((_VOCAB, _HIDDEN), 0.05, k[1]),
        "blocks": {
            "0": {
                "attn": {
                    "w_q": normal((_HIDDEN, _HIDDEN), 0.05, k[2]),
                    "attn_gate": normal((_HIDDEN,), 0.1, k[3]),
                },
                "mlp": {
                    "router": normal((_HIDDEN, _NUM_EXPERTS), 0.05, k[4]),
                    "router_bias": normal((_NUM_EXPERTS,), 0.1, k[5]),
                    "expert_mlp": {
                        "w_gate": normal((_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN), 0.05, k[6]),
                        "w_down": normal((_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE), 0.05, k[7]),
                    },
                    "shared": {"w_gate": normal((_INTERMEDIATE, _HIDDEN), 0.05, k[8])},
                },
                "gated_norm": {"weight": normal((_HIDDEN,), 0.5, k[9])},
            }
        },
    }


def _run_steps(config, params, num_steps):
    mesh = Mesh(jax.devices(), axis_names=("i",))
    sharding = NamedSharding(mesh, PartitionSpec())
    params = jax.tree.map(lambda leaf: jax.device_put(leaf, sharding), params)
    initial = params

    optimizer = config.build(num_train_steps=num_steps)
    opt_state = optimizer.init(params)

    @jax.jit
    def step_fn(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    key = jax.random.PRNGKey(1)
    for _ in range(num_steps):
        key, subkey = jax.random.split(key)
        grads = jax.tree.map(lambda p, step_key=subkey: jax.random.normal(step_key, p.shape).astype(p.dtype), params)
        params, opt_state = step_fn(params, opt_state, grads)
    return initial, params


def _bits_differ(reference, current):
    uint = _UINT_DTYPE[reference.dtype.name]
    # The runtime stores or exports parameters in the policy dtype; a zero-rate
    # update is exactly zero, so this cast never hides a change.
    reference_bits = jax.lax.bitcast_convert_type(reference, uint)
    current_bits = jax.lax.bitcast_convert_type(current.astype(reference.dtype), uint)
    return reference_bits != current_bits


@pytest.mark.parametrize(
    ("config_factory", "dtype"),
    [
        (GrugMoeMuonHConfig, jnp.bfloat16),
        (GrugMoeMuonHConfig, jnp.float32),
        (GrugMoeAdamHConfig, jnp.bfloat16),
        (GrugMoeAdamHConfig, jnp.float32),
    ],
)
def test_grug_moe_optimizer_zero_lr_leaves_params_bit_identical(config_factory, dtype):
    config = config_factory(learning_rate=0.0, adam_lr=0.0, lr_schedule="constant")

    initial, final = _run_steps(config, _moe_params(dtype), num_steps=3)

    initial_leaves = jax.tree_util.tree_flatten_with_path(initial)[0]
    final_leaves = jax.tree_util.tree_flatten_with_path(final)[0]
    changed = [
        f"{path} ({reference.dtype.name})"
        for (path, reference), (_, current) in zip(initial_leaves, final_leaves, strict=True)
        if bool(_bits_differ(reference, current).any())
    ]
    assert changed == [], f"parameters changed at learning rate 0: {changed}"


@pytest.mark.parametrize("config_factory", [GrugMoeMuonHConfig, GrugMoeAdamHConfig])
def test_grug_moe_optimizer_positive_lr_updates_params(config_factory):
    config = config_factory(learning_rate=0.01, adam_lr=0.01, lr_schedule="constant")

    initial, final = _run_steps(config, _moe_params(jnp.float32), num_steps=2)

    initial_leaves = jax.tree_util.tree_flatten_with_path(initial)[0]
    final_leaves = jax.tree_util.tree_flatten_with_path(final)[0]
    # Matrix leaves cover every scale-invariant re-projection; one-dimensional
    # MuonH leaves only change sign, so they are excluded.
    unchanged = [
        f"{path}"
        for (path, reference), (_, current) in zip(initial_leaves, final_leaves, strict=True)
        if reference.ndim >= 2 and not _bits_differ(reference, current).any()
    ]
    assert unchanged == [], f"matrix leaves failed to update at a positive learning rate: {unchanged}"


def test_grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group():
    params = {
        "blocks": {
            "0": {
                "mlp": {
                    "router": jnp.ones((8, 4), dtype=jnp.float32),
                    "expert_mlp": {
                        "w_gate": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
                    },
                },
                "shared": {
                    "w_gate": jnp.ones((8, 16), dtype=jnp.float32),
                },
            },
        },
        "token_embed": jnp.ones((128, 8), dtype=jnp.float32),
    }

    mask = GrugMoeAdamHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_up"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_gate"] == "adamh_expert"
    assert mask["token_embed"] == "adam"
