# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe_hero_ep import model
from experiments.grug.moe_hero_ep.launch_hero_sft import (
    SFT_COMPONENT_PREFIX,
    SFT_POOLED_COMPONENT,
    SftStore,
    one_epoch_weight_stages,
    plan_sft_mixture,
    sft_components,
)
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.train import (
    RouterBiasMode,
    SpecialTokenInit,
    _make_train_step,
    initial_state,
    reseed_special_token_columns,
    scale_special_token_updates,
)

_FP32 = jmp.get_policy("params=float32,compute=float32,output=float32")


def _store(packed_sequences: int) -> SftStore:
    return SftStore(cache_path="s3://bucket/store", tokenizer="tok", max_length=8, packed_sequences=packed_sequences)


def _tiny_config():
    return model.GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=4.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )


def _one_device_mesh():
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


def _optimizer():
    return GrugMoeMuonHConfig(learning_rate=1e-2, adam_lr=1e-2, warmup=0, use_syrk=False).build(100)


def test_one_epoch_stages_read_every_sft_store_exactly_once():
    stores = {"a": _store(600_000), "b": _store(200_000), "c": _store(50)}
    plan = plan_sft_mixture(stores, batch_size=16, replay_fraction=0.2)
    assert plan.num_blocks > 1
    _, sft_counts = sft_components(stores, plan)
    stages = one_epoch_weight_stages(sft_counts, {"replay/x": 3.0, "replay/y": 1.0}, plan)

    assert len(stages) == plan.num_blocks
    assert [step for step, _ in stages] == [block * plan.block_size // 16 for block in range(plan.num_blocks)]
    draws: dict[str, int] = {}
    for _, weights in stages:
        counts = {name: round(weight * plan.block_size) for name, weight in weights.items()}
        assert sum(counts.values()) == plan.block_size
        assert counts["replay/x"] >= 1 and counts["replay/y"] >= 1
        for name, count in counts.items():
            draws[name] = draws.get(name, 0) + count
    assert {name: draws[name] for name in sft_counts} == sft_counts
    assert draws["replay/x"] + draws["replay/y"] == plan.run_sequences - plan.sft_sequences


def test_stores_smaller_than_the_block_count_share_one_component():
    stores = {"big": _store(1_000_000), "tiny1": _store(9), "tiny2": _store(15)}
    plan = plan_sft_mixture(stores, batch_size=16, replay_fraction=0.2)
    assert 15 < plan.num_blocks <= 24

    components, counts = sft_components(stores, plan)

    assert set(components) == {SFT_COMPONENT_PREFIX + "big", SFT_POOLED_COMPONENT}
    assert counts[SFT_POOLED_COMPONENT] == 24


def test_sft_stores_offer_no_validation_split():
    # Evaluation asks every component for a validation split; the SFT stores hold only train.
    stores = {"a": _store(600_000), "tiny": _store(14)}
    plan = plan_sft_mixture(stores, batch_size=16, replay_fraction=0.2)
    components, _ = sft_components(stores, plan)
    config = LmDataConfig(tokenizer="gpt2", cache_dir=None, components=components)

    assert config.build_caches("validation") == {}


def test_special_token_updates_scale_only_the_listed_rows():
    with jax.set_mesh(_one_device_mesh()):
        updates = jax.tree.map(jnp.ones_like, model.Transformer.init(_tiny_config(), key=jax.random.key(0)))
        scaled = scale_special_token_updates(updates, (3, 9), 4.0)

    embed_scale = np.asarray(scaled.token_embed)[:, 0]
    head_scale = np.asarray(scaled.output_proj)[0, :]
    expected = np.ones(64)
    expected[[3, 9]] = 4.0
    np.testing.assert_array_equal(embed_scale, expected)
    np.testing.assert_array_equal(head_scale, expected)
    np.testing.assert_array_equal(np.asarray(scaled.stacked_blocks.stacked.mlp.router_bias), 1.0)


def test_reseed_copies_anchor_columns_and_zeros_only_their_moments():
    special = SpecialTokenInit(token_ids=(5, 7), anchor_ids=(1, 2), start_step=0, lr_multiplier=2.0)
    with jax.set_mesh(_one_device_mesh()):
        state = initial_state(_tiny_config(), optimizer=_optimizer(), mp=_FP32, key=jax.random.key(0), ema_beta=None)
        # A resumed optimizer has nonzero moments everywhere.
        trained_moments = jax.tree.map(
            lambda x: jnp.ones_like(x) if jnp.issubdtype(x.dtype, jnp.floating) else x, state.opt_state
        )
        state = state.__class__(**{**vars(state), "opt_state": trained_moments})
        before = np.asarray(state.params.output_proj)
        embed_before = np.asarray(state.params.token_embed)
        reseeded = reseed_special_token_columns(state, special, offload_opt_state=False)

    head = np.asarray(reseeded.params.output_proj)
    np.testing.assert_array_equal(head[:, [5, 7]], before[:, [1, 2]])
    others = [i for i in range(64) if i not in (5, 7)]
    np.testing.assert_array_equal(head[:, others], before[:, others])
    np.testing.assert_array_equal(np.asarray(reseeded.params.token_embed), embed_before)

    head_moments = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(reseeded.opt_state):
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            continue
        values = np.asarray(leaf)
        if str(path[-1]) == ".output_proj":
            head_moments += 1
            np.testing.assert_array_equal(values[:, [5, 7]], 0.0)
            np.testing.assert_array_equal(values[:, others], 1.0)
        else:
            np.testing.assert_array_equal(values, 1.0)
    assert head_moments >= 2  # AdamH keeps a first and a second moment for the LM head


@pytest.mark.parametrize("mode", [RouterBiasMode.FROZEN, RouterBiasMode.UPDATE])
def test_frozen_router_bias_keeps_the_restored_betas(mode):
    config = _tiny_config()
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, config.vocab_size, size=(2, 8), dtype=np.int32))
    batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones((2, 8), jnp.float32).at[:, -1].set(0.0))
    # Adam keeps momentum on the router bias, which a frozen bias must not follow.
    optimizer = optax.adam(1e-2)
    with jax.set_mesh(_one_device_mesh()):
        state = initial_state(config, optimizer=optimizer, mp=_FP32, key=jax.random.key(0), ema_beta=None)
        restored_betas = jax.random.normal(jax.random.key(1), state.pending_qb_betas.shape)
        state = state.__class__(**{**vars(state), "pending_qb_betas": restored_betas})
        # The step donates its state, so keep host copies of what it restores.
        restored_betas = np.asarray(restored_betas)
        expected_bias = np.asarray(
            model.apply_qb_betas(state.params, restored_betas).stacked_blocks.stacked.mlp.router_bias
        )
        step = _make_train_step(optimizer, _FP32, z_loss_weight=1e-4, ema_beta=None, router_bias_mode=mode)
        for _ in range(2):
            state, _, _ = step(state, batch)

    betas = np.asarray(state.pending_qb_betas)
    bias = np.asarray(state.params.stacked_blocks.stacked.mlp.router_bias)
    if mode == RouterBiasMode.FROZEN:
        np.testing.assert_array_equal(betas, restored_betas)
        np.testing.assert_allclose(bias, expected_bias, rtol=0, atol=0)
    else:
        assert not np.array_equal(betas, restored_betas)
