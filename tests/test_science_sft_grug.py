# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.schedule import BatchSchedule
from marin.datakit.sft import SftSourceCounts, SftTokenStore

from experiments.grug.science_sft.launch import MODEL_PATH, build_run_config
from experiments.grug.science_sft.prepare import SOURCE_NAME
from experiments.grug.science_sft.train import (
    RouterBiasUpdate,
    RouterFreeze,
    _apply_qb_betas,
    _make_train_step,
    build_train_dataset,
    initial_state,
    reinitialize_token_rows,
)
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig


def test_converted_science_run_uses_one_pass_and_per_step_qb():
    store = SftTokenStore(
        path="local",
        cache_path="local/train",
        tokenizer=MODEL_PATH,
        max_length=32_768,
        seed=0,
        sources={SOURCE_NAME: SftSourceCounts(conversations=130)},
        packed_sequences=130,
    )
    config = build_run_config(store, "test")

    assert config.trainer.trainer.num_train_steps == 2
    assert config.trainer.max_data_epochs == 1
    assert config.trainer.router_freeze == RouterFreeze.BIAS
    assert config.trainer.router_bias_update == RouterBiasUpdate.PER_STEP


@pytest.mark.asyncio
async def test_train_dataset_masks_cross_conversation_loss():
    example = GrugLmExample.causal(
        tokens=jnp.arange(4),
        loss_weight=jnp.ones(4),
        segment_ids=jnp.array([0, 0, 1, 1]),
    )
    config = LmDataConfig(
        components={"direct": DirectDatasetComponent(datasets={"train": ListAsyncDataset([example])})},
        train_weights={"direct": 1.0},
        vocab_size=4,
        tokenizer="passthrough",
        block_cross_document_attention=True,
    )

    dataset = build_train_dataset(
        config,
        max_seq_len=4,
        batch_schedule=BatchSchedule(1),
        key=jax.random.key(0),
    )
    [prepared] = await dataset.get_batch([0])

    np.testing.assert_array_equal(prepared.loss_weight, [1, 0, 1, 0])
    np.testing.assert_array_equal(prepared.attn_mask.segment_ids[0], [0, 0, 1, 1])


def test_frozen_router_and_exact_token_initialization():
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    cfg = GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        sliding_window=8,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
    )
    with jax.set_mesh(mesh):
        opt = optax.adam(1e-3)
        mp = jmp.get_policy("f32")
        s = initial_state(cfg, optimizer=opt, mp=mp, key=jax.random.key(0), ema_beta=None)
        s = dataclasses.replace(
            s,
            opt_state=jax.tree.map(lambda x: x + 1 if eqx.is_array(x) else x, s.opt_state),
            pending_qb_betas=jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        )
        original_embed = np.array(s.params.token_embed)
        s = reinitialize_token_rows(s, (60, 61), ((2,), (3,)))
        np.testing.assert_array_equal(np.array(s.params.token_embed), original_embed)
        np.testing.assert_array_equal(
            np.array(s.params.output_proj)[:, 60:62], np.array(s.params.output_proj)[:, [2, 3]]
        )
        for path, value in jax.tree_util.tree_flatten_with_path(s.opt_state)[0]:
            if str(path[-1]) == ".token_embed":
                np.testing.assert_array_equal(np.array(value), 1)
            if str(path[-1]) == ".output_proj":
                np.testing.assert_array_equal(np.array(value)[:, 60:62], 0)
        baseline = _apply_qb_betas(s.params, s.pending_qb_betas)
        r = np.array(baseline.stacked_blocks.stacked.mlp.router)
        bias = np.array(baseline.stacked_blocks.stacked.mlp.router_bias)
        pending = np.array(s.pending_qb_betas)
        other = np.array(s.params.output_proj)
        batch = GrugLmExample(
            tokens=jax.sharding.reshard(jnp.arange(64, dtype=jnp.int32).reshape(4, 16) % 64, P("data", None)),
            loss_weight=jax.sharding.reshard(jnp.ones((4, 16)), P("data", None)),
            attn_mask=GrugAttentionMask.causal(),
        )
        step = _make_train_step(opt, mp, z_loss_weight=1e-4, ema_beta=None)
        scaled_step = _make_train_step(
            opt,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            special_token_lr_ids=(60, 61),
            special_token_lr_multiplier=4.0,
        )
        bias_only_step = _make_train_step(
            opt,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            router_freeze=RouterFreeze.BIAS,
        )
        original = jax.tree.map(lambda x: x.copy() if eqx.is_array(x) else x, s)
        scaled_input = jax.tree.map(lambda x: x.copy() if eqx.is_array(x) else x, s)
        baseline_input = jax.tree.map(lambda x: x.copy() if eqx.is_array(x) else x, s)
        scaled, _, _ = scaled_step(scaled_input, batch)
        normal, _, _ = step(baseline_input, batch)
        bias_only_input = jax.tree.map(lambda x: x.copy() if eqx.is_array(x) else x, s)
        bias_only, _, _ = bias_only_step(bias_only_input, batch)
        np.testing.assert_array_equal(np.array(bias_only.params.stacked_blocks.stacked.mlp.router_bias), bias)
        assert not np.array_equal(np.array(bias_only.params.stacked_blocks.stacked.mlp.router), r)
        for before, after, boosted in zip(
            jax.tree.leaves(original.params),
            jax.tree.leaves(normal.params),
            jax.tree.leaves(scaled.params),
            strict=True,
        ):
            expected = np.array(after)
            if before.shape == original.params.token_embed.shape:
                expected[60:62] = np.array(before)[60:62] + 4 * (np.array(after)[60:62] - np.array(before)[60:62])
            elif before.shape == original.params.output_proj.shape:
                expected[:, 60:62] = np.array(before)[:, 60:62] + 4 * (
                    np.array(after)[:, 60:62] - np.array(before)[:, 60:62]
                )
            np.testing.assert_allclose(np.array(boosted), expected, rtol=1e-6, atol=1e-7)
        for normal_moment, scaled_moment in zip(
            jax.tree.leaves(normal.opt_state), jax.tree.leaves(scaled.opt_state), strict=True
        ):
            np.testing.assert_array_equal(np.array(normal_moment), np.array(scaled_moment))
        for _ in range(3):
            s, metrics, _ = step(s, batch)
            jax.block_until_ready(s)
            np.testing.assert_array_equal(np.array(s.params.stacked_blocks.stacked.mlp.router), r)
            np.testing.assert_array_equal(np.array(s.params.stacked_blocks.stacked.mlp.router_bias), bias)
            np.testing.assert_array_equal(np.array(s.pending_qb_betas), pending)
        assert np.isfinite(float(metrics["train/loss"]))
        assert not np.array_equal(np.array(s.params.output_proj), other)

        qb_step = _make_train_step(
            opt,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            router_freeze=RouterFreeze.BIAS,
            router_bias_update=RouterBiasUpdate.PER_STEP,
        )
        qb_state, qb_metrics, _ = qb_step(original, batch)
        new_betas = np.asarray(qb_state.pending_qb_betas)
        np.testing.assert_allclose(new_betas, np.asarray(qb_metrics["qb_beta_per_layer"]))
        assert not np.array_equal(new_betas, pending)
        qb_state, _, _ = qb_step(qb_state, batch)
        expected_bias = -new_betas
        expected_bias -= expected_bias.mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(np.asarray(qb_state.params.stacked_blocks.stacked.mlp.router_bias), expected_bias)
