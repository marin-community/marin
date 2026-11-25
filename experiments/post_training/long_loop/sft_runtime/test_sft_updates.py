# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.devices import use_cpu_devices

from experiments.post_training.long_loop.sft_runtime.model import (
    GrugModelConfig,
    Transformer,
    resolve_attention_implementation,
)
from experiments.post_training.long_loop.sft_runtime.train import (
    GrugEvalConfig,
    GrugTrainerConfig,
    _loss_and_grads,
    _make_train_step,
    build_tagged_evaluator,
    initial_state,
    initialize_snowball_sft,
    reseed_lm_head,
)


def _tiny_grug_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=4.0,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )


def test_tagged_eval_loss_is_invariant_to_uniform_weight_scaling():
    with use_cpu_devices(1):
        mesh = compact_grug_mesh(expert_axis_size=1)
        config = _tiny_grug_config()
        example = GrugLmExample.causal(jnp.arange(1, 9, dtype=jnp.int32))
        data = LmDataConfig(
            components={
                "unit": DirectDatasetComponent({"validation": ListAsyncDataset([example])}),
                "half": DirectDatasetComponent(
                    {
                        "validation": ListAsyncDataset(
                            [dataclasses.replace(example, loss_weight=example.loss_weight * 0.5)]
                        )
                    }
                ),
            }
        )
        with jax.set_mesh(mesh):
            model = Transformer.init(config, key=jax.random.key(1))
            evaluator = build_tagged_evaluator(
                data_config=data,
                max_seq_len=config.max_seq_len,
                mesh=mesh,
                eval_cfg=GrugEvalConfig(eval_batch_size=1, compute_bpb=False),
                mp=jmp.get_policy("float32"),
            )
            assert evaluator is not None
            losses = evaluator.evaluate(model).tag_micro_losses
        np.testing.assert_allclose(losses["half"], losses["unit"], rtol=1e-6)


@pytest.mark.timeout(300)
def test_sft_update_preserves_router_and_scales_selected_rows():
    with use_cpu_devices(1):
        config = _tiny_grug_config()
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.adam(0.001)
            mp = jmp.get_policy("float32")
            state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(1), ema_beta=None)
            before_embed = np.asarray(state.params.token_embed).copy()
            before_head = np.asarray(state.params.output_proj).copy()
            seeded = reseed_lm_head(state.params, (30, 31), (2, 3))
            np.testing.assert_array_equal(seeded.token_embed, before_embed)
            np.testing.assert_array_equal(seeded.output_proj[:, 30:32], before_head[:, 2:4])
            np.testing.assert_array_equal(seeded.output_proj[:, :30], before_head[:, :30])
            assert state.params.stacked_blocks.stacked.shared is not None
            legacy = eqx.tree_at(
                lambda m: m.stacked_blocks.stacked.shared, state.params, state.params.stacked_blocks.stacked.shared[0]
            )
            with tempfile.TemporaryDirectory() as directory:
                save_checkpoint({"params": legacy, "pending_qb_betas": state.pending_qb_betas}, 0, directory)
                fresh = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(8), ema_beta=None)
                loaded = initialize_snowball_sft(
                    fresh, directory, mesh, GrugTrainerConfig(special_token_ids=(30, 31), anchor_token_ids=(2, 3))
                )
                for actual, expected in zip(jax.tree.leaves(loaded.params), jax.tree.leaves(seeded), strict=True):
                    np.testing.assert_array_equal(actual, expected)
                assert int(loaded.step) == 0
                for actual, expected in zip(
                    jax.tree.leaves(loaded.opt_state), jax.tree.leaves(fresh.opt_state), strict=True
                ):
                    np.testing.assert_array_equal(actual, expected)
            seeded = eqx.tree_at(
                lambda m: m.stacked_blocks.stacked.mlp.router_bias,
                seeded,
                jnp.array([[0.7, -0.4, 0.2, -0.5], [-0.3, 0.1, 0.8, -0.6]]),
            )
            (_, opt_state) = optimizer.update(jax.tree.map(jnp.ones_like, seeded), state.opt_state, seeded)
            state = dataclasses.replace(
                state,
                params=seeded,
                opt_state=opt_state,
                pending_qb_betas=jnp.tile(jnp.array([0.1, 0.2, 0.3, 0.4]), (2, 1)),
            )
            tokens = jnp.array([[1, 2, 30, 4, 5, 31, 7, 8]], dtype=jnp.int32)
            batch = GrugLmExample(tokens, jnp.ones_like(tokens, dtype=jnp.float32).at[:, -1].set(0))
            qb = state.params
            ((_, _), grads) = _loss_and_grads(qb, batch, mp, None)
            (ordinary_updates, _) = optimizer.update(grads, state.opt_state, qb)
            expected_embed = np.asarray(qb.token_embed + ordinary_updates.token_embed).copy()
            expected_selected = np.asarray(qb.token_embed[30:] + 4 * ordinary_updates.token_embed[30:]).copy()
            expected_head = np.asarray(qb.output_proj[:, 30:] + 4 * ordinary_updates.output_proj[:, 30:]).copy()
            expected_router = np.asarray(qb.stacked_blocks.stacked.mlp.router_bias).copy()
            expected_pending = np.asarray(state.pending_qb_betas).copy()
            step = _make_train_step(
                optimizer, mp, z_loss_weight=0, ema_beta=None, special_token_ids=(30, 31), special_token_lr_multiplier=4
            )
            (result, metrics, _) = step(state, batch)
            assert np.isfinite(float(metrics["train/loss"]))
            np.testing.assert_allclose(result.params.token_embed[:30], expected_embed[:30], rtol=1e-06, atol=1e-07)
            np.testing.assert_allclose(result.params.token_embed[30:], expected_selected, rtol=1e-06, atol=1e-07)
            np.testing.assert_allclose(result.params.output_proj[:, 30:], expected_head, rtol=1e-06, atol=1e-07)
            np.testing.assert_array_equal(result.params.stacked_blocks.stacked.mlp.router_bias, expected_router)
            np.testing.assert_array_equal(result.pending_qb_betas, expected_pending)


@pytest.mark.timeout(180)
def test_splash_mixed_layer_windows_match_reference(monkeypatch):
    # Run the actual Splash kernel on CPU, preserving the model's scanned mask selection.
    monkeypatch.setattr(
        splash_attention_kernel,
        "make_splash_mha",
        functools.partial(splash_attention_kernel.make_splash_mha, interpret=True),
    )
    config = GrugModelConfig(
        vocab_size=256,
        hidden_dim=128,
        intermediate_dim=128,
        shared_expert_intermediate_dim=128,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=128,
        max_seq_len=512,
        sliding_window=128,
        global_every=2,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        capacity_factor=4.0,
    )
    with use_cpu_devices(1):
        mesh = Mesh(
            np.asarray(jax.devices("cpu")[:1]).reshape(1, 1, 1, 1, 1),
            ("replica_dcn", "data", "expert", "context", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            reference = Transformer.init(config, key=jax.random.key(29))
            splash = Transformer.init(
                dataclasses.replace(config, attention_implementation="tpu_splash"), key=jax.random.key(29)
            )
            full_causal = Transformer.init(dataclasses.replace(config, sliding_window=512), key=jax.random.key(29))
            for actual, expected in zip(jax.tree.leaves(splash), jax.tree.leaves(reference), strict=True):
                np.testing.assert_array_equal(actual, expected)
            sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
            tokens = jax.device_put(np.random.default_rng(4).integers(0, 256, (1, 512), dtype=np.int32), sharding)
            segments = jax.device_put(np.asarray([[0] * 256 + [1] * 256], dtype=np.int32), sharding)
            mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
            actual = np.asarray(jax.jit(lambda x: splash.logits(x, mask))(tokens))
            expected = np.asarray(jax.jit(lambda x: reference.logits(x, mask))(tokens))
            without_local_window = np.asarray(jax.jit(lambda x: full_causal.logits(x, mask))(tokens))
    assert np.max(np.abs(actual - expected)) <= 1e-6
    assert np.max(np.abs(expected - without_local_window)) > 1e-4


@pytest.mark.parametrize(
    ("configured", "backend", "expected"),
    [
        (None, "tpu", "tpu_splash"),
        (None, "gpu", None),
        ("tpu_splash", "cpu", "tpu_splash"),
        ("reference", "tpu", "reference"),
    ],
)
def test_unset_attention_implementation_resolves_to_splash_on_tpu(configured, backend, expected):
    # An unset implementation reaches Splash on TPU, so it must get Splash's static window selection.
    assert resolve_attention_implementation(configured, backend) == expected
