# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
import dataclasses
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.testing.devices import use_cpu_devices

from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig
from experiments.post_training.long_loop.sft_runtime.state_rebind import HostRebindConfig
from experiments.post_training.long_loop.sft_runtime.temperature_matching import (
    TemperatureMatchingConfig,
    chunked_linear_tempered_kl_weighted_sum,
    global_l2_norm,
    tm_row_mask_for_mixture_indices,
    validate_tm_source_names,
)
from experiments.post_training.long_loop.sft_runtime.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    _accumulated_loss_and_grads,
    _accumulated_loss_and_grads_with_tm,
    _balance_attention_rows_with_mask,
    _compile_first_step_and_rebind,
    _make_tm_grad_ratio_step,
    _make_train_step,
    cast_teacher_params,
    initial_state,
    initialize_snowball_sft,
    prepare_frozen_teacher,
)


def _assert_fp32_state(state) -> None:
    for leaf in jax.tree.leaves((state.params, state.opt_state)):
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == jnp.float32, f"Expected float32 state leaf, got {leaf.dtype}"


def _tiny_model_config(*, report_capacity_overflow: bool = False) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=24,
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
        capacity_factor=1.0,
        attention_implementation="reference",
        moe_implementation="scatter",
        report_capacity_overflow=report_capacity_overflow,
    )


def _dense_tempered_kl_weighted_sum(
    student_hidden: jax.Array,
    student_lm_head: jax.Array,
    teacher_hidden: jax.Array,
    teacher_lm_head: jax.Array,
    weight: jax.Array,
    temperature: float,
) -> jax.Array:
    """Naive full-vocabulary FP32 reference without Hinton's T^2 factor."""
    s_logits = jnp.einsum("...h,hv->...v", student_hidden.astype(jnp.float32), student_lm_head.astype(jnp.float32))
    t_logits = jnp.einsum(
        "...h,hv->...v",
        jax.lax.stop_gradient(teacher_hidden).astype(jnp.float32),
        jax.lax.stop_gradient(teacher_lm_head).astype(jnp.float32),
    )
    log_q = jax.nn.log_softmax(t_logits / temperature, axis=-1)
    q = jnp.exp(log_q)
    log_p = jax.nn.log_softmax(s_logits / temperature, axis=-1)
    kl = jnp.sum(q * (log_q - log_p), axis=-1)
    w = weight.astype(jnp.float32)
    kl_sharding = jax.typeof(kl).sharding
    if isinstance(kl_sharding, NamedSharding) and kl_sharding.mesh.axis_names:
        w = jax.sharding.reshard(w, kl_sharding)
    return jnp.sum(kl * w)


def test_temperature_matching_config_validation():
    # Disabled config accepts defaults without triggering active checks.
    assert not TemperatureMatchingConfig().enabled

    # Critical hyperparameters must be explicit (no silent defaults) when enabled=True.
    with pytest.raises(ValueError, match="weight"):
        TemperatureMatchingConfig(enabled=True, temperature=2.5, microbatch_count=2)
    with pytest.raises(ValueError, match="temperature"):
        TemperatureMatchingConfig(enabled=True, weight=1.0 / 3.0, microbatch_count=2)
    with pytest.raises(ValueError, match="microbatch_count"):
        TemperatureMatchingConfig(enabled=True, weight=1.0 / 3.0, temperature=2.5)

    with pytest.raises(ValueError, match="weight"):
        TemperatureMatchingConfig(enabled=True, weight=0.0, temperature=2.5, microbatch_count=2)
    with pytest.raises(ValueError, match="temperature"):
        TemperatureMatchingConfig(enabled=True, weight=1.0 / 3.0, temperature=-1.0, microbatch_count=2)
    with pytest.raises(ValueError, match="microbatch_count"):
        TemperatureMatchingConfig(enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=0)
    with pytest.raises(ValueError, match="token_block_size"):
        TemperatureMatchingConfig(
            enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=2, token_block_size=0
        )
    with pytest.raises(ValueError, match="at most one"):
        TemperatureMatchingConfig(
            enabled=True,
            weight=1.0 / 3.0,
            temperature=2.5,
            microbatch_count=2,
            disabled_sources=("a",),
            enabled_sources=("b",),
        )

    # microbatch_count must not exceed gradient_accumulation_steps.
    with pytest.raises(ValueError, match="microbatch_count"):
        GrugTrainerConfig(
            gradient_accumulation_steps=4,
            temperature_matching=TemperatureMatchingConfig(
                enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=5
            ),
        )

    # Expert-parallel capacity drops are rejected by default so student and teacher see identical tokens.
    with pytest.raises(ValueError, match="dropless MoE"):
        GrugTrainerConfig(
            gradient_accumulation_steps=8,
            expert_axis_size=2,
            temperature_matching=TemperatureMatchingConfig(
                enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=2
            ),
        )

    # Explicit opt-in passes when needed.
    cfg = GrugTrainerConfig(
        gradient_accumulation_steps=8,
        expert_axis_size=2,
        temperature_matching=TemperatureMatchingConfig(
            enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=2, allow_ep_capacity_drops=True
        ),
    )
    assert cfg.temperature_matching.enabled


@pytest.mark.parametrize("token_block_size", [1, 3, 5, 16, 1024])
def test_chunked_tempered_kl_matches_dense_value_and_grads(token_block_size: int):
    rng = np.random.default_rng(42)
    s_h = jnp.asarray(rng.normal(size=(2, 7, 16)).astype(np.float32))
    s_w = jnp.asarray(rng.normal(size=(16, 24)).astype(np.float32))
    t_h = jnp.asarray(rng.normal(size=(2, 7, 16)).astype(np.float32))
    t_w = jnp.asarray(rng.normal(size=(16, 24)).astype(np.float32))
    weight = jnp.asarray(np.array([[1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 1]], dtype=np.float32))
    temperature = 2.5

    ref_val, ref_grads = jax.value_and_grad(_dense_tempered_kl_weighted_sum, argnums=(0, 1, 2, 3))(
        s_h, s_w, t_h, t_w, weight, temperature
    )
    chunk_val, chunk_grads = jax.value_and_grad(
        lambda sh, sw, th, tw: chunked_linear_tempered_kl_weighted_sum(
            sh, sw, th, tw, weight, temperature=temperature, token_block_size=token_block_size
        ),
        argnums=(0, 1, 2, 3),
    )(s_h, s_w, t_h, t_w)

    np.testing.assert_allclose(chunk_val, ref_val, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(chunk_grads[0], ref_grads[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(chunk_grads[1], ref_grads[1], rtol=1e-5, atol=1e-5)
    # Teacher must remain strictly frozen (zero cotangent).
    np.testing.assert_array_equal(chunk_grads[2], np.zeros_like(t_h))
    np.testing.assert_array_equal(chunk_grads[3], np.zeros_like(t_w))


def test_chunked_tempered_kl_omits_hinton_t_squared_factor():
    """Verify Proposition E.1 scaling: without T^2, small-perturbation gradient shrinks as ~1/T^2."""
    rng = np.random.default_rng(7)
    t_h = jnp.asarray(rng.normal(size=(1, 6, 16)).astype(np.float32))
    t_w = jnp.asarray(rng.normal(size=(16, 24)).astype(np.float32))
    # Perturb student logits slightly from teacher.
    eps_h = jnp.asarray(0.01 * rng.normal(size=(1, 6, 16)).astype(np.float32))
    s_h = t_h + eps_h
    s_w = t_w
    weight = jnp.ones((1, 6), dtype=jnp.float32)

    grad_t1 = jax.grad(
        lambda sh: chunked_linear_tempered_kl_weighted_sum(
            sh, s_w, t_h, t_w, weight, temperature=1.0, token_block_size=4
        )
    )(s_h)
    grad_t25 = jax.grad(
        lambda sh: chunked_linear_tempered_kl_weighted_sum(
            sh, s_w, t_h, t_w, weight, temperature=2.5, token_block_size=4
        )
    )(s_h)

    # Exact closed-form check: dKL/dz_s = (p_T - q_T) / T, NOT T * (p_T - q_T).
    z_s = jnp.einsum("bsh,hv->bsv", s_h, s_w)
    z_t = jnp.einsum("bsh,hv->bsv", t_h, t_w)
    expected_dlogits_t25 = (jax.nn.softmax(z_s / 2.5, axis=-1) - jax.nn.softmax(z_t / 2.5, axis=-1)) / 2.5
    expected_grad_t25 = jnp.einsum("bsv,hv->bsh", expected_dlogits_t25, s_w)
    np.testing.assert_allclose(grad_t25, expected_grad_t25, rtol=1e-5, atol=1e-6)
    # Without Hinton's T^2 factor, T=2.5 is damped by ~1/T^2 (~6.25x smaller), never larger than T=1.
    ratio = float(jnp.linalg.norm(grad_t25) / jnp.linalg.norm(grad_t1))
    assert ratio < 0.35, f"Expected strong 1/T^2 damping without Hinton T^2 factor, got norm ratio {ratio}"


def test_chunked_tempered_kl_sharded_matches_single_device():
    rng = np.random.default_rng(11)
    s_h_np = rng.normal(size=(4, 8, 16)).astype(np.float32)
    s_w_np = rng.normal(size=(16, 24)).astype(np.float32)
    t_h_np = rng.normal(size=(4, 8, 16)).astype(np.float32)
    t_w_np = rng.normal(size=(16, 24)).astype(np.float32)
    w_np = np.array(
        [
            [1, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=np.float32,
    )

    ref_val, (ref_g_sh, ref_g_sw) = jax.value_and_grad(_dense_tempered_kl_weighted_sum, argnums=(0, 1))(
        jnp.asarray(s_h_np), jnp.asarray(s_w_np), jnp.asarray(t_h_np), jnp.asarray(t_w_np), jnp.asarray(w_np), 2.5
    )

    with use_cpu_devices(4):
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 2, 2, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            h_shard = NamedSharding(mesh, P("data", "context", None))
            w_shard = NamedSharding(mesh, P("data", "context"))
            lm_shard = NamedSharding(mesh, P(("data", "context"), "model"))
            s_h = jax.device_put(s_h_np, h_shard)
            t_h = jax.device_put(t_h_np, h_shard)
            s_w = jax.device_put(s_w_np, lm_shard)
            t_w = jax.device_put(t_w_np, lm_shard)
            weight = jax.device_put(w_np, w_shard)

            @jax.jit
            def run(sh, sw, th, tw, wt):
                return jax.value_and_grad(
                    lambda a, b: chunked_linear_tempered_kl_weighted_sum(
                        a, b, th, tw, wt, temperature=2.5, token_block_size=3
                    ),
                    argnums=(0, 1),
                )(sh, sw)

            val, (g_sh, g_sw) = run(s_h, s_w, t_h, t_w, weight)
            np.testing.assert_allclose(val, ref_val, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.asarray(g_sh), np.asarray(ref_g_sh), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.asarray(g_sw), np.asarray(ref_g_sw), rtol=1e-5, atol=1e-5)


@pytest.mark.timeout(300)
def test_same_step_2_of_8_microbatches_gradient_parity_and_watch_ratio():
    """Verify CE on all 8 microbatches + TM on 2 of 8 inside one step, watch ratio, and dropless MoE."""
    with use_cpu_devices(2):
        config = _tiny_model_config(report_capacity_overflow=True)
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 2, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.adam(1e-3)
            mp = jmp.get_policy("float32")
            student_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(1), ema_beta=None)
            teacher_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(2), ema_beta=None)
            teacher_params = cast_teacher_params(teacher_state.params, mp)

            # 8 microbatches of 1 row each (batch_size=8, steps=8, tm_microbatches=2).
            rng = np.random.default_rng(123)
            tokens_np = rng.integers(1, 22, size=(8, 8), dtype=np.int32)
            weights_np = np.array(
                [
                    [1, 0, 1, 1, 0, 1, 1, 0],  # mb 0 (TM active)
                    [0, 1, 1, 0, 1, 0, 1, 0],  # mb 1 (TM active)
                    [1, 1, 1, 1, 1, 1, 1, 0],  # mb 2 (CE only)
                    [0, 0, 1, 1, 0, 0, 1, 0],  # mb 3 (CE only)
                    [1, 0, 0, 1, 1, 0, 0, 0],  # mb 4 (CE only)
                    [0, 0, 0, 0, 0, 0, 0, 0],  # mb 5 (CE only, all zero)
                    [1, 1, 0, 0, 1, 1, 0, 0],  # mb 6 (CE only)
                    [1, 1, 1, 0, 1, 1, 1, 0],  # mb 7 (CE only)
                ],
                dtype=np.float32,
            )
            segments_np = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32), (8, 1))
            b_shard = NamedSharding(mesh, P("data", None))
            batch = GrugLmExample(
                jax.device_put(tokens_np, b_shard),
                jax.device_put(weights_np, b_shard),
                AttentionMask.causal().with_segment_ids(jax.device_put(segments_np, b_shard)),
            )

            # Verify teacher.hidden_states matches teacher(...)[0] bitwise.
            h_direct, _ = teacher_params(batch.tokens, mask=batch.attn_mask)
            h_fast = teacher_params.hidden_states(batch.tokens, mask=batch.attn_mask)
            np.testing.assert_allclose(np.asarray(h_fast), np.asarray(h_direct), rtol=0.0, atol=0.0)

            tm_cfg = TemperatureMatchingConfig(
                enabled=True,
                weight=1.0 / 3.0,
                temperature=2.5,
                microbatch_count=2,
                token_block_size=4,
                offload_teacher=False,
                log_grad_norm_ratio=True,
            )

            # Combined accumulated loss & grads (CE on all 8 + TM on first 2).
            (total_loss, metrics), grads_combined = _accumulated_loss_and_grads_with_tm(
                student_state.params, batch, teacher_params, None, mp, None, 8, tm_cfg
            )

            # Independent CE-only over all 8 microbatches.
            (ce_loss, _), grads_ce = _accumulated_loss_and_grads(student_state.params, batch, mp, None, 8)

            # Independent TM-only reference over first 2 rows (dense full-vocab KL).
            sub_tokens = jax.device_put(tokens_np[:2], b_shard)
            sub_weights = jax.device_put(weights_np[:2], b_shard)
            sub_mask = AttentionMask.causal().with_segment_ids(jax.device_put(segments_np[:2], b_shard))
            t_sub_hidden = jax.lax.stop_gradient(teacher_params.hidden_states(sub_tokens, mask=sub_mask))

            def ref_tm_loss(model):
                model = eqx.tree_at(
                    lambda m: m.stacked_blocks.stacked.mlp.router_bias,
                    model,
                    replace_fn=jax.lax.stop_gradient,
                )
                s_h, _ = mp.cast_to_compute(model)(sub_tokens, mask=sub_mask)
                kl_sum = _dense_tempered_kl_weighted_sum(
                    s_h,
                    mp.cast_to_compute(model).output_proj,
                    t_sub_hidden,
                    teacher_params.output_proj,
                    sub_weights,
                    2.5,
                )
                return (1.0 / 3.0) * kl_sum / jnp.sum(sub_weights, dtype=jnp.float32)

            ref_tm_weighted, grads_tm_ref = jax.value_and_grad(ref_tm_loss)(student_state.params)

            # Loss & gradient sum parity.
            np.testing.assert_allclose(metrics["train/cross_entropy_loss"], ce_loss, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(metrics["train/tm/weighted_loss"], ref_tm_weighted, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(total_loss, ce_loss + ref_tm_weighted, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(metrics["train/tm/active_tokens"], float(weights_np[:2].sum()), atol=1e-6)
            np.testing.assert_allclose(metrics["train/tm/active_fraction"], 1.0, atol=1e-6)

            for g_comb, g_c, g_t in zip(
                jax.tree.leaves(grads_combined),
                jax.tree.leaves(grads_ce),
                jax.tree.leaves(grads_tm_ref),
                strict=True,
            ):
                np.testing.assert_allclose(np.asarray(g_comb), np.asarray(g_c + g_t), rtol=2e-4, atol=1e-5)

            # Watch metric ||grad_TM|| / ||grad_CE|| parity + linear scaling with weight.
            ratio_step = _make_tm_grad_ratio_step(
                mp, z_loss_weight=0.0, accumulation_steps=8, temperature_matching=tm_cfg
            )
            watch_stats = ratio_step(student_state.params, batch, teacher_params, None)
            expected_ce_norm = float(global_l2_norm(grads_ce))
            expected_tm_norm = float(global_l2_norm(grads_tm_ref))
            np.testing.assert_allclose(watch_stats["watch/tm/ce_grad_norm"], expected_ce_norm, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(watch_stats["watch/tm/grad_norm"], expected_tm_norm, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                watch_stats["watch/tm/grad_norm_ratio"], expected_tm_norm / expected_ce_norm, rtol=1e-4, atol=1e-6
            )

            # Doubling weight doubles ||grad_TM|| and ratio exactly.
            tm_cfg_2x = dataclasses.replace(tm_cfg, weight=2.0 / 3.0)
            ratio_step_2x = _make_tm_grad_ratio_step(
                mp, z_loss_weight=0.0, accumulation_steps=8, temperature_matching=tm_cfg_2x
            )
            watch_stats_2x = ratio_step_2x(student_state.params, batch, teacher_params, None)
            np.testing.assert_allclose(
                watch_stats_2x["watch/tm/grad_norm_ratio"], 2.0 * float(watch_stats["watch/tm/grad_norm_ratio"]), rtol=1e-5
            )

            # Dropless confirmation: expert_axis_size == 1 has strictly zero capacity drops.
            assert int(np.asarray(metrics["moe/dropped_assignments"]).sum()) == 0

            # Full train_step keeps teacher frozen and GrugTrainState strictly FP32 with no teacher leaves.
            teacher_before = [np.asarray(x).copy() for x in jax.tree.leaves(teacher_params)]
            train_step = _make_train_step(
                optimizer,
                mp,
                z_loss_weight=0.0,
                ema_beta=None,
                accumulation_steps=8,
                temperature_matching=tm_cfg,
            )
            next_state, step_metrics, _ = train_step(student_state, batch, teacher_params, None)
            assert int(next_state.step) == 1
            assert np.isfinite(float(step_metrics["train/loss"]))
            _assert_fp32_state(next_state)
            for before, after in zip(teacher_before, jax.tree.leaves(teacher_params), strict=True):
                np.testing.assert_array_equal(before, np.asarray(after))


@pytest.mark.timeout(300)
def test_step_zero_self_teacher_has_zero_tm_loss_and_matches_ce_exactly():
    """At step 0 (student == teacher), KL=0, grad_TM=0, and update equals pure CE."""
    with use_cpu_devices(1):
        config = _tiny_model_config()
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.adam(1e-3)
            mp = jmp.get_policy("float32")
            ce_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(5), ema_beta=None)
            tm_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(5), ema_beta=None)
            teacher_params = cast_teacher_params(tm_state.params, mp)

            tokens = jnp.asarray((np.arange(32).reshape(4, 8) % 21) + 1, dtype=jnp.int32)
            weights = jnp.ones((4, 8), dtype=jnp.float32).at[:, -1].set(0.0)
            batch = GrugLmExample(tokens=tokens, loss_weight=weights)

            tm_cfg = TemperatureMatchingConfig(
                enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=2, token_block_size=4, offload_teacher=False
            )
            ce_step = _make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None, accumulation_steps=4)
            tm_step = _make_train_step(
                optimizer, mp, z_loss_weight=0.0, ema_beta=None, accumulation_steps=4, temperature_matching=tm_cfg
            )
            ce_next, ce_metrics, _ = ce_step(ce_state, batch)
            tm_next, tm_metrics, _ = tm_step(tm_state, batch, teacher_params, None)

            np.testing.assert_allclose(tm_metrics["train/tm/loss"], 0.0, atol=1e-6)
            np.testing.assert_allclose(tm_metrics["train/loss"], ce_metrics["train/loss"], rtol=1e-5, atol=1e-6)
            for a, b in zip(jax.tree.leaves(tm_next.params), jax.tree.leaves(ce_next.params), strict=True):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


@pytest.mark.timeout(300)
def test_per_source_switch_masks_disabled_rows_and_handles_all_disabled_gracefully():
    """Verify MixtureDataset per-source switch, balance_attention_rows lockstep, and all-off zero-NaN behavior."""
    ds_general = ListAsyncDataset([{"src": "general", "i": i} for i in range(32)])
    ds_repair = ListAsyncDataset([{"src": "repair", "i": i} for i in range(32)])
    mixture = MixtureDataset(
        datasets={"general": ds_general, "repair": ds_repair},
        weights={"general": 0.75, "repair": 0.25},
        block_size=8,
        key=0,
    )

    tm_deny = TemperatureMatchingConfig(
        enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=1, disabled_sources=("repair",), offload_teacher=False
    )
    tm_allow = TemperatureMatchingConfig(
        enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=1, enabled_sources=("general",), offload_teacher=False
    )
    validate_tm_source_names(mixture, tm_deny)
    validate_tm_source_names(mixture, tm_allow)

    with pytest.raises(ValueError, match="unknown mixture sources"):
        validate_tm_source_names(
            mixture,
            TemperatureMatchingConfig(
                enabled=True,
                weight=1.0 / 3.0,
                temperature=2.5,
                microbatch_count=1,
                disabled_sources=("nonexistent",),
                offload_teacher=False,
            ),
        )

    mask_deny = tm_row_mask_for_mixture_indices(mixture, range(8), tm_deny)
    mask_allow = tm_row_mask_for_mixture_indices(mixture, range(8), tm_allow)
    np.testing.assert_array_equal(mask_deny, mask_allow)
    # Block size 8 with 0.75 / 0.25 weights yields exactly 6 general (1.0) and 2 repair (0.0) rows.
    assert mask_deny.shape == (8,)
    assert int(mask_deny.sum()) == 6

    # Verify all-disabled TM rows produce 0 TM loss, 0 TM gradient, finite total loss, and exact CE parity.
    with use_cpu_devices(2):
        config = _tiny_model_config()
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 2, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.adam(1e-3)
            mp = jmp.get_policy("float32")
            student_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(3), ema_beta=None)
            teacher_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(4), ema_beta=None)
            teacher_params = cast_teacher_params(teacher_state.params, mp)

            tokens = jax.device_put(
                np.asarray((np.arange(32).reshape(4, 8) % 21) + 1, dtype=np.int32), NamedSharding(mesh, P("data", None))
            )
            weights = jax.device_put(
                np.ones((4, 8), dtype=np.float32), NamedSharding(mesh, P("data", None))
            )
            segments = jax.device_put(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [0, 0, 1, 1, 2, 2, 3, 3],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                    ],
                    dtype=np.int32,
                ),
                NamedSharding(mesh, P("data", None)),
            )
            batch = GrugLmExample(tokens=tokens, loss_weight=weights, attn_mask=AttentionMask.causal().with_segment_ids(segments))

            # Check balance_attention_rows permutes row_mask in lockstep with tokens.
            row_ids = jax.device_put(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32), NamedSharding(mesh, P("data")))
            bal_batch, bal_mask, _ = _balance_attention_rows_with_mask(batch, row_ids)
            bal_tokens_np = np.asarray(bal_batch.tokens)
            orig_tokens_np = np.asarray(batch.tokens)
            bal_mask_np = np.asarray(bal_mask)
            orig_ids_np = np.asarray(row_ids)
            for row_idx in range(4):
                orig_idx = next(j for j in range(4) if np.array_equal(bal_tokens_np[row_idx], orig_tokens_np[j]))
                assert float(bal_mask_np[row_idx]) == float(orig_ids_np[orig_idx])

            # All-zero row mask on TM microbatches => exact CE parity, 0 active tokens, no NaN.
            zero_mask = jax.device_put(np.zeros((4,), dtype=np.float32), NamedSharding(mesh, P("data")))
            tm_cfg = TemperatureMatchingConfig(
                enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=1, token_block_size=4, offload_teacher=False
            )
            (tm_total, tm_metrics), tm_grads = _accumulated_loss_and_grads_with_tm(
                student_state.params, batch, teacher_params, zero_mask, mp, None, 2, tm_cfg
            )
            (ce_total, _), ce_grads = _accumulated_loss_and_grads(student_state.params, batch, mp, None, 2)
            assert float(tm_metrics["train/tm/loss"]) == 0.0
            assert float(tm_metrics["train/tm/active_tokens"]) == 0.0
            np.testing.assert_allclose(tm_total, ce_total, rtol=1e-5, atol=1e-6)
            for g1, g2 in zip(jax.tree.leaves(tm_grads), jax.tree.leaves(ce_grads), strict=True):
                np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), rtol=1e-5, atol=1e-6)


@pytest.mark.timeout(300)
def test_accumulated_tm_full_batch_parity_with_unequal_weights():
    """Verify M=4, K=4 matches M=1, K=1 with unequal loss_weight and partial tm_row_mask."""
    with use_cpu_devices(1):
        config = _tiny_model_config()
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.adam(1e-3)
            mp = jmp.get_policy("float32")
            student_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(10), ema_beta=None)
            teacher_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(11), ema_beta=None)
            teacher_params = cast_teacher_params(teacher_state.params, mp)

            rng = np.random.default_rng(99)
            tokens = jnp.asarray(rng.integers(1, 22, size=(4, 8), dtype=np.int32))
            weights = jnp.asarray(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 0, 1, 0, 0, 0],
                    ],
                    dtype=np.float32,
                )
            )
            row_mask = jnp.asarray(np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32))
            batch = GrugLmExample(tokens=tokens, loss_weight=weights)

            tm_full = TemperatureMatchingConfig(
                enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=1, token_block_size=4, offload_teacher=False
            )
            tm_acc = TemperatureMatchingConfig(
                enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=4, token_block_size=4, offload_teacher=False
            )
            (loss_full, m_full), grads_full = _accumulated_loss_and_grads_with_tm(
                student_state.params, batch, teacher_params, row_mask, mp, None, 1, tm_full
            )
            (loss_acc, m_acc), grads_acc = _accumulated_loss_and_grads_with_tm(
                student_state.params, batch, teacher_params, row_mask, mp, None, 4, tm_acc
            )
            np.testing.assert_allclose(loss_acc, loss_full, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(m_acc["train/tm/loss"], m_full["train/tm/loss"], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(m_acc["train/tm/active_tokens"], m_full["train/tm/active_tokens"], atol=1e-6)
            for ga, gf in zip(jax.tree.leaves(grads_acc), jax.tree.leaves(grads_full), strict=True):
                np.testing.assert_allclose(np.asarray(ga), np.asarray(gf), rtol=2e-4, atol=1e-5)


@pytest.mark.timeout(300)
def test_prepare_frozen_teacher_reloads_step0_checkpoint_on_resume_and_casts_bf16():
    """On a resumed run (step > 0), teacher reloads from initialize_from in BF16, not drifted student weights."""
    from fray.cluster import ResourceConfig
    from levanter.data.text.datasets import LmDataConfig
    from levanter.trainer import TrainerConfig

    with use_cpu_devices(1):
        config = _tiny_model_config()
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.adam(1e-3)
            mp = jmp.get_policy("compute=bfloat16,params=float32,output=bfloat16")
            step0_state = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(21), ema_beta=None)
            drifted_state = dataclasses.replace(
                initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(22), ema_beta=None),
                step=jax.sharding.reshard(jnp.array(5, dtype=jnp.int32), P()),
            )
            assert step0_state.params.stacked_blocks.stacked.shared is not None
            legacy = eqx.tree_at(
                lambda m: m.stacked_blocks.stacked.shared,
                step0_state.params,
                step0_state.params.stacked_blocks.stacked.shared[0],
            )
            with tempfile.TemporaryDirectory() as ckpt_dir:
                save_checkpoint({"params": legacy, "pending_qb_betas": step0_state.pending_qb_betas}, 0, ckpt_dir)
                trainer_cfg = GrugTrainerConfig(
                    trainer=TrainerConfig(mp=mp, initialize_from=ckpt_dir, use_explicit_mesh_axes=True),
                    special_token_ids=(20, 21),
                    anchor_token_ids=(2, 3),
                    gradient_accumulation_steps=4,
                    temperature_matching=TemperatureMatchingConfig(
                        enabled=True, weight=1.0 / 3.0, temperature=2.5, microbatch_count=2, offload_teacher=False
                    ),
                )
                run_cfg = GrugRunConfig(
                    model=config,
                    data=LmDataConfig(components={}),
                    resources=ResourceConfig.with_cpu(),
                    trainer=trainer_cfg,
                )
                expected_fp32 = initialize_snowball_sft(
                    initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(99), ema_beta=None),
                    ckpt_dir,
                    mesh,
                    trainer_cfg,
                ).params
                expected_bf16 = cast_teacher_params(expected_fp32, mp)

                teacher = prepare_frozen_teacher(drifted_state, config=run_cfg, mesh=mesh)
                # Every floating leaf in teacher must be BF16 and match step-0 checkpoint, not drifted student.
                for t_leaf, exp_leaf, d_leaf in zip(
                    jax.tree.leaves(teacher),
                    jax.tree.leaves(expected_bf16),
                    jax.tree.leaves(drifted_state.params),
                    strict=True,
                ):
                    if isinstance(t_leaf, jax.Array) and jnp.issubdtype(t_leaf.dtype, jnp.floating):
                        assert t_leaf.dtype == jnp.bfloat16
                        np.testing.assert_array_equal(np.asarray(t_leaf), np.asarray(exp_leaf))
                # Confirm drifted student actually differs from step-0 teacher.
                assert not np.array_equal(
                    np.asarray(drifted_state.params.token_embed.astype(jnp.bfloat16)),
                    np.asarray(teacher.token_embed),
                )


@pytest.mark.timeout(300)
def test_first_step_host_rebind_threads_teacher_and_row_mask(tmp_path):
    """Verify first_step_host_rebind compiles with (teacher_params, tm_row_mask) and preserves them across rebind."""

    class _RecordingTiming:
        def __init__(self):
            self.spans: list[str] = []

        def span(self, name: str, **_kwargs):
            self.spans.append(name)
            return nullcontext()

    (tmp_path / "memory.current").write_text("12345")
    (tmp_path / "memory.max").write_text("99999")

    with use_cpu_devices(2):
        config = _tiny_model_config()
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 2, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.adam(1e-3)
            mp = jmp.get_policy("float32")
            state_rebind = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(41), ema_beta=None)
            state_ref = initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(41), ema_beta=None)
            teacher_params = cast_teacher_params(
                initial_state(config, optimizer=optimizer, mp=mp, key=jax.random.key(42), ema_beta=None).params,
                mp,
            )
            teacher_before = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(teacher_params) if isinstance(leaf, jax.Array)]

            rng = np.random.default_rng(404)
            b_shard = NamedSharding(mesh, P("data", None))
            batch = GrugLmExample(
                jax.device_put(rng.integers(1, 22, size=(4, 8), dtype=np.int32), b_shard),
                jax.device_put(np.ones((4, 8), dtype=np.float32), b_shard),
                AttentionMask.causal(),
            )
            row_mask = jax.device_put(np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32), NamedSharding(mesh, P("data")))

            tm_cfg = TemperatureMatchingConfig(
                enabled=True,
                weight=1.0 / 3.0,
                temperature=2.5,
                microbatch_count=1,
                token_block_size=4,
                offload_teacher=False,
            )
            train_step = _make_train_step(
                optimizer, mp, z_loss_weight=0.0, ema_beta=None, accumulation_steps=2, temperature_matching=tm_cfg
            )
            timing = _RecordingTiming()
            original_state_leaves = [leaf for leaf in jax.tree.leaves(state_rebind) if isinstance(leaf, jax.Array)]

            rebound_state = _compile_first_step_and_rebind(
                train_step,
                state_rebind,
                batch,
                extra_args=(teacher_params, row_mask),
                compile_first_step=False,
                rebind_config=HostRebindConfig(cgroup_path=str(tmp_path)),
                timing=timing,
                step=0,
            )

            assert timing.spans == ["first_step_compile", "first_step_host_rebind"]
            assert all(leaf.is_deleted() for leaf in original_state_leaves)
            # teacher_params and row_mask must remain live and bit-identical after state deletion/restore.
            teacher_after = [leaf for leaf in jax.tree.leaves(teacher_params) if isinstance(leaf, jax.Array)]
            assert all(not leaf.is_deleted() for leaf in teacher_after)
            for before, after in zip(teacher_before, teacher_after, strict=True):
                np.testing.assert_array_equal(before, np.asarray(after))
            assert not row_mask.is_deleted()

            next_rebound, metrics_rebound, _ = train_step(rebound_state, batch, teacher_params, row_mask)
            next_ref, metrics_ref, _ = train_step(state_ref, batch, teacher_params, row_mask)
            assert int(next_rebound.step) == 1
            np.testing.assert_allclose(metrics_rebound["train/loss"], metrics_ref["train/loss"], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                metrics_rebound["train/tm/loss"], metrics_ref["train/tm/loss"], rtol=1e-6, atol=1e-6
            )
            for got, wanted in zip(
                jax.tree.leaves(next_rebound.params), jax.tree.leaves(next_ref.params), strict=True
            ):
                if isinstance(got, jax.Array):
                    np.testing.assert_allclose(np.asarray(got), np.asarray(wanted), rtol=1e-6, atol=1e-6)

