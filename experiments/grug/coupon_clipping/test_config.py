# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.json_logger import JsonLoggerConfig
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping import (
    launch,
    paloma_c0,
    paloma_c_short,
    paloma_l4,
    paloma_random_layer_dropout,
    paloma_wd1,
    paloma_wd2,
)
from experiments.grug.coupon_clipping.config import (
    AGGRESSIVE_DECAY_STEPS,
    AGGRESSIVE_GROWTH_CONFIG,
    AGGRESSIVE_TRANSITION_STEP,
    DECAY_STEPS,
    EXTREME_DECAY_STEPS,
    EXTREME_GROWTH_CONFIG,
    EXTREME_TRANSITION_STEP,
    L4_DECAY_STEPS,
    L4_GROWTH_CONFIG,
    L4_TRANSITION_STEP,
    RANDOM_LAYER_DROPOUT_COUNT,
    RANDOM_LAYER_DROPOUT_GROWTH_CONFIG,
    SEGMENT_LENGTHS,
    SELECTED_LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    TRAIN_STEPS,
    CouponClippingArm,
    CouponClippingLearningRate,
    build_model_config,
    build_optimizer_config,
    model_accounting,
)
from experiments.grug.coupon_clipping.depth_launch import (
    PILOT_GROWN_STEPS,
    PILOT_SOURCE_STEPS,
    build_aggressive_growth_pilot_checkpoint,
    build_aggressive_source_model_config,
    build_d1_checkpoint,
    build_depth_source_model_config,
    build_extreme_checkpoint,
    build_extreme_growth_pilot_checkpoint,
    build_extreme_source_model_config,
    build_extreme_target_model_config,
    build_growth_pilot_checkpoint,
    build_growth_target_only_checkpoint,
    build_l4_checkpoint,
    build_l4_growth_pilot_checkpoint,
    build_l4_source_model_config,
    build_l4_source_pilot_checkpoint,
    build_random_layer_dropout_checkpoint,
    build_random_layer_dropout_growth_pilot_checkpoint,
    build_random_layer_dropout_source_model_config,
    build_random_layer_dropout_source_pilot_checkpoint,
)
from experiments.grug.coupon_clipping.eval_launch import build_paloma_eval
from experiments.grug.coupon_clipping.model import GrugModelConfig, Transformer
from experiments.grug.coupon_clipping.train import (
    _freeze_inactive_layer_updates,
    _random_active_layer_indices,
    _updated_qb_betas,
)
from experiments.grug.depth_growth import DepthGrowthConfig, NewLayerInitialization
from experiments.marin_tokenizer import marin_tokenizer


def _test_data_config():
    model = build_model_config(CouponClippingArm.C0_P0)
    return launch.datakit_data_config(
        total_steps=TRAIN_STEPS,
        batch_size=TRAIN_BATCH_SIZE,
        max_seq_len=model.max_seq_len,
        enable_simulated_epoching=False,
        val_components={},
    )


def test_pyramid_arms_match_parameters_flops_and_scan_boundaries():
    configs = {arm: build_model_config(arm) for arm in CouponClippingArm}
    accounting = {arm: model_accounting(config) for arm, config in configs.items()}

    assert len(set(accounting.values())) == 1
    assert accounting[CouponClippingArm.C0_P0].stored_parameters == 46_063_592_448
    assert accounting[CouponClippingArm.C0_P0].active_parameters == 5_294_957_568
    assert {config.resolved_block_segment_lengths for config in configs.values()} == {SEGMENT_LENGTHS}

    assert configs[CouponClippingArm.P1].shared_expert_intermediate_dims_by_layer[:4] == (4096,) * 4
    assert configs[CouponClippingArm.P1].shared_expert_intermediate_dims_by_layer[4:] == (1024,) * 44
    assert configs[CouponClippingArm.P2].shared_expert_intermediate_dims_by_layer[22:26] == (4096,) * 4


def test_segment_widths_must_preserve_shared_parameter_budget():
    control = build_model_config(CouponClippingArm.C0_P0)

    with pytest.raises(ValueError, match="preserve the configured per-layer average"):
        dataclasses.replace(control, block_segment_shared_expert_intermediate_dims=(4096, 1024, 1024, 1025))


def test_learning_rate_gate_uses_three_fixed_candidates():
    candidates = {
        recipe: (build_optimizer_config(recipe).learning_rate, build_optimizer_config(recipe).adam_lr)
        for recipe in CouponClippingLearningRate
    }

    assert candidates == {
        CouponClippingLearningRate.LOW: (0.005768679, 0.001331234),
        CouponClippingLearningRate.CENTER: (0.006423539, 0.001482355),
        CouponClippingLearningRate.HIGH: (0.007210848, 0.001664041),
    }
    assert SELECTED_LEARNING_RATE is CouponClippingLearningRate.HIGH
    assert build_optimizer_config().learning_rate == candidates[SELECTED_LEARNING_RATE][0]


def test_pilot_keeps_production_mesh_allocator_and_optimizer_horizon(monkeypatch, tmp_path):
    dispatched = []
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    monkeypatch.setattr(launch, "run_grug", dispatched.append)
    config = launch.CouponClippingLaunchConfig(
        model=build_model_config(CouponClippingArm.C0_P0),
        data=_test_data_config(),
        output_path=str(tmp_path),
        run_id="cc16-c0-p0-pilot128",
        resources=launch._TRAIN_RESOURCES,
        tracker=JsonLoggerConfig(logger_name="test.coupon_clipping"),
        steps=launch.PILOT_STEPS,
        watch_interval=8,
    )

    launch.run_coupon_clipping_trial(config)

    assert launch.os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"
    assert len(dispatched) == 1
    run_config = dispatched[0]
    assert run_config.trainer.expert_axis_size == 1
    assert run_config.trainer.replica_axis_size == 1
    assert run_config.trainer.trainer.num_train_steps == 128
    assert run_config.trainer.trainer.watch.watch_targets == ["grads"]
    assert run_config.trainer.trainer.watch.interval == 8
    assert run_config.optimizer_num_train_steps == TRAIN_STEPS
    assert run_config.optimizer.decay == DECAY_STEPS
    assert DECAY_STEPS < run_config.optimizer_num_train_steps


def test_depth_launch_propagates_transition_contract(monkeypatch, tmp_path):
    dispatched = []
    monkeypatch.setattr(launch, "run_grug", dispatched.append)
    growth = DepthGrowthConfig(
        source_layers=1,
        target_layers=48,
        width_expansion_factor=1,
        new_layer_initialization=NewLayerInitialization.REPEAT,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * 256,
    )
    config = launch.CouponClippingLaunchConfig(
        model=build_model_config(CouponClippingArm.C0_P0),
        data=_test_data_config(),
        output_path=str(tmp_path),
        run_id="cc16-growth-pilot-l1-to-l48-16",
        resources=launch._TRAIN_RESOURCES,
        tracker=JsonLoggerConfig(logger_name="test.coupon_clipping"),
        steps=PILOT_SOURCE_STEPS + PILOT_GROWN_STEPS,
        optimizer_decay_steps=320,
        initialize_from="s3://example/source/checkpoints",
        depth_growth=growth,
    )

    launch.run_coupon_clipping_trial(config)

    run_config = dispatched[0]
    assert run_config.trainer.trainer.initialize_from == "s3://example/source/checkpoints"
    assert run_config.depth_growth == growth
    assert run_config.optimizer_num_train_steps == TRAIN_STEPS
    assert run_config.optimizer.decay == 320


def test_depth_artifacts_chain_source_before_growth():
    source_model = build_depth_source_model_config()
    assert source_model.num_layers == 1
    assert source_model.resolved_block_segment_lengths == (1,)

    pilot = build_growth_pilot_checkpoint(version="test-dev")
    d1 = build_d1_checkpoint(version="test-dev")
    aggressive = build_aggressive_growth_pilot_checkpoint(version="test-dev")
    extreme = build_extreme_growth_pilot_checkpoint(version="test-dev")
    extreme_full = build_extreme_checkpoint(version="test-dev")

    assert len(pilot.deps) == 1
    assert len(d1.deps) == 1
    assert len(aggressive.deps) == 1
    assert len(extreme.deps) == 1
    assert len(extreme_full.deps) == 1

    target_only = build_growth_target_only_checkpoint(
        source_checkpoint_root="s3://example/source/checkpoints",
        version="test-dev",
    )
    assert target_only.deps == ()


def test_aggressive_source_attacks_fixed_compute_and_preserves_target_contract():
    target = build_model_config(CouponClippingArm.C0_P0)
    source = build_aggressive_source_model_config()
    target_accounting = model_accounting(target)
    source_accounting = model_accounting(source)

    assert (source.hidden_dim, source.num_layers, source.num_heads, source.num_kv_heads) == (1536, 1, 12, 3)
    assert source.inferred_head_dim == target.inferred_head_dim == 128
    assert target_accounting.active_parameters == 5_294_957_568
    assert source_accounting.active_parameters == 418_701_376
    assert target_accounting.forward_flops_per_token / source_accounting.forward_flops_per_token > 25
    assert AGGRESSIVE_GROWTH_CONFIG.width_expansion_factor == 2
    assert AGGRESSIVE_GROWTH_CONFIG.new_layer_initialization is NewLayerInitialization.IDENTITY_PREFIX
    assert AGGRESSIVE_TRANSITION_STEP == 6080
    assert AGGRESSIVE_DECAY_STEPS == 320
    assert build_optimizer_config(decay_steps=AGGRESSIVE_DECAY_STEPS).decay == AGGRESSIVE_DECAY_STEPS


def test_extreme_source_targets_tenfold_speed_without_more_experts():
    control = build_model_config(CouponClippingArm.C0_P0)
    target = build_extreme_target_model_config()
    aggressive = build_aggressive_source_model_config()
    extreme = build_extreme_source_model_config()
    accounting = model_accounting(extreme)

    assert (extreme.hidden_dim, extreme.num_layers, extreme.num_heads, extreme.num_kv_heads) == (768, 1, 6, 2)
    assert extreme.num_experts == target.num_experts == control.num_experts == 64
    assert extreme.num_experts_per_token == target.num_experts_per_token == control.num_experts_per_token == 4
    assert extreme.intermediate_dim == 1536
    assert extreme.intermediate_dim > aggressive.intermediate_dim
    assert extreme.shared_expert_intermediate_dim == target.shared_expert_intermediate_dim == 1536
    assert (target.hidden_dim, target.num_layers, target.num_heads, target.num_kv_heads) == (3072, 48, 24, 8)
    assert target.intermediate_dim == extreme.intermediate_dim
    assert target.hidden_dim == 4 * extreme.hidden_dim
    assert target.num_heads == 4 * extreme.num_heads
    assert target.num_kv_heads == 4 * extreme.num_kv_heads
    assert accounting.active_parameters < 250_000_000
    assert accounting.stored_parameters < 450_000_000
    assert model_accounting(target).forward_flops_per_token / accounting.forward_flops_per_token > 40
    assert EXTREME_GROWTH_CONFIG.width_expansion_factor == 4
    assert EXTREME_GROWTH_CONFIG.new_layer_initialization is NewLayerInitialization.IDENTITY_PREFIX
    assert EXTREME_TRANSITION_STEP == 5760
    assert EXTREME_DECAY_STEPS == 640
    assert build_optimizer_config(decay_steps=EXTREME_DECAY_STEPS).decay == EXTREME_DECAY_STEPS


def test_l4_arms_hold_width_tail_and_layer_budget_constant():
    target = build_model_config(CouponClippingArm.C0_P0)
    physical_l4 = build_l4_source_model_config()
    random_layer_dropout = build_random_layer_dropout_source_model_config()

    assert (physical_l4.hidden_dim, physical_l4.num_layers) == (1536, 4)
    assert physical_l4.resolved_block_segment_lengths == (4,)
    assert (random_layer_dropout.hidden_dim, random_layer_dropout.num_layers) == (1536, 48)
    assert random_layer_dropout.resolved_block_segment_lengths == (48,)
    assert RANDOM_LAYER_DROPOUT_COUNT == physical_l4.num_layers
    assert L4_GROWTH_CONFIG.source_layers == 4
    assert RANDOM_LAYER_DROPOUT_GROWTH_CONFIG.source_layers == target.num_layers
    assert L4_GROWTH_CONFIG.target_layers == RANDOM_LAYER_DROPOUT_GROWTH_CONFIG.target_layers == target.num_layers
    assert L4_TRANSITION_STEP == 5120
    assert L4_DECAY_STEPS == 1280

    physical_pipeline = build_l4_checkpoint(version="test-dev")
    dropout_pipeline = build_random_layer_dropout_checkpoint(version="test-dev")
    physical_growth_pilot = build_l4_growth_pilot_checkpoint(version="test-dev")
    dropout_growth_pilot = build_random_layer_dropout_growth_pilot_checkpoint(version="test-dev")
    assert len(physical_pipeline.deps) == len(dropout_pipeline.deps) == 1
    assert len(physical_growth_pilot.deps) == len(dropout_growth_pilot.deps) == 1
    assert len(build_l4_source_pilot_checkpoint(version="test-dev").deps) == 0
    assert len(build_random_layer_dropout_source_pilot_checkpoint(version="test-dev").deps) == 0


def test_random_layer_dropout_selection_is_sorted_unique_and_updates_only_active_qb_layers():
    indices = _random_active_layer_indices(
        jnp.array(17, dtype=jnp.int32),
        num_layers=48,
        active_layer_count=4,
        seed=0,
    )
    repeated_indices = _random_active_layer_indices(
        jnp.array(17, dtype=jnp.int32),
        num_layers=48,
        active_layer_count=4,
        seed=0,
    )
    assert jnp.array_equal(indices, repeated_indices)
    assert indices.shape == (4,)
    assert jnp.all(indices[:-1] < indices[1:])

    previous = jnp.arange(48 * 2, dtype=jnp.float32).reshape(48, 2)
    current = jnp.full((4, 2), -1, dtype=jnp.float32)
    updated = _updated_qb_betas(previous, current, indices)
    assert jnp.array_equal(updated[indices], current)
    assert jnp.count_nonzero(jnp.all(updated == previous, axis=1)) == 44


def test_random_layer_dropout_freezes_inactive_parameters_and_optimizer_state():
    updates = {
        "stacked_block_segments": ({"stacked": jnp.full((4, 2), 3.0)},),
        "output_proj": jnp.full((2, 2), 5.0),
    }
    previous_opt_state = {
        "momentum": {
            "stacked_block_segments": ({"stacked": jnp.arange(8, dtype=jnp.float32).reshape(4, 2)},),
            "output_proj": jnp.full((2, 2), 7.0),
        }
    }
    next_opt_state = jax.tree.map(lambda value: value + 100, previous_opt_state)

    frozen_updates, frozen_opt_state = _freeze_inactive_layer_updates(
        updates,
        previous_opt_state,
        next_opt_state,
        active_layer_indices=jnp.array([1, 3], dtype=jnp.int32),
        num_layers=4,
    )

    assert jnp.array_equal(
        frozen_updates["stacked_block_segments"][0]["stacked"],
        jnp.array([[0, 0], [3, 3], [0, 0], [3, 3]], dtype=jnp.float32),
    )
    assert jnp.array_equal(frozen_updates["output_proj"], updates["output_proj"])
    frozen_momentum = frozen_opt_state["momentum"]["stacked_block_segments"][0]["stacked"]
    previous_momentum = previous_opt_state["momentum"]["stacked_block_segments"][0]["stacked"]
    inactive_indices = jnp.array([0, 2])
    active_indices = jnp.array([1, 3])
    assert jnp.array_equal(frozen_momentum[inactive_indices], previous_momentum[inactive_indices])
    assert jnp.array_equal(frozen_momentum[active_indices], previous_momentum[active_indices] + 100)
    assert jnp.array_equal(
        frozen_opt_state["momentum"]["output_proj"],
        next_opt_state["momentum"]["output_proj"],
    )


def test_array_stacked_active_layer_indices_match_full_order_when_all_selected():
    model_config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        block_storage="array_stacked",
        block_segment_lengths=(4,),
        block_segment_shared_expert_intermediate_dims=(8,),
        moe_implementation="ring",
    )
    tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)

    with set_mesh(compact_grug_mesh()):
        model = Transformer.init(model_config, key=jax.random.PRNGKey(0))
        full_hidden, full_metrics = model(tokens)
        selected_hidden, selected_metrics = model(
            tokens,
            active_layer_indices=jnp.arange(4, dtype=jnp.int32),
        )

    assert jnp.array_equal(selected_hidden, full_hidden)
    assert jnp.array_equal(selected_metrics["qb_beta_per_layer"], full_metrics["qb_beta_per_layer"])


def test_paloma_eval_is_checkpoint_only_and_bounded(tmp_path):
    checkpoint = ArtifactStep.adopt(
        "tests/coupon-checkpoint",
        "test-dev",
        "s3://example/coupon-checkpoint",
        kind=LevanterCheckpoint,
    )
    evaluation = build_paloma_eval(
        checkpoint,
        label="test",
        version="test-dev",
        eval_batch_size=32,
        max_eval_batches=3,
    )

    for dependency in evaluation.deps[1:]:
        dependency_path = dependency.path(str(tmp_path))
        write_record(
            ArtifactRecord(
                name=dependency.name,
                version=dependency.version,
                output_path=dependency_path,
                result_type=f"{TokenizedCache.__module__}.{TokenizedCache.__qualname__}",
                config={"tokenizer": marin_tokenizer, "format": {"text_key": "text"}},
            )
        )

    config = materialized_config(evaluation, str(tmp_path))

    assert evaluation.deps[0] is checkpoint
    assert len(evaluation.deps) == 17
    assert config.checkpoint_path == "s3://example/coupon-checkpoint/checkpoints"
    assert config.eval_batch_size == 32
    assert config.max_eval_batches == 3
    assert len(config.data.train_weights) == 16
    assert set(config.data.train_weights.values()) == {0.0}


@pytest.mark.parametrize(
    ("builder", "checkpoint_source"),
    [
        (paloma_wd1.build, "users/power/grug/coupon-clipping/ccx-wd1-d1536-l1-to-d3072-l48/dev"),
        (paloma_wd2.build, "users/power/grug/coupon-clipping/ccx-wd2-d768-l1-to-d3072-l48-tail640/dev"),
        (paloma_c_short.build, "users/power/grug/coupon-clipping/ccx-c-short-l48-step3200/dev"),
        (paloma_c0.build, "users/power/grug/coupon-clipping/cc16-c0-p0/dev"),
        (paloma_l4.build, "users/power/grug/coupon-clipping/ccx-l4-d1536-l4-to-d3072-l48-tail1280/dev"),
        (
            paloma_random_layer_dropout.build,
            "users/power/grug/coupon-clipping/ccx-ld4-d1536-l48-sample4-to-d3072-l48-tail1280/dev",
        ),
    ],
)
def test_paloma_entrypoint_adopts_completed_checkpoint_without_training_dependencies(builder, checkpoint_source):
    evaluation = builder(version="test-dev")
    checkpoint = evaluation.deps[0]

    assert checkpoint.adopt_source == checkpoint_source
    assert checkpoint.artifact_type is LevanterCheckpoint
    assert checkpoint.deps == ()


@pytest.mark.parametrize(("eval_batch_size", "max_eval_batches"), [(0, 1), (1, 0)])
def test_paloma_eval_rejects_unbounded_empty_work(eval_batch_size: int, max_eval_batches: int):
    checkpoint = ArtifactStep.adopt(
        "tests/coupon-checkpoint",
        "test-dev",
        "s3://example/coupon-checkpoint",
        kind=LevanterCheckpoint,
    )

    with pytest.raises(ValueError, match="must be positive"):
        build_paloma_eval(
            checkpoint,
            label="test",
            version="test-dev",
            eval_batch_size=eval_batch_size,
            max_eval_batches=max_eval_batches,
        )


def test_coupon_optimizer_routes_segmented_model_parameters_to_intended_groups():
    model_config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=8,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=3,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        block_storage="array_stacked",
        block_segment_lengths=(1, 2),
        block_segment_shared_expert_intermediate_dims=(8, 8),
    )
    with set_mesh(compact_grug_mesh()):
        params = Transformer.init(model_config, key=jax.random.PRNGKey(0))
        mask = build_optimizer_config().create_mask(params)

    assert params.stacked_block_segments is not None
    assert mask.stacked_block_segments is not None
    params_segment = params.stacked_block_segments[1].stacked
    mask_segment = mask.stacked_block_segments[1].stacked
    assert params_segment.mlp.expert_mlp.w_gate.ndim == 4
    assert mask_segment.mlp.expert_mlp.w_gate == "muonh"
    assert params_segment.shared is not None
    assert mask_segment.shared is not None
    assert mask_segment.shared.w_gate == "muonh"
    assert mask_segment.attn.w_q == "muonh"
    assert mask_segment.attn_gated_norm.w_down == "muonh"

    assert params_segment.rms_attn.weight.ndim == 2
    assert mask_segment.rms_attn.weight == "adam"
    assert mask_segment.mlp.router == "adam"
    assert mask_segment.mlp.router_bias == "adam"
    assert mask_segment.attn.attn_gate == "adam"
    assert mask.token_embed == "adam"
    assert mask.embed_norm.weight == "adam"
    assert mask.output_proj == "adamh"
