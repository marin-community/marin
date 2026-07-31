# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import pytest
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.json_logger import JsonLoggerConfig
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping import launch
from experiments.grug.coupon_clipping.config import (
    AGGRESSIVE_DECAY_STEPS,
    AGGRESSIVE_GROWTH_CONFIG,
    AGGRESSIVE_TRANSITION_STEP,
    DECAY_STEPS,
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
    build_growth_pilot_checkpoint,
    build_growth_target_only_checkpoint,
)
from experiments.grug.coupon_clipping.eval_launch import build_paloma_eval
from experiments.grug.coupon_clipping.model import GrugModelConfig, Transformer
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

    assert len(pilot.deps) == 1
    assert len(d1.deps) == 1
    assert len(aggressive.deps) == 1

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
