# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from jax.tree_util import register_dataclass
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.coupon_clipping.model import GrugModelConfig, Transformer
from experiments.grug.depth_growth import (
    DepthGrowthConfig,
    NewLayerInitialization,
    grow_grug_depth_state,
    load_and_grow_grug_depth_state,
    validate_depth_growth_data_offset,
)


class _Block(eqx.Module):
    weight: jax.Array
    router_bias: jax.Array


class _StackedBlocks(eqx.Module):
    stacked: _Block
    num_layers: int = eqx.field(static=True)


class _Model(eqx.Module):
    token_embed: jax.Array
    stacked_block_segments: tuple[_StackedBlocks, ...]


class _OptimizerState(eqx.Module):
    count: jax.Array
    token_embed_momentum: jax.Array
    stacked_block_segments: tuple[_StackedBlocks, ...]


class _WidthOptimizerState(eqx.Module):
    count: jax.Array
    parameter_buffer: jax.Array


@register_dataclass
@dataclass(frozen=True)
class _TrainState:
    step: jax.Array
    params: _Model
    opt_state: _OptimizerState
    ema_params: _Model | None
    pending_qb_betas: jax.Array


@register_dataclass
@dataclass(frozen=True)
class _TransformerTrainState:
    step: jax.Array
    params: Transformer
    opt_state: _WidthOptimizerState
    ema_params: Transformer | None
    pending_qb_betas: jax.Array


def _model(segment_lengths: tuple[int, ...], *, offset: int) -> _Model:
    num_layers = sum(segment_lengths)
    layer_values = jnp.arange(num_layers, dtype=jnp.float32) + offset
    segments = []
    first_layer = 0
    for segment_length in segment_lengths:
        segment_values = layer_values[first_layer : first_layer + segment_length]
        segments.append(
            _StackedBlocks(
                stacked=_Block(
                    weight=segment_values[:, None],
                    router_bias=jnp.stack((segment_values, segment_values + 0.5), axis=-1),
                ),
                num_layers=segment_length,
            )
        )
        first_layer += segment_length
    return _Model(
        token_embed=jnp.array([offset, offset + 1], dtype=jnp.float32),
        stacked_block_segments=tuple(segments),
    )


def _state(segment_lengths: tuple[int, ...], *, step: int, offset: int) -> _TrainState:
    num_layers = sum(segment_lengths)
    params = _model(segment_lengths, offset=offset)
    optimizer_segments = tuple(
        _StackedBlocks(
            stacked=_Block(
                weight=jnp.full((segment_length, 1), offset + 4, dtype=jnp.float32),
                router_bias=jnp.full((segment_length, 2), offset + 5, dtype=jnp.float32),
            ),
            num_layers=segment_length,
        )
        for segment_length in segment_lengths
    )
    return _TrainState(
        step=jnp.array(step, dtype=jnp.int32),
        params=params,
        opt_state=_OptimizerState(
            count=jnp.array(step, dtype=jnp.int32),
            token_embed_momentum=jnp.array([offset + 2, offset + 3], dtype=jnp.float32),
            stacked_block_segments=optimizer_segments,
        ),
        ema_params=_model(segment_lengths, offset=offset + 20),
        pending_qb_betas=jnp.arange(num_layers * 2, dtype=jnp.float32).reshape(num_layers, 2) + offset,
    )


def test_grow_grug_depth_state_repeats_blocks_and_preserves_resume_state():
    source = _state((2,), step=7, offset=10)
    fresh_target = _state((1, 2, 1), step=0, offset=0)
    config = DepthGrowthConfig(
        source_layers=2,
        target_layers=4,
        width_expansion_factor=1,
        new_layer_initialization=NewLayerInitialization.REPEAT,
        expected_step=7,
        expected_data_offset=112,
    )

    grown, report = grow_grug_depth_state(source, fresh_target, config)

    assert int(grown.step) == 7
    assert jnp.array_equal(grown.params.token_embed, source.params.token_embed)
    assert tuple(segment.num_layers for segment in grown.params.stacked_block_segments) == (1, 2, 1)
    grown_weights = jnp.concatenate([segment.stacked.weight for segment in grown.params.stacked_block_segments], axis=0)
    assert jnp.array_equal(
        grown_weights,
        jnp.tile(source.params.stacked_block_segments[0].stacked.weight, (2, 1)),
    )
    assert grown.ema_params is not None
    grown_ema_bias = jnp.concatenate(
        [segment.stacked.router_bias for segment in grown.ema_params.stacked_block_segments], axis=0
    )
    assert jnp.array_equal(
        grown_ema_bias,
        jnp.tile(source.ema_params.stacked_block_segments[0].stacked.router_bias, (2, 1)),
    )
    assert jnp.array_equal(grown.pending_qb_betas, jnp.tile(source.pending_qb_betas, (2, 1)))

    assert int(grown.opt_state.count) == 7
    assert jnp.array_equal(grown.opt_state.token_embed_momentum, source.opt_state.token_embed_momentum)
    for grown_segment, fresh_segment in zip(
        grown.opt_state.stacked_block_segments,
        fresh_target.opt_state.stacked_block_segments,
        strict=True,
    ):
        assert jnp.array_equal(grown_segment.stacked.weight, fresh_segment.stacked.weight)
    assert report.step == 7
    assert report.data_offset == 112
    assert report.reset_optimizer_leaves == 6
    assert report.preserved_optimizer_leaves == 2


def test_grow_grug_depth_state_rejects_wrong_transition_checkpoint():
    source = _state((2,), step=6, offset=10)
    fresh_target = _state((1, 2, 1), step=0, offset=0)
    config = DepthGrowthConfig(
        source_layers=2,
        target_layers=4,
        width_expansion_factor=1,
        new_layer_initialization=NewLayerInitialization.REPEAT,
        expected_step=7,
        expected_data_offset=112,
    )

    with pytest.raises(ValueError, match="source checkpoint is at step 6, expected 7"):
        grow_grug_depth_state(source, fresh_target, config)


def test_width_only_growth_allows_equal_source_and_target_depth():
    source = _state((2,), step=7, offset=10)
    fresh_target = _state((1, 1), step=0, offset=0)
    config = DepthGrowthConfig(
        source_layers=2,
        target_layers=2,
        width_expansion_factor=2,
        new_layer_initialization=NewLayerInitialization.IDENTITY_PREFIX,
        expected_step=7,
        expected_data_offset=112,
    )

    grown, report = grow_grug_depth_state(source, fresh_target, config)

    grown_weights = jnp.concatenate([segment.stacked.weight for segment in grown.params.stacked_block_segments])
    assert jnp.array_equal(grown_weights, source.params.stacked_block_segments[0].stacked.weight)
    assert report.step == 7

    with pytest.raises(ValueError, match="must increase the layer count, width, or both"):
        DepthGrowthConfig(
            source_layers=48,
            target_layers=48,
            width_expansion_factor=1,
            new_layer_initialization=NewLayerInitialization.IDENTITY_PREFIX,
            expected_step=5120,
            expected_data_offset=5120 * 256,
        )


def test_load_and_grow_grug_depth_state_resolves_latest_checkpoint():
    source = _state((1,), step=7, offset=10)
    fresh_target = _state((4,), step=0, offset=0)
    config = DepthGrowthConfig(
        source_layers=1,
        target_layers=4,
        width_expansion_factor=1,
        new_layer_initialization=NewLayerInitialization.REPEAT,
        expected_step=7,
        expected_data_offset=112,
    )
    loaded_paths = []

    def fake_latest_checkpoint(path: str) -> str:
        assert path == "s3://example/checkpoints"
        return f"{path}/step-7"

    def fake_load(exemplar, path: str, **_kwargs):
        assert exemplar is source
        loaded_paths.append(path)
        return source

    grown, report = load_and_grow_grug_depth_state(
        source,
        fresh_target,
        "s3://example/checkpoints",
        config=config,
        mesh=None,
        _load_fn=fake_load,
        _latest_checkpoint_fn=fake_latest_checkpoint,
    )

    assert loaded_paths == ["s3://example/checkpoints/step-7"]
    assert int(grown.step) == 7
    assert report.step == 7


def test_grow_grug_depth_state_wraps_twelve_layers_across_production_segments():
    source = _state((5, 7), step=40, offset=10)
    fresh_target = _state((4, 18, 4, 22), step=0, offset=0)
    config = DepthGrowthConfig(
        source_layers=12,
        target_layers=48,
        width_expansion_factor=1,
        new_layer_initialization=NewLayerInitialization.REPEAT,
        expected_step=40,
        expected_data_offset=640,
    )

    grown, _ = grow_grug_depth_state(source, fresh_target, config)

    source_weights = jnp.concatenate(
        [segment.stacked.weight for segment in source.params.stacked_block_segments], axis=0
    )
    grown_weights = jnp.concatenate([segment.stacked.weight for segment in grown.params.stacked_block_segments], axis=0)
    assert tuple(segment.num_layers for segment in grown.params.stacked_block_segments) == (4, 18, 4, 22)
    assert jnp.array_equal(grown_weights, jnp.tile(source_weights, (4, 1)))


def test_grow_grug_depth_state_rejects_mismatched_optimizer_schedule_count():
    source = _state((1,), step=7, offset=10)
    source = eqx.tree_at(lambda state: state.opt_state.count, source, jnp.array(6, dtype=jnp.int32))
    fresh_target = _state((4,), step=0, offset=0)
    config = DepthGrowthConfig(
        source_layers=1,
        target_layers=4,
        width_expansion_factor=1,
        new_layer_initialization=NewLayerInitialization.REPEAT,
        expected_step=7,
        expected_data_offset=112,
    )

    with pytest.raises(ValueError, match="optimizer schedule is at step 6, expected 7"):
        grow_grug_depth_state(source, fresh_target, config)


def test_validate_depth_growth_data_offset_rejects_changed_batch_schedule():
    config = DepthGrowthConfig(
        source_layers=1,
        target_layers=48,
        width_expansion_factor=1,
        new_layer_initialization=NewLayerInitialization.REPEAT,
        expected_step=2_240,
        expected_data_offset=2_293_760,
    )

    with pytest.raises(ValueError, match="target batch schedule changes the depth-growth data cursor"):
        validate_depth_growth_data_offset(config, actual_data_offset=2_293_761)


def test_grow_grug_width_and_depth_preserves_source_function_with_identity_layers():
    common = {
        "vocab_size": 32,
        "num_experts": 2,
        "num_experts_per_token": 1,
        "head_dim": 4,
        "max_seq_len": 4,
        "sliding_window": 4,
        "block_storage": "array_stacked",
    }
    source_config = GrugModelConfig(
        **common,
        hidden_dim=4,
        intermediate_dim=2,
        shared_expert_intermediate_dim=2,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        block_segment_lengths=(1,),
        block_segment_shared_expert_intermediate_dims=(2,),
    )
    target_config = GrugModelConfig(
        **common,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        block_segment_lengths=(1, 1),
        block_segment_shared_expert_intermediate_dims=(4, 4),
    )
    config = DepthGrowthConfig(
        source_layers=1,
        target_layers=2,
        width_expansion_factor=2,
        new_layer_initialization=NewLayerInitialization.IDENTITY_PREFIX,
        expected_step=7,
        expected_data_offset=112,
    )

    with set_mesh(compact_grug_mesh()):
        source_model = Transformer.init(source_config, key=jax.random.PRNGKey(0))
        fresh_target_model = Transformer.init(target_config, key=jax.random.PRNGKey(1))
        source = _TransformerTrainState(
            step=jnp.array(7, dtype=jnp.int32),
            params=source_model,
            opt_state=_WidthOptimizerState(
                count=jnp.array(7, dtype=jnp.int32),
                parameter_buffer=jnp.arange(4, dtype=jnp.float32),
            ),
            ema_params=None,
            pending_qb_betas=jnp.zeros((1, 2), dtype=jnp.float32),
        )
        fresh_target = _TransformerTrainState(
            step=jnp.array(0, dtype=jnp.int32),
            params=fresh_target_model,
            opt_state=_WidthOptimizerState(
                count=jnp.array(0, dtype=jnp.int32),
                parameter_buffer=jnp.full((8,), 5, dtype=jnp.float32),
            ),
            ema_params=None,
            pending_qb_betas=jnp.ones((2, 2), dtype=jnp.float32),
        )

        grown, report = grow_grug_depth_state(source, fresh_target, config)

    source_embedding = source.params.token_embed[3]
    grown_embedding = grown.params.token_embed[3]
    assert jnp.array_equal(grown_embedding, jnp.tile(source_embedding, 2))
    assert jnp.allclose(grown_embedding @ grown.params.output_proj, source_embedding @ source.params.output_proj)

    assert source.params.stacked_block_segments is not None
    assert grown.params.stacked_block_segments is not None
    source_block = source.params.stacked_block_segments[0].stacked
    grown_source_block = grown.params.stacked_block_segments[1].stacked
    source_hidden = jnp.arange(4, dtype=jnp.float32)
    grown_hidden = jnp.tile(source_hidden, 2)
    source_query = source_hidden @ source_block.attn.w_q[0]
    grown_query = grown_hidden @ grown_source_block.attn.w_q[0]
    assert np.allclose(np.asarray(grown_query), np.tile(np.asarray(source_query), 2))
    source_expert_gate = source_hidden @ source_block.mlp.expert_mlp.w_gate[0, 0]
    grown_expert_gate = grown_hidden @ grown_source_block.mlp.expert_mlp.w_gate[0, 0]
    assert np.allclose(np.asarray(grown_expert_gate), np.tile(np.asarray(source_expert_gate), 2))

    assert jnp.array_equal(grown.opt_state.parameter_buffer, fresh_target.opt_state.parameter_buffer)
    assert report.reset_optimizer_leaves > 0
    identity_block = grown.params.stacked_block_segments[0].stacked
    assert jnp.count_nonzero(identity_block.attn.w_o) == 0
    assert jnp.count_nonzero(identity_block.mlp.expert_mlp.w_down) == 0
    assert identity_block.shared is not None
    assert jnp.count_nonzero(identity_block.shared.w_down) == 0


def test_factor_four_growth_preserves_mlp_with_fixed_intermediate_width():
    common = {
        "vocab_size": 32,
        "intermediate_dim": 4,
        "shared_expert_intermediate_dim": 4,
        "num_experts": 2,
        "num_experts_per_token": 1,
        "head_dim": 4,
        "max_seq_len": 4,
        "sliding_window": 4,
        "block_storage": "array_stacked",
    }
    source_config = GrugModelConfig(
        **common,
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        block_segment_lengths=(1,),
        block_segment_shared_expert_intermediate_dims=(4,),
    )
    target_config = GrugModelConfig(
        **common,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        block_segment_lengths=(1, 1),
        block_segment_shared_expert_intermediate_dims=(4, 4),
    )
    config = DepthGrowthConfig(
        source_layers=1,
        target_layers=2,
        width_expansion_factor=4,
        new_layer_initialization=NewLayerInitialization.IDENTITY_PREFIX,
        expected_step=7,
        expected_data_offset=112,
    )

    with set_mesh(compact_grug_mesh()):
        source_model = Transformer.init(source_config, key=jax.random.PRNGKey(0))
        fresh_target_model = Transformer.init(target_config, key=jax.random.PRNGKey(1))
        source = _TransformerTrainState(
            step=jnp.array(7, dtype=jnp.int32),
            params=source_model,
            opt_state=_WidthOptimizerState(
                count=jnp.array(7, dtype=jnp.int32),
                parameter_buffer=jnp.arange(4, dtype=jnp.float32),
            ),
            ema_params=None,
            pending_qb_betas=jnp.zeros((1, 2), dtype=jnp.float32),
        )
        fresh_target = _TransformerTrainState(
            step=jnp.array(0, dtype=jnp.int32),
            params=fresh_target_model,
            opt_state=_WidthOptimizerState(
                count=jnp.array(0, dtype=jnp.int32),
                parameter_buffer=jnp.full((16,), 5, dtype=jnp.float32),
            ),
            ema_params=None,
            pending_qb_betas=jnp.ones((2, 2), dtype=jnp.float32),
        )

        grown, _ = grow_grug_depth_state(source, fresh_target, config)

    assert source.params.stacked_block_segments is not None
    assert grown.params.stacked_block_segments is not None
    source_block = source.params.stacked_block_segments[0].stacked
    grown_source_block = grown.params.stacked_block_segments[1].stacked
    source_hidden = jnp.arange(4, dtype=jnp.float32)
    grown_hidden = jnp.tile(source_hidden, 4)

    source_gate = source_hidden @ source_block.mlp.expert_mlp.w_gate[0, 0]
    grown_gate = grown_hidden @ grown_source_block.mlp.expert_mlp.w_gate[0, 0]
    source_up = source_hidden @ source_block.mlp.expert_mlp.w_up[0, 0]
    grown_up = grown_hidden @ grown_source_block.mlp.expert_mlp.w_up[0, 0]
    assert np.allclose(np.asarray(grown_gate), np.asarray(source_gate))
    assert np.allclose(np.asarray(grown_up), np.asarray(source_up))

    source_gate_array = np.asarray(source_gate)
    grown_gate_array = np.asarray(grown_gate)
    source_activation = source_gate_array / (1 + np.exp(-source_gate_array))
    grown_activation = grown_gate_array / (1 + np.exp(-grown_gate_array))
    source_output = (source_activation * np.asarray(source_up)) @ np.asarray(source_block.mlp.expert_mlp.w_down[0, 0])
    grown_output = (grown_activation * np.asarray(grown_up)) @ np.asarray(grown_source_block.mlp.expert_mlp.w_down[0, 0])
    assert np.allclose(grown_output, np.tile(source_output, 4))

    source_embedding = source.params.token_embed[3]
    grown_embedding = grown.params.token_embed[3]
    assert jnp.allclose(grown_embedding @ grown.params.output_proj, source_embedding @ source.params.output_proj)
