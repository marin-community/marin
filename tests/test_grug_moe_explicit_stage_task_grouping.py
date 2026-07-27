# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import subprocess
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from fray.cluster import ResourceConfig
from jax.extend import core as jax_core
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import (
    CHECKPOINT_ROUTER_COMBINE_WEIGHTS,
    CHECKPOINT_ROUTER_SELECTED_EXPERTS,
    CHECKPOINT_ROUTER_SIGMOID_WEIGHTS,
    CHECKPOINT_ROUTER_UNBIASED_TOPK,
    CHECKPOINT_ROUTER_WEIGHT_DENOMINATOR,
    MOE_REMAT_SAVE_NAMES,
)

from experiments.grug.moe import launch_cw_jaxpp_may_d2560
from experiments.grug.moe.check_jaxpp_group2_component_mpmd_parity import (
    component_structure,
    explicit_component_task_structure,
    joined_explicit_routing_finish_boundary_forward,
    joined_explicit_routing_finish_boundary_value_and_grads,
    joined_moe_finish_boundary_forward,
    joined_moe_finish_boundary_value_and_grads,
    paired_complete_checkpoint_boundary_forward,
    paired_complete_checkpoint_boundary_value_and_grads,
    paired_distinct_mlp_master_router_gradients,
    paired_full_block_boundary_value_and_grads,
    paired_full_block_per_microbatch_router_gradients,
    paired_moe_joint_checkpoint_value_and_grads,
    paired_moe_no_checkpoint_value_and_grads,
    paired_moe_per_checkpoint_value_and_grads,
    paired_pre_moe_checkpoint_boundary_forward,
    paired_pre_moe_checkpoint_boundary_value_and_grads,
    single_explicit_router_pre_boundary_forward,
    single_explicit_router_pre_boundary_primal_and_value_and_grads,
    single_full_block_allow_cse_boundary_value_and_grads,
    single_full_block_boundary_forward,
    single_full_block_boundary_value_and_grads,
    single_full_block_no_checkpoint_boundary_value_and_grads,
    single_moe_finish_boundary_value_and_grads,
    single_moe_value_and_grads,
    single_pre_moe_barrier_checkpoint_boundary_forward,
    single_pre_moe_barrier_checkpoint_boundary_primal_and_value_and_grads,
    single_pre_moe_barrier_checkpoint_boundary_value_and_grads,
    single_pre_moe_checkpoint_boundary_forward,
    single_pre_moe_checkpoint_boundary_primal_and_value_and_grads,
    single_pre_moe_checkpoint_boundary_value_and_grads,
    sum_explicit_routing_parameter_gradients,
    validate_distinct_combine_weight_gradients,
    validate_explicit_expert_task_structure,
    validate_explicit_pre_task_structure,
    validate_full_task_structure,
    validate_pure_moe_structure,
)
from experiments.grug.moe.check_jaxpp_group2_moe_boundary_parity import (
    grouped_block_value_and_grads,
    joined_attention_pair_value_and_grads,
    joined_final_head_loss_and_grads,
    joined_moe_pair_value_and_grads,
    ordered_attention_value_and_grads,
    ordered_final_head_loss_and_grads,
    ordered_last_stage_loss_and_grads,
    ordered_moe_pair_value_and_grads,
    ordered_moe_preparation_and_routes,
    packed_attention_value_and_grads,
)
from experiments.grug.moe.model import (
    GrugModelConfig,
    Transformer,
    _run_block_with_remat,
    paired_moe_component_forward,
)
from experiments.grug.moe.train import (
    GrugJaxPPConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _accumulate_microbatch_tree,
    _average_microbatch_tree,
    _combine_grouped_router_metrics,
    _compute_block,
    _compute_stage,
    _grouped_last_stage_loss_and_grads,
    _pack_group_attention_mask,
    _pack_microbatch_pair,
    _sum_microbatch_group,
    _unpack_microbatch_pair,
    explicit_router_pre_boundary_bootstrap,
    explicit_router_pre_boundary_forward,
    explicit_router_pre_boundary_primal_and_value_and_grads,
    explicit_stage_gradient_from_blocks,
    explicit_std_1f1b_stage_schedule,
    pack_fp8_pipeline_wire,
    paired_compute_block_forward,
    paired_compute_block_monolithic_value_and_grads,
    paired_compute_block_value_and_grads,
    unpack_fp8_pipeline_wire,
)

_RUN_SCRIPT = Path("experiments/grug/moe/run_cw_jaxpp_may_d2560.sh")


def test_grouped_explicit_std_1f1b_schedule_preserves_contiguous_pair_order() -> None:
    schedules = tuple(
        explicit_std_1f1b_stage_schedule(
            stages=4,
            microbatches=8,
            stage_index=stage_index,
            group_size=2,
        )
        for stage_index in range(4)
    )

    assert schedules == (
        (
            ("fwd", (0, 1)),
            ("fwd", (2, 3)),
            ("fwd", (4, 5)),
            ("fwd", (6, 7)),
            ("bwd", (0, 1)),
            ("bwd", (2, 3)),
            ("bwd", (4, 5)),
            ("bwd", (6, 7)),
        ),
        (
            ("fwd", (0, 1)),
            ("fwd", (2, 3)),
            ("fwd", (4, 5)),
            ("bwd", (0, 1)),
            ("fwd", (6, 7)),
            ("bwd", (2, 3)),
            ("bwd", (4, 5)),
            ("bwd", (6, 7)),
        ),
        (
            ("fwd", (0, 1)),
            ("fwd", (2, 3)),
            ("bwd", (0, 1)),
            ("fwd", (4, 5)),
            ("bwd", (2, 3)),
            ("fwd", (6, 7)),
            ("bwd", (4, 5)),
            ("bwd", (6, 7)),
        ),
        (
            ("fwd", (0, 1)),
            ("bwd", (0, 1)),
            ("fwd", (2, 3)),
            ("bwd", (2, 3)),
            ("fwd", (4, 5)),
            ("bwd", (4, 5)),
            ("fwd", (6, 7)),
            ("bwd", (6, 7)),
        ),
    )


def test_grouped_stage_task_config_composes_with_fp8_wire_format() -> None:
    config = GrugJaxPPConfig(
        stages=4,
        microbatches=8,
        schedule="std_1f1b",
        implementation="explicit_mpmd",
        explicit_mpmd_pipeline_wire_format="fp8",
        explicit_mpmd_stage_task_microbatch_group_size=2,
    )
    values = (
        jnp.asarray([[1.0, -2.0, 3.0]], dtype=jnp.bfloat16),
        jnp.asarray([[-4.0, 5.0, -6.0]], dtype=jnp.bfloat16),
    )

    restored = tuple(unpack_fp8_pipeline_wire(pack_fp8_pipeline_wire(value, "e4m3"), "e4m3") for value in values)

    assert config.explicit_mpmd_stage_task_microbatch_group_size == 2
    assert jax.tree.structure(restored) == jax.tree.structure(values)
    for actual, expected in zip(restored, values, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.03)


def test_may_launcher_reads_grouped_stage_task_size_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("PP_IMPLEMENTATION", "explicit_mpmd")
    monkeypatch.setenv("PP_SCHEDULE", "std_1f1b")
    monkeypatch.setenv("PP_STAGES", "2")
    monkeypatch.setenv("PP_MPMD_DIM", "2")
    monkeypatch.setenv("PP_MICROBATCHES", "4")
    monkeypatch.delenv("PP_EXPLICIT_MPMD_STAGE_TASK_MICROBATCH_GROUP_SIZE", raising=False)

    default_config = launch_cw_jaxpp_may_d2560.build_pipeline_config()
    monkeypatch.setenv("PP_EXPLICIT_MPMD_STAGE_TASK_MICROBATCH_GROUP_SIZE", "2")
    grouped_config = launch_cw_jaxpp_may_d2560.build_pipeline_config()

    assert default_config.explicit_mpmd_stage_task_microbatch_group_size == 1
    assert grouped_config.explicit_mpmd_stage_task_microbatch_group_size == 2


def test_may_shell_launcher_forwards_grouped_stage_task_size_in_dry_run(tmp_path) -> None:
    environment = {
        **os.environ,
        "HOME": str(tmp_path),
    }
    default_result = subprocess.run(
        ("bash", str(_RUN_SCRIPT), "--run-id", "default-stage-task-test"),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = subprocess.run(
        (
            "bash",
            str(_RUN_SCRIPT),
            "--run-id",
            "grouped-stage-task-test",
            "--implementation",
            "explicit_mpmd",
            "--explicit-mpmd-stage-task-microbatch-group-size",
            "2",
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "explicit_mpmd_stage_task_microbatch_group_size: 1" in default_result.stdout
    assert "explicit_mpmd_stage_task_microbatch_group_size: 2" in result.stdout
    assert (
        '-e PP_EXPLICIT_MPMD_STAGE_TASK_MICROBATCH_GROUP_SIZE "$EXPLICIT_MPMD_STAGE_TASK_MICROBATCH_GROUP_SIZE"'
        in _RUN_SCRIPT.read_text()
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"explicit_mpmd_stage_task_microbatch_group_size": 3}, "group size must be 1 or 2"),
        ({"microbatches": 7}, "even microbatch count"),
        ({"implementation": "auto"}, "grouped explicit MPMD stage tasks require"),
        ({"explicit_mpmd_schedule_mode": "input_gradient_first"}, "do not support input_gradient_first"),
    ),
)
def test_grouped_stage_task_config_rejects_unsupported_modes(overrides, message) -> None:
    kwargs = {
        "stages": 4,
        "microbatches": 8,
        "schedule": "std_1f1b",
        "implementation": "explicit_mpmd",
        "explicit_mpmd_stage_task_microbatch_group_size": 2,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        GrugJaxPPConfig(**kwargs)


def test_grouped_stage_tasks_require_exact_bulk_ring_model() -> None:
    pipeline = GrugJaxPPConfig(
        stages=2,
        microbatches=2,
        schedule="std_1f1b",
        implementation="explicit_mpmd",
        explicit_mpmd_stage_task_microbatch_group_size=2,
    )
    model = GrugModelConfig(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_implementation="ring_fused",
    )

    with pytest.raises(ValueError, match="exact bulk-ring"):
        GrugRunConfig(
            model=model,
            data=LmDataConfig(tokenizer="passthrough", vocab_size=128, components={}),
            resources=ResourceConfig.with_cpu(),
            trainer=GrugTrainerConfig(pipeline=pipeline),
        )


def test_grouped_gradient_sums_average_over_original_microbatch_count() -> None:
    microbatch_gradients = (
        {"weight": jnp.asarray([1.0, 3.0]), "bias": jnp.asarray(2.0)},
        {"weight": jnp.asarray([5.0, 7.0]), "bias": jnp.asarray(4.0)},
        {"weight": jnp.asarray([9.0, 11.0]), "bias": jnp.asarray(6.0)},
        {"weight": jnp.asarray([13.0, 15.0]), "bias": jnp.asarray(8.0)},
    )

    @jax.jit
    def grouped_average(gradients):
        first_pair = _sum_microbatch_group(gradients[:2])
        second_pair = _sum_microbatch_group(gradients[2:])
        grouped_sum = _accumulate_microbatch_tree(first_pair, second_pair)
        return _average_microbatch_tree(grouped_sum, len(gradients))

    actual_average = grouped_average(microbatch_gradients)
    reference_average = jax.tree.map(
        lambda *values: sum(values) / len(values),
        *microbatch_gradients,
    )

    np.testing.assert_allclose(actual_average["weight"], reference_average["weight"])
    np.testing.assert_allclose(actual_average["bias"], reference_average["bias"])


def _assert_tree_rel_l2(actual, expected, *, tolerance: float = 0.002) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        actual_array = np.asarray(actual_leaf, dtype=np.float64)
        expected_array = np.asarray(expected_leaf, dtype=np.float64)
        assert np.all(np.isfinite(actual_array))
        assert np.all(np.isfinite(expected_array))
        if np.array_equal(actual_array, expected_array):
            continue
        denominator = max(float(np.linalg.norm(expected_array.ravel())), 1e-12)
        relative_l2 = float(np.linalg.norm((actual_array - expected_array).ravel())) / denominator
        assert relative_l2 <= tolerance


def _tiny_grouped_last_stage(remat_mode: str, *, top_k: int = 1):
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=max(2, top_k + 1),
        num_experts_per_token=top_k,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
        sliding_window=4,
        router_z_loss_coef=0.1,
        attention_implementation="reference",
        moe_implementation="ring",
        loss_implementation="reference",
        remat_mode=remat_mode,
    )
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.PRNGKey(0))
    return mesh, model.split_for_pipeline(2)[1]


def _shard_group_batch(batch: GrugLmExample, mesh: Mesh) -> GrugLmExample:
    token_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    return dataclasses.replace(
        batch,
        tokens=jax.device_put(batch.tokens, token_sharding),
        loss_weight=jax.device_put(batch.loss_weight, token_sharding),
    )


def _shard_group_hidden(value: jax.Array, mesh: Mesh) -> jax.Array:
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None))
    return jax.device_put(value, sharding)


def test_grouped_pair_pack_round_trips_local_batch_order() -> None:
    first = jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32)
    second = jnp.asarray([[10, 20], [30, 40]], dtype=jnp.int32)

    packed = jax.jit(_pack_microbatch_pair, static_argnames=("name",))((first, second), name="test values")
    unpacked = jax.jit(_unpack_microbatch_pair)(packed)

    np.testing.assert_array_equal(packed, jnp.asarray([[1, 2], [10, 20], [3, 4], [30, 40]]))
    np.testing.assert_array_equal(unpacked[0], first)
    np.testing.assert_array_equal(unpacked[1], second)

    batches = (
        GrugLmExample(
            tokens=first,
            loss_weight=first.astype(jnp.float32),
            attn_mask=AttentionMask.causal().with_segment_ids(first),
        ),
        GrugLmExample(
            tokens=second,
            loss_weight=second.astype(jnp.float32),
            attn_mask=AttentionMask.causal().with_segment_ids(second),
        ),
    )
    packed_mask = jax.jit(_pack_group_attention_mask)(batches)
    assert packed_mask.segment_ids is not None
    np.testing.assert_array_equal(packed_mask.segment_ids[0], packed)
    np.testing.assert_array_equal(packed_mask.segment_ids[1], packed)


def test_grouped_router_metric_reduction_preserves_sum_and_mean_semantics() -> None:
    first = {
        "routing_counts_per_layer": jnp.asarray([[1.0, 2.0]]),
        "capacity_overflow_per_layer": jnp.asarray([3.0]),
        "qb_beta_per_layer": jnp.asarray([[4.0, 5.0]]),
        "routing_entropy_per_layer": jnp.asarray([6.0]),
        "load_balancing_loss_per_layer": jnp.asarray([7.0]),
        "router_z_loss_per_layer": jnp.asarray([8.0]),
    }
    second = jax.tree.map(lambda value: value + 10.0, first)

    combined = jax.jit(_combine_grouped_router_metrics)((first, second))

    for key in ("routing_counts_per_layer", "capacity_overflow_per_layer", "qb_beta_per_layer"):
        np.testing.assert_array_equal(combined[key], first[key] + second[key])
    for key in ("routing_entropy_per_layer", "load_balancing_loss_per_layer", "router_z_loss_per_layer"):
        np.testing.assert_array_equal(combined[key], (first[key] + second[key]) * 0.5)


@pytest.mark.parametrize("remat_mode", ("recompute_all", "save_moe"))
def test_grouped_last_stage_matches_ordered_value_and_vjp_under_jit(remat_mode: str) -> None:
    mesh, stage = _tiny_grouped_last_stage(remat_mode)
    mp = jmp.get_policy("f32")
    qb_betas = jnp.asarray([[0.2, -0.1]], dtype=jnp.float32)
    batches = (
        _shard_group_batch(
            GrugLmExample(
                tokens=jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32),
                loss_weight=jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
                attn_mask=AttentionMask.causal(),
            ),
            mesh,
        ),
        _shard_group_batch(
            GrugLmExample(
                tokens=jnp.asarray([[4, 3, 2, 1]], dtype=jnp.int32),
                loss_weight=jnp.asarray([[1.0, 1.0, 1.0, 0.0]], dtype=jnp.float32),
                attn_mask=AttentionMask.causal(),
            ),
            mesh,
        ),
    )
    hiddens = tuple(_shard_group_hidden(jax.random.normal(jax.random.PRNGKey(key), (1, 4, 8)), mesh) for key in (1, 2))

    grouped_fn = jax.jit(
        lambda params, stage_hiddens, stage_batches: _grouped_last_stage_loss_and_grads(
            params,
            qb_betas,
            stage_hiddens,
            stage_batches,
            mp,
            logsumexp_weight=0.01,
        )
    )
    ordered_fn = jax.jit(
        lambda params, stage_hiddens, stage_batches: ordered_last_stage_loss_and_grads(
            params,
            qb_betas,
            stage_hiddens,
            stage_batches,
            mp,
            logsumexp_weight=0.01,
        )
    )
    with jax.set_mesh(mesh):
        actual = grouped_fn(stage, hiddens, batches)
        expected = ordered_fn(stage, hiddens, batches)

    _assert_tree_rel_l2(actual, expected)


def _tiny_boundary_inputs(mesh: Mesh):
    batches = (
        _shard_group_batch(
            GrugLmExample(
                tokens=jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32),
                loss_weight=jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
                attn_mask=AttentionMask.causal(),
            ),
            mesh,
        ),
        _shard_group_batch(
            GrugLmExample(
                tokens=jnp.asarray([[4, 3, 2, 1]], dtype=jnp.int32),
                loss_weight=jnp.asarray([[1.0, 1.0, 1.0, 0.0]], dtype=jnp.float32),
                attn_mask=AttentionMask.causal(),
            ),
            mesh,
        ),
    )
    hiddens = tuple(_shard_group_hidden(jax.random.normal(jax.random.PRNGKey(key), (1, 4, 8)), mesh) for key in (11, 12))
    cotangents = tuple(
        _shard_group_hidden(jax.random.normal(jax.random.PRNGKey(key), (1, 4, 8)), mesh) for key in (13, 14)
    )
    return batches, hiddens, cotangents


def _block_projection(output: jax.Array, cotangent: jax.Array) -> jax.Array:
    return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))


def _ordered_master_block_value_and_grads(
    params,
    qb_beta,
    hiddens,
    masks,
    output_cotangents,
    mp,
    *,
    remat_mode,
    router_z_loss_scale,
):
    losses = []
    outputs = []
    router_stats = []
    block_gradients = []
    input_gradients = []
    for hidden, mask, output_cotangent in zip(hiddens, masks, output_cotangents, strict=True):

        def projected_block(master_block, current_hidden, mask=mask, output_cotangent=output_cotangent):
            compute_block = _compute_block(master_block, qb_beta, mp)
            output, stats = _run_block_with_remat(
                compute_block,
                current_hidden,
                mask,
                use_pko=False,
                disable_rope=False,
                remat_mode=remat_mode,
                effectful_moe=False,
            )
            loss = _block_projection(output, output_cotangent)
            loss = loss + router_z_loss_scale * stats["router_z_loss"]
            return loss, (output, stats)

        (loss, (output, stats)), (block_gradient, input_gradient) = jax.value_and_grad(
            projected_block,
            argnums=(0, 1),
            has_aux=True,
        )(params, hidden)
        losses.append(loss)
        outputs.append(output)
        router_stats.append(stats)
        block_gradients.append(block_gradient)
        input_gradients.append(input_gradient)
    return (
        tuple(losses),
        tuple(outputs),
        tuple(router_stats),
        _sum_microbatch_group(tuple(block_gradients)),
        tuple(input_gradients),
    )


@pytest.mark.parametrize("remat_mode", ("recompute_all", "save_moe"))
def test_paired_component_block_matches_two_ordered_master_block_vjps(remat_mode: str) -> None:
    mesh, stage = _tiny_grouped_last_stage(remat_mode, top_k=2)
    batches, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    masks = tuple(batch.attn_mask for batch in batches)
    params = stage.blocks[0]
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    router_z_loss_scale = 0.1

    paired_forward = jax.jit(
        lambda block, hidden_pair: paired_compute_block_forward(
            block,
            qb_beta,
            hidden_pair,
            masks,
            mp,
            use_pko=False,
            disable_rope=False,
            remat_mode=remat_mode,
        )
    )
    paired_vjp = jax.jit(
        lambda block, hidden_pair, cotangent_pair: paired_compute_block_value_and_grads(
            block,
            qb_beta,
            hidden_pair,
            masks,
            cotangent_pair,
            mp,
            use_pko=False,
            disable_rope=False,
            remat_mode=remat_mode,
            router_z_loss_scale=router_z_loss_scale,
        )
    )
    ordered_vjp = jax.jit(
        lambda block, hidden_pair, cotangent_pair: _ordered_master_block_value_and_grads(
            block,
            qb_beta,
            hidden_pair,
            masks,
            cotangent_pair,
            mp,
            remat_mode=remat_mode,
            router_z_loss_scale=router_z_loss_scale,
        )
    )

    with jax.set_mesh(mesh):
        component_compute_block = _compute_block(params, qb_beta, mp)
        production_compute_block = _compute_stage(stage, qb_beta[None, :], mp).blocks[0]
        actual = paired_vjp(params, hiddens, output_cotangents)
        expected = ordered_vjp(params, hiddens, output_cotangents)
        forward_outputs, forward_stats = paired_forward(params, hiddens)

    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(component_compute_block),
        jax.tree.leaves(production_compute_block),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    _assert_tree_rel_l2(actual, expected)
    _assert_tree_rel_l2(forward_outputs, expected[1])
    _assert_tree_rel_l2(forward_stats, expected[2])
    assert len(jax.tree.leaves(actual[3])) == len(jax.tree.leaves(expected[3]))
    assert len(jax.tree.leaves(actual[4])) == 2
    for actual_stats, expected_stats in zip(actual[2], expected[2], strict=True):
        np.testing.assert_array_equal(actual_stats["routing_counts"], expected_stats["routing_counts"])
        _assert_tree_rel_l2(actual_stats["qb_beta"], expected_stats["qb_beta"])


@pytest.mark.parametrize("remat_mode", ("recompute_all", "save_moe"))
def test_monolithic_paired_component_block_matches_two_ordered_master_block_vjps(remat_mode: str) -> None:
    mesh, stage = _tiny_grouped_last_stage(remat_mode, top_k=2)
    batches, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    masks = tuple(batch.attn_mask for batch in batches)
    params = stage.blocks[0]
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    router_z_loss_scale = 0.1

    monolithic_vjp = jax.jit(
        lambda block, hidden_pair, cotangent_pair: paired_compute_block_monolithic_value_and_grads(
            block,
            qb_beta,
            hidden_pair,
            masks,
            cotangent_pair,
            mp,
            use_pko=False,
            disable_rope=False,
            remat_mode=remat_mode,
            router_z_loss_scale=router_z_loss_scale,
        )
    )
    ordered_vjp = jax.jit(
        lambda block, hidden_pair, cotangent_pair: _ordered_master_block_value_and_grads(
            block,
            qb_beta,
            hidden_pair,
            masks,
            cotangent_pair,
            mp,
            remat_mode=remat_mode,
            router_z_loss_scale=router_z_loss_scale,
        )
    )

    with jax.set_mesh(mesh):
        actual = monolithic_vjp(params, hiddens, output_cotangents)
        expected = ordered_vjp(params, hiddens, output_cotangents)

    _assert_tree_rel_l2(actual, expected)
    for actual_stats, expected_stats in zip(actual[2], expected[2], strict=True):
        np.testing.assert_array_equal(actual_stats["routing_counts"], expected_stats["routing_counts"])


@pytest.mark.parametrize(
    "paired_vjp",
    (
        paired_moe_joint_checkpoint_value_and_grads,
        paired_moe_no_checkpoint_value_and_grads,
        paired_moe_per_checkpoint_value_and_grads,
    ),
)
def test_paired_moe_checkpoint_boundaries_match_separate_calls(paired_vjp) -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    batches, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    masks = tuple(batch.attn_mask for batch in batches)

    with jax.set_mesh(mesh):
        block = _compute_block(stage.blocks[0], qb_beta, mp)
        post_attention = tuple(
            jax.jit(lambda hidden, mask=mask: block.attention_residual(hidden, mask))(hidden)
            for hidden, mask in zip(hiddens, masks, strict=True)
        )
        mlp_inputs = tuple(block.mlp_gated_norm(block.rms_mlp(hidden)) for hidden in post_attention)
        separate = tuple(
            jax.jit(single_moe_value_and_grads)(block.mlp, mlp_input, cotangent)
            for mlp_input, cotangent in zip(mlp_inputs, output_cotangents, strict=True)
        )
        actual = jax.jit(paired_vjp)(block.mlp, mlp_inputs, output_cotangents)

    expected = (
        (separate[0][0], separate[1][0]),
        (separate[0][1], separate[1][1]),
        (separate[0][2], separate[1][2]),
        _sum_microbatch_group((separate[0][3], separate[1][3])),
        (separate[0][4], separate[1][4]),
        (separate[0][5], separate[1][5]),
    )
    _assert_tree_rel_l2(actual, expected)
    for actual_stats, expected_stats in zip(actual[2], expected[2], strict=True):
        np.testing.assert_array_equal(actual_stats["routing_counts"], expected_stats["routing_counts"])
    for actual_route, expected_route in zip(actual[5], expected[5], strict=True):
        np.testing.assert_array_equal(actual_route["selected_experts"], expected_route["selected_experts"])


def test_full_block_boundary_diagnostic_matches_ordered_checkpoint_controls() -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    _, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    params = stage.blocks[0]
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    with jax.set_mesh(mesh):
        ordered = tuple(
            jax.jit(single_full_block_boundary_value_and_grads)(
                params,
                qb_beta,
                hidden,
                cotangent,
            )
            for hidden, cotangent in zip(hiddens, output_cotangents, strict=True)
        )
        ordered_no_checkpoint = tuple(
            jax.jit(single_full_block_no_checkpoint_boundary_value_and_grads)(
                params,
                qb_beta,
                hidden,
                cotangent,
            )
            for hidden, cotangent in zip(hiddens, output_cotangents, strict=True)
        )
        paired = jax.jit(paired_full_block_boundary_value_and_grads)(
            params,
            qb_beta,
            hiddens,
            output_cotangents,
        )
        per_microbatch_router_gradients = jax.jit(paired_full_block_per_microbatch_router_gradients)(
            params,
            qb_beta,
            hiddens,
            output_cotangents,
        )
        distinct_mlp = jax.jit(paired_distinct_mlp_master_router_gradients)(
            params,
            qb_beta,
            hiddens,
            output_cotangents,
        )

    expected = (
        (ordered[0][0], ordered[1][0]),
        (ordered[0][1], ordered[1][1]),
        (ordered[0][2], ordered[1][2]),
        (ordered[0][3], ordered[1][3]),
        (ordered[0][4], ordered[1][4]),
        (ordered[0][5], ordered[1][5]),
        (ordered[0][6], ordered[1][6]),
        (ordered[0][7], ordered[1][7]),
        _sum_microbatch_group((ordered[0][8], ordered[1][8])),
        (ordered[0][9], ordered[1][9]),
    )
    expected_no_checkpoint = (
        (ordered_no_checkpoint[0][0], ordered_no_checkpoint[1][0]),
        (ordered_no_checkpoint[0][1], ordered_no_checkpoint[1][1]),
        (ordered_no_checkpoint[0][2], ordered_no_checkpoint[1][2]),
        (ordered_no_checkpoint[0][3], ordered_no_checkpoint[1][3]),
        (ordered_no_checkpoint[0][4], ordered_no_checkpoint[1][4]),
        (ordered_no_checkpoint[0][5], ordered_no_checkpoint[1][5]),
        (ordered_no_checkpoint[0][6], ordered_no_checkpoint[1][6]),
        (ordered_no_checkpoint[0][7], ordered_no_checkpoint[1][7]),
        _sum_microbatch_group((ordered_no_checkpoint[0][8], ordered_no_checkpoint[1][8])),
        (ordered_no_checkpoint[0][9], ordered_no_checkpoint[1][9]),
    )
    expected_router_gradients = (
        ordered[0][8].mlp.router,
        ordered[1][8].mlp.router,
        ordered[0][8].mlp.router + ordered[1][8].mlp.router,
    )

    _assert_tree_rel_l2(paired, expected)
    _assert_tree_rel_l2(expected_no_checkpoint, expected)
    _assert_tree_rel_l2(per_microbatch_router_gradients, expected_router_gradients)
    _assert_tree_rel_l2(distinct_mlp[1], expected[5])
    _assert_tree_rel_l2(distinct_mlp[2], expected[7])
    _assert_tree_rel_l2(distinct_mlp[3:6], expected_router_gradients)
    for actual_routes, expected_routes in zip(paired[4], expected[4], strict=True):
        np.testing.assert_array_equal(actual_routes["selected_experts"], expected_routes["selected_experts"])


@pytest.mark.parametrize(
    ("forward", "value_and_grads"),
    (
        (paired_complete_checkpoint_boundary_forward, paired_complete_checkpoint_boundary_value_and_grads),
        (paired_pre_moe_checkpoint_boundary_forward, paired_pre_moe_checkpoint_boundary_value_and_grads),
    ),
)
def test_paired_remat_scope_candidates_match_ordered_full_block_checkpoint(forward, value_and_grads) -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    _, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    params = stage.blocks[0]
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    with jax.set_mesh(mesh):
        ordered_forwards = tuple(
            jax.jit(single_full_block_boundary_forward)(params, qb_beta, hidden) for hidden in hiddens
        )
        actual_forward = jax.jit(forward)(params, qb_beta, hiddens)
        ordered_vjps = tuple(
            jax.jit(single_full_block_boundary_value_and_grads)(params, qb_beta, hidden, cotangent)
            for hidden, cotangent in zip(hiddens, output_cotangents, strict=True)
        )
        actual_vjp = jax.jit(value_and_grads)(params, qb_beta, hiddens, output_cotangents)

    expected_forward = tuple(
        (ordered_forwards[0][index], ordered_forwards[1][index]) for index in range(len(ordered_forwards[0]))
    )
    expected_vjp = (
        (ordered_vjps[0][0], ordered_vjps[1][0]),
        (ordered_vjps[0][1], ordered_vjps[1][1]),
        (ordered_vjps[0][2], ordered_vjps[1][2]),
        (ordered_vjps[0][3], ordered_vjps[1][3]),
        (ordered_vjps[0][4], ordered_vjps[1][4]),
        (ordered_vjps[0][5], ordered_vjps[1][5]),
        (ordered_vjps[0][6], ordered_vjps[1][6]),
        (ordered_vjps[0][7], ordered_vjps[1][7]),
        _sum_microbatch_group((ordered_vjps[0][8], ordered_vjps[1][8])),
        (ordered_vjps[0][9], ordered_vjps[1][9]),
    )
    _assert_tree_rel_l2(actual_forward, expected_forward)
    _assert_tree_rel_l2(actual_vjp, expected_vjp)
    for actual_routes, expected_routes in zip(actual_forward[3], expected_forward[3], strict=True):
        np.testing.assert_array_equal(actual_routes["selected_experts"], expected_routes["selected_experts"])


@pytest.mark.parametrize(
    ("pre_forward", "pre_value_and_grads", "pre_primal_and_value_and_grads"),
    (
        (
            single_pre_moe_checkpoint_boundary_forward,
            single_pre_moe_checkpoint_boundary_value_and_grads,
            single_pre_moe_checkpoint_boundary_primal_and_value_and_grads,
        ),
        (
            single_pre_moe_barrier_checkpoint_boundary_forward,
            single_pre_moe_barrier_checkpoint_boundary_value_and_grads,
            single_pre_moe_barrier_checkpoint_boundary_primal_and_value_and_grads,
        ),
    ),
)
def test_split_executable_boundary_dataflow_matches_ordered_full_block_checkpoint(
    pre_forward,
    pre_value_and_grads,
    pre_primal_and_value_and_grads,
) -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    _, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    params = stage.blocks[0]
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    with jax.set_mesh(mesh):
        ordered_forwards = tuple(
            jax.jit(single_full_block_boundary_forward)(params, qb_beta, hidden) for hidden in hiddens
        )
        ordered_vjps = tuple(
            jax.jit(single_full_block_boundary_value_and_grads)(params, qb_beta, hidden, cotangent)
            for hidden, cotangent in zip(hiddens, output_cotangents, strict=True)
        )
        prepared = tuple(jax.jit(pre_forward)(params, qb_beta, hidden) for hidden in hiddens)
        post_attention = tuple(result[0] for result in prepared)
        mlp_inputs = tuple(result[1] for result in prepared)
        shared_outputs = tuple(result[2] for result in prepared)
        pre_ring = tuple(result[3] for result in prepared)
        routed_outputs, outputs, router_stats = jax.jit(joined_moe_finish_boundary_forward)(
            params,
            qb_beta,
            post_attention,
            mlp_inputs,
            shared_outputs,
        )
        finish_vjp = jax.jit(joined_moe_finish_boundary_value_and_grads)(
            params,
            qb_beta,
            post_attention,
            mlp_inputs,
            shared_outputs,
            output_cotangents,
        )
        boundary_cotangents = tuple(
            (finish_vjp[5][index], finish_vjp[6][index], finish_vjp[7][index]) for index in range(2)
        )
        pre_vjps = tuple(
            jax.jit(pre_value_and_grads)(params, qb_beta, hidden, boundary_cotangent)
            for hidden, boundary_cotangent in zip(hiddens, boundary_cotangents, strict=True)
        )
        pre_vjps_with_primals = tuple(
            jax.jit(pre_primal_and_value_and_grads)(params, qb_beta, hidden, boundary_cotangent)
            for hidden, boundary_cotangent in zip(hiddens, boundary_cotangents, strict=True)
        )

    expected_forward = tuple(
        (ordered_forwards[0][index], ordered_forwards[1][index]) for index in range(len(ordered_forwards[0]))
    )
    actual_forward = (
        post_attention,
        mlp_inputs,
        shared_outputs,
        pre_ring,
        routed_outputs,
        outputs,
        router_stats,
    )
    expected_vjp = (
        (ordered_vjps[0][0], ordered_vjps[1][0]),
        (ordered_vjps[0][1], ordered_vjps[1][1]),
        (ordered_vjps[0][2], ordered_vjps[1][2]),
        (ordered_vjps[0][3], ordered_vjps[1][3]),
        (ordered_vjps[0][4], ordered_vjps[1][4]),
        (ordered_vjps[0][5], ordered_vjps[1][5]),
        (ordered_vjps[0][6], ordered_vjps[1][6]),
        (ordered_vjps[0][7], ordered_vjps[1][7]),
        _sum_microbatch_group((ordered_vjps[0][8], ordered_vjps[1][8])),
        (ordered_vjps[0][9], ordered_vjps[1][9]),
    )
    actual_vjp = (
        finish_vjp[0],
        post_attention,
        mlp_inputs,
        shared_outputs,
        pre_ring,
        finish_vjp[1],
        finish_vjp[2],
        finish_vjp[3],
        _sum_microbatch_group((finish_vjp[4], pre_vjps[0][0], pre_vjps[1][0])),
        (pre_vjps[0][1], pre_vjps[1][1]),
    )
    _assert_tree_rel_l2(actual_forward, expected_forward)
    _assert_tree_rel_l2(actual_vjp, expected_vjp)
    _assert_tree_rel_l2(
        tuple(result[0] for result in pre_vjps_with_primals),
        prepared,
    )
    _assert_tree_rel_l2(
        tuple(result[1:] for result in pre_vjps_with_primals),
        pre_vjps,
    )
    for actual_routes, expected_routes in zip(actual_forward[3], expected_forward[3], strict=True):
        np.testing.assert_array_equal(actual_routes["selected_experts"], expected_routes["selected_experts"])


def test_split_single_finish_vjps_match_joined_finish_and_ordered_full_block() -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    _, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    params = stage.blocks[0]
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    with jax.set_mesh(mesh):
        ordered_vjps = tuple(
            jax.jit(single_full_block_boundary_value_and_grads)(params, qb_beta, hidden, cotangent)
            for hidden, cotangent in zip(hiddens, output_cotangents, strict=True)
        )
        no_checkpoint_vjps = tuple(
            jax.jit(single_full_block_no_checkpoint_boundary_value_and_grads)(
                params,
                qb_beta,
                hidden,
                cotangent,
            )
            for hidden, cotangent in zip(hiddens, output_cotangents, strict=True)
        )
        standalone_prepared = tuple(
            jax.jit(single_pre_moe_checkpoint_boundary_forward)(params, qb_beta, hidden) for hidden in hiddens
        )
        standalone_post_attention = tuple(result[0] for result in standalone_prepared)
        standalone_mlp_inputs = tuple(result[1] for result in standalone_prepared)
        standalone_shared_outputs = tuple(result[2] for result in standalone_prepared)
        bootstrap_finish = jax.jit(joined_moe_finish_boundary_value_and_grads)(
            params,
            qb_beta,
            standalone_post_attention,
            standalone_mlp_inputs,
            standalone_shared_outputs,
            output_cotangents,
        )
        bootstrap_cotangents = tuple(
            (bootstrap_finish[5][index], bootstrap_finish[6][index], bootstrap_finish[7][index]) for index in range(2)
        )
        exposed_pre = tuple(
            jax.jit(single_pre_moe_checkpoint_boundary_primal_and_value_and_grads)(
                params,
                qb_beta,
                hidden,
                boundary_cotangent,
            )
            for hidden, boundary_cotangent in zip(hiddens, bootstrap_cotangents, strict=True)
        )
        exposed_boundaries = tuple(result[0] for result in exposed_pre)
        post_attention = tuple(result[0] for result in exposed_boundaries)
        mlp_inputs = tuple(result[1] for result in exposed_boundaries)
        shared_outputs = tuple(result[2] for result in exposed_boundaries)
        pre_ring = tuple(result[3] for result in exposed_boundaries)
        joined_finish = jax.jit(joined_moe_finish_boundary_value_and_grads)(
            params,
            qb_beta,
            post_attention,
            mlp_inputs,
            shared_outputs,
            output_cotangents,
        )
        single_finish = tuple(
            jax.jit(single_moe_finish_boundary_value_and_grads)(
                params,
                qb_beta,
                post_attention[index],
                mlp_inputs[index],
                shared_outputs[index],
                output_cotangents[index],
            )
            for index in range(2)
        )
        single_boundary_cotangents = tuple(
            (single_finish[index][5], single_finish[index][6], single_finish[index][7]) for index in range(2)
        )
        pre_vjps = tuple(
            jax.jit(single_pre_moe_checkpoint_boundary_primal_and_value_and_grads)(
                params,
                qb_beta,
                hiddens[index],
                single_boundary_cotangents[index],
            )
            for index in range(2)
        )

    combined_single_finish = (
        (single_finish[0][0], single_finish[1][0]),
        (single_finish[0][1], single_finish[1][1]),
        (single_finish[0][2], single_finish[1][2]),
        (single_finish[0][3], single_finish[1][3]),
        _sum_microbatch_group((single_finish[0][4], single_finish[1][4])),
        (single_finish[0][5], single_finish[1][5]),
        (single_finish[0][6], single_finish[1][6]),
        (single_finish[0][7], single_finish[1][7]),
    )
    expected = (
        (ordered_vjps[0][0], ordered_vjps[1][0]),
        (ordered_vjps[0][1], ordered_vjps[1][1]),
        (ordered_vjps[0][2], ordered_vjps[1][2]),
        (ordered_vjps[0][3], ordered_vjps[1][3]),
        (ordered_vjps[0][4], ordered_vjps[1][4]),
        (ordered_vjps[0][5], ordered_vjps[1][5]),
        (ordered_vjps[0][6], ordered_vjps[1][6]),
        (ordered_vjps[0][7], ordered_vjps[1][7]),
        _sum_microbatch_group((ordered_vjps[0][8], ordered_vjps[1][8])),
        (ordered_vjps[0][9], ordered_vjps[1][9]),
    )
    expected_no_checkpoint = (
        (no_checkpoint_vjps[0][0], no_checkpoint_vjps[1][0]),
        (no_checkpoint_vjps[0][1], no_checkpoint_vjps[1][1]),
        (no_checkpoint_vjps[0][2], no_checkpoint_vjps[1][2]),
        (no_checkpoint_vjps[0][3], no_checkpoint_vjps[1][3]),
        (no_checkpoint_vjps[0][4], no_checkpoint_vjps[1][4]),
        (no_checkpoint_vjps[0][5], no_checkpoint_vjps[1][5]),
        (no_checkpoint_vjps[0][6], no_checkpoint_vjps[1][6]),
        (no_checkpoint_vjps[0][7], no_checkpoint_vjps[1][7]),
        _sum_microbatch_group((no_checkpoint_vjps[0][8], no_checkpoint_vjps[1][8])),
        (no_checkpoint_vjps[0][9], no_checkpoint_vjps[1][9]),
    )
    actual = (
        combined_single_finish[0],
        post_attention,
        mlp_inputs,
        shared_outputs,
        pre_ring,
        combined_single_finish[1],
        combined_single_finish[2],
        combined_single_finish[3],
        _sum_microbatch_group((combined_single_finish[4], pre_vjps[0][1], pre_vjps[1][1])),
        (pre_vjps[0][2], pre_vjps[1][2]),
    )
    _assert_tree_rel_l2(combined_single_finish, joined_finish)
    _assert_tree_rel_l2(actual, expected)
    _assert_tree_rel_l2(actual[8:], expected_no_checkpoint[8:])
    for actual_finish, expected_vjp in zip(single_finish, ordered_vjps, strict=True):
        _assert_tree_rel_l2(actual_finish[4].mlp, expected_vjp[8].mlp)
    for actual_routes, expected_routes in zip(pre_ring, expected[4], strict=True):
        np.testing.assert_array_equal(actual_routes["selected_experts"], expected_routes["selected_experts"])
    for actual_stats, expected_stats in zip(combined_single_finish[3], expected[7], strict=True):
        np.testing.assert_array_equal(actual_stats["routing_counts"], expected_stats["routing_counts"])


def test_explicit_routing_split_matches_ordered_full_block() -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    _, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    params = stage.blocks[0]
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    with jax.set_mesh(mesh):
        ordered = tuple(
            jax.jit(single_full_block_boundary_value_and_grads)(params, qb_beta, hidden, cotangent)
            for hidden, cotangent in zip(hiddens, output_cotangents, strict=True)
        )
        standalone_pre = tuple(
            jax.jit(single_explicit_router_pre_boundary_forward)(params, qb_beta, hidden) for hidden in hiddens
        )
        zero_cotangents = tuple(
            tuple(jnp.zeros_like(value) for value in (*boundaries[:3], boundaries[3].combine_weights))
            for boundaries in standalone_pre
        )
        exposed_pre = tuple(
            jax.jit(single_explicit_router_pre_boundary_primal_and_value_and_grads)(
                params,
                qb_beta,
                hidden,
                boundary_cotangent,
            )[0]
            for hidden, boundary_cotangent in zip(hiddens, zero_cotangents, strict=True)
        )
        post_attention = tuple(boundaries[0] for boundaries in exposed_pre)
        mlp_inputs = tuple(boundaries[1] for boundaries in exposed_pre)
        shared_outputs = tuple(boundaries[2] for boundaries in exposed_pre)
        routing_states = tuple(boundaries[3] for boundaries in exposed_pre)
        finish_forward = jax.jit(joined_explicit_routing_finish_boundary_forward)(
            params,
            qb_beta,
            post_attention,
            mlp_inputs,
            shared_outputs,
            routing_states,
            output_cotangents,
        )
        finish = jax.jit(joined_explicit_routing_finish_boundary_value_and_grads)(
            params,
            qb_beta,
            post_attention,
            mlp_inputs,
            shared_outputs,
            routing_states,
            output_cotangents,
        )
        boundary_cotangents = tuple(
            (finish[5][index], finish[6][index], finish[7][index], finish[8][index]) for index in range(2)
        )
        pre_vjps = tuple(
            jax.jit(single_explicit_router_pre_boundary_primal_and_value_and_grads)(
                params,
                qb_beta,
                hiddens[index],
                boundary_cotangents[index],
            )
            for index in range(2)
        )
        pre_structure_jaxpr, _, _ = eqx.filter_make_jaxpr(
            single_explicit_router_pre_boundary_primal_and_value_and_grads
        )(params, qb_beta, hiddens[0], zero_cotangents[0])
        expert_structure_jaxpr, _, _ = eqx.filter_make_jaxpr(joined_explicit_routing_finish_boundary_forward)(
            params,
            qb_beta,
            post_attention,
            mlp_inputs,
            shared_outputs,
            routing_states,
            output_cotangents,
        )

    pre_ring = tuple(
        {
            "router_logits": state.router_logits,
            "selected_experts": state.selected_experts,
            "combine_weights": state.combine_weights,
            "boundary_margin": state.boundary_margin,
        }
        for state in routing_states
    )
    actual = (
        finish_forward[0],
        post_attention,
        mlp_inputs,
        shared_outputs,
        pre_ring,
        finish_forward[1],
        finish_forward[2],
        finish_forward[3],
        sum_explicit_routing_parameter_gradients(finish[4], pre_vjps[0][1], pre_vjps[1][1]),
        (pre_vjps[0][2], pre_vjps[1][2]),
    )
    expected = (
        (ordered[0][0], ordered[1][0]),
        (ordered[0][1], ordered[1][1]),
        (ordered[0][2], ordered[1][2]),
        (ordered[0][3], ordered[1][3]),
        (ordered[0][4], ordered[1][4]),
        (ordered[0][5], ordered[1][5]),
        (ordered[0][6], ordered[1][6]),
        (ordered[0][7], ordered[1][7]),
        _sum_microbatch_group((ordered[0][8], ordered[1][8])),
        (ordered[0][9], ordered[1][9]),
    )
    _assert_tree_rel_l2(exposed_pre, standalone_pre)
    _assert_tree_rel_l2(actual, expected)
    validate_distinct_combine_weight_gradients(routing_states, finish[8])
    validate_explicit_pre_task_structure(explicit_component_task_structure(pre_structure_jaxpr))
    validate_explicit_expert_task_structure(explicit_component_task_structure(expert_structure_jaxpr))
    for actual_routes, expected_routes in zip(pre_ring, expected[4], strict=True):
        np.testing.assert_array_equal(actual_routes["selected_experts"], expected_routes["selected_experts"])


def test_production_explicit_router_bootstrap_matches_zero_cotangent_vjp() -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    batches, hiddens, _ = _tiny_boundary_inputs(mesh)
    params = stage.blocks[0]
    hidden = hiddens[0]
    mask = batches[0].attn_mask
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

    def pre_forward(block, block_hidden):
        return explicit_router_pre_boundary_forward(
            block,
            qb_beta,
            block_hidden,
            mask,
            mp,
            use_pko=False,
            disable_rope=False,
            remat_mode="save_moe",
        )

    def pre_vjp(block, block_hidden, boundary_cotangents):
        return explicit_router_pre_boundary_primal_and_value_and_grads(
            block,
            qb_beta,
            block_hidden,
            boundary_cotangents,
            mask,
            mp,
            use_pko=False,
            disable_rope=False,
            remat_mode="save_moe",
            router_z_loss_scale=0.1,
        )

    def pre_bootstrap(block, block_hidden):
        return explicit_router_pre_boundary_bootstrap(
            block,
            qb_beta,
            block_hidden,
            mask,
            mp,
            use_pko=False,
            disable_rope=False,
            remat_mode="save_moe",
            router_z_loss_scale=0.1,
        )

    with jax.set_mesh(mesh):
        boundaries = jax.jit(pre_forward)(params, hidden)
        zero_cotangents = tuple(jnp.zeros_like(value) for value in (*boundaries[:3], boundaries[3].combine_weights))
        expected = jax.jit(pre_vjp)(params, hidden, zero_cotangents)
        actual = jax.jit(pre_bootstrap)(params, hidden)
        bootstrap_jaxpr, _, _ = eqx.filter_make_jaxpr(pre_bootstrap)(params, hidden)

    _assert_tree_rel_l2(actual, expected)
    np.testing.assert_array_equal(
        actual[0][3].selected_experts,
        expected[0][3].selected_experts,
    )
    validate_explicit_pre_task_structure(explicit_component_task_structure(bootstrap_jaxpr))


def test_explicit_stage_gradient_from_blocks_preserves_stage_tree() -> None:
    _, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    block_gradient = jax.tree.map(jnp.ones_like, stage.blocks[0])
    expected = eqx.tree_at(
        lambda value: value.blocks,
        jax.tree.map(jnp.zeros_like, stage),
        (block_gradient,),
    )

    actual = jax.jit(explicit_stage_gradient_from_blocks)(stage, (block_gradient,))

    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual),
        jax.tree.leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_ordered_checkpoint_allow_cse_control_matches_default_and_no_checkpoint() -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    _, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    params = stage.blocks[0]
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    with jax.set_mesh(mesh):
        default = jax.jit(single_full_block_boundary_value_and_grads)(
            params,
            qb_beta,
            hiddens[0],
            output_cotangents[0],
        )
        allow_cse = jax.jit(single_full_block_allow_cse_boundary_value_and_grads)(
            params,
            qb_beta,
            hiddens[0],
            output_cotangents[0],
        )
        no_checkpoint = jax.jit(single_full_block_no_checkpoint_boundary_value_and_grads)(
            params,
            qb_beta,
            hiddens[0],
            output_cotangents[0],
        )

    _assert_tree_rel_l2(allow_cse, default)
    _assert_tree_rel_l2(allow_cse, no_checkpoint)


def _closed_jaxpr_name_stacks(closed_jaxpr: jax_core.ClosedJaxpr) -> tuple[str, ...]:
    names = []

    def walk_value(value) -> None:
        if isinstance(value, jax_core.ClosedJaxpr):
            walk_jaxpr(value.jaxpr)
        elif isinstance(value, jax_core.Jaxpr):
            walk_jaxpr(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk_value(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk_value(item)

    def walk_jaxpr(jaxpr: jax_core.Jaxpr) -> None:
        for equation in jaxpr.eqns:
            names.append(str(equation.source_info.name_stack))
            walk_value(equation.params)

    walk_jaxpr(closed_jaxpr.jaxpr)
    return tuple(names)


def _closed_jaxpr_checkpoint_shapes(closed_jaxpr: jax_core.ClosedJaxpr) -> dict[str, tuple[int, ...]]:
    checkpoint_shapes = {}

    def walk_value(value) -> None:
        if isinstance(value, jax_core.ClosedJaxpr):
            walk_jaxpr(value.jaxpr)
        elif isinstance(value, jax_core.Jaxpr):
            walk_jaxpr(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk_value(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk_value(item)

    def walk_jaxpr(jaxpr: jax_core.Jaxpr) -> None:
        for equation in jaxpr.eqns:
            if equation.primitive.name == "name":
                checkpoint_shapes[equation.params["name"]] = equation.outvars[0].aval.shape
            walk_value(equation.params)

    walk_jaxpr(closed_jaxpr.jaxpr)
    return checkpoint_shapes


def test_save_moe_policy_retains_compact_router_vjp_state() -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    _, hiddens, _ = _tiny_boundary_inputs(mesh)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)
    policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)

    with jax.set_mesh(mesh):
        block = _compute_block(stage.blocks[0], qb_beta, mp)
        checkpointed_mlp = eqx.filter_checkpoint(lambda mlp, hidden: mlp(hidden), policy=policy)
        closed_jaxpr, _, _ = eqx.filter_make_jaxpr(checkpointed_mlp)(block.mlp, hiddens[0])

    checkpoint_shapes = _closed_jaxpr_checkpoint_shapes(closed_jaxpr)
    expected_router_shapes = {
        CHECKPOINT_ROUTER_SELECTED_EXPERTS: (4, 2),
        CHECKPOINT_ROUTER_UNBIASED_TOPK: (4, 2),
        CHECKPOINT_ROUTER_SIGMOID_WEIGHTS: (4, 2),
        CHECKPOINT_ROUTER_WEIGHT_DENOMINATOR: (4, 1),
        CHECKPOINT_ROUTER_COMBINE_WEIGHTS: (4, 2),
    }
    assert {name: checkpoint_shapes[name] for name in expected_router_shapes} == expected_router_shapes
    assert all(shape != (4, 3) for name, shape in checkpoint_shapes.items() if "router" in name)


def test_paired_moe_component_jaxpr_contains_two_router_calls_and_no_attention() -> None:
    mesh, stage = _tiny_grouped_last_stage("save_moe", top_k=2)
    batches, hiddens, output_cotangents = _tiny_boundary_inputs(mesh)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    qb_beta = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    with jax.set_mesh(mesh):
        block = _compute_block(stage.blocks[0], qb_beta, mp)
        mlp_inputs = tuple(block.mlp_gated_norm(block.rms_mlp(hidden)) for hidden in hiddens)
        closed_jaxpr, _, _ = eqx.filter_make_jaxpr(
            lambda mlp, inputs: paired_moe_component_forward(mlp, inputs, remat_mode="save_moe")
        )(block.mlp, mlp_inputs)

    name_stacks = _closed_jaxpr_name_stacks(closed_jaxpr)
    router_calls = [name for name in name_stacks if name.endswith("_paired_moe_calls/MoEMLP/MoEMLP.route/td,de->te")]
    assert len(router_calls) == 2
    assert not any("Attention" in name or "_BlockAttentionView" in name for name in name_stacks)
    structure = component_structure(closed_jaxpr)
    validate_pure_moe_structure(structure)

    masks = tuple(batch.attn_mask for batch in batches)
    with jax.set_mesh(mesh):
        full_vjp_jaxpr, _, _ = eqx.filter_make_jaxpr(
            lambda params, hidden_pair, cotangent_pair: paired_compute_block_value_and_grads(
                params,
                qb_beta,
                hidden_pair,
                masks,
                cotangent_pair,
                mp,
                use_pko=False,
                disable_rope=False,
                remat_mode="save_moe",
                router_z_loss_scale=0.1,
            )
        )(stage.blocks[0], hiddens, output_cotangents)
    full_vjp_structure = component_structure(full_vjp_jaxpr)
    validate_full_task_structure(full_vjp_structure)
    assert full_vjp_structure["ring_body_count"] > structure["ring_body_count"]
    assert full_vjp_structure["router_call_count"] > structure["router_call_count"]


def test_two_learned_router_moe_calls_in_one_vag_match_separate_vags() -> None:
    mesh, stage = _tiny_grouped_last_stage("recompute_all", top_k=2)
    batches, hiddens, cotangents = _tiny_boundary_inputs(mesh)
    block = stage.blocks[0]
    masks = tuple(batch.attn_mask for batch in batches)
    with jax.set_mesh(mesh):
        post_attention = tuple(
            jax.jit(lambda hidden, mask=mask: block.attention_residual(hidden, mask))(hidden)
            for hidden, mask in zip(hiddens, masks, strict=True)
        )
        joined = jax.jit(joined_moe_pair_value_and_grads)(block, post_attention, cotangents)
        ordered = jax.jit(ordered_moe_pair_value_and_grads)(block, post_attention, cotangents)

    _assert_tree_rel_l2(joined, ordered)


def test_two_attention_calls_in_one_vag_match_separate_vags() -> None:
    mesh, stage = _tiny_grouped_last_stage("recompute_all", top_k=2)
    batches, hiddens, cotangents = _tiny_boundary_inputs(mesh)
    masks = tuple(batch.attn_mask for batch in batches)

    with jax.set_mesh(mesh):
        joined = jax.jit(
            lambda block, hidden_pair, cotangent_pair: joined_attention_pair_value_and_grads(
                block,
                hidden_pair,
                cotangent_pair,
                masks,
                use_pko=False,
                disable_rope=False,
            )
        )(stage.blocks[0], hiddens, cotangents)
        ordered = jax.jit(
            lambda block, hidden_pair, cotangent_pair: ordered_attention_value_and_grads(
                block,
                hidden_pair,
                cotangent_pair,
                masks,
                use_pko=False,
                disable_rope=False,
            )
        )(stage.blocks[0], hiddens, cotangents)

    _assert_tree_rel_l2(joined, ordered)


def test_grouped_block_remat_modes_match_no_checkpoint() -> None:
    mesh, stage = _tiny_grouped_last_stage("recompute_all", top_k=2)
    batches, hiddens, cotangents = _tiny_boundary_inputs(mesh)
    packed_hidden = _pack_microbatch_pair(hiddens, name="hidden")
    packed_cotangent = _pack_microbatch_pair(cotangents, name="cotangent")
    packed_mask = _pack_group_attention_mask(batches)

    with jax.set_mesh(mesh):
        no_checkpoint = jax.jit(
            lambda block, hidden, cotangent: grouped_block_value_and_grads(
                block,
                hidden,
                cotangent,
                packed_mask,
                remat_mode=None,
            )
        )(stage.blocks[0], packed_hidden, packed_cotangent)
        recompute_all = jax.jit(
            lambda block, hidden, cotangent: grouped_block_value_and_grads(
                block,
                hidden,
                cotangent,
                packed_mask,
                remat_mode="recompute_all",
            )
        )(stage.blocks[0], packed_hidden, packed_cotangent)
        save_moe = jax.jit(
            lambda block, hidden, cotangent: grouped_block_value_and_grads(
                block,
                hidden,
                cotangent,
                packed_mask,
                remat_mode="save_moe",
            )
        )(stage.blocks[0], packed_hidden, packed_cotangent)

    _assert_tree_rel_l2(recompute_all, no_checkpoint)
    _assert_tree_rel_l2(save_moe, no_checkpoint)


def test_packed_reference_attention_value_and_vjp_match_ordered_calls() -> None:
    mesh, stage = _tiny_grouped_last_stage("recompute_all", top_k=2)
    batches, hiddens, cotangents = _tiny_boundary_inputs(mesh)
    masks = tuple(batch.attn_mask for batch in batches)
    packed_mask = _pack_group_attention_mask(batches)

    with jax.set_mesh(mesh):
        packed = jax.jit(
            lambda block, hidden_pair, cotangent_pair: packed_attention_value_and_grads(
                block,
                hidden_pair,
                cotangent_pair,
                packed_mask,
                use_pko=False,
                disable_rope=False,
            )
        )(stage.blocks[0], hiddens, cotangents)
        ordered = jax.jit(
            lambda block, hidden_pair, cotangent_pair: ordered_attention_value_and_grads(
                block,
                hidden_pair,
                cotangent_pair,
                masks,
                use_pko=False,
                disable_rope=False,
            )
        )(stage.blocks[0], hiddens, cotangents)
        packed_preparation = jax.jit(ordered_moe_preparation_and_routes)(stage.blocks[0], packed[1])
        ordered_preparation = jax.jit(ordered_moe_preparation_and_routes)(stage.blocks[0], ordered[1])

    _assert_tree_rel_l2(packed, ordered)
    _assert_tree_rel_l2(packed_preparation[0], ordered_preparation[0])
    for packed_routes, ordered_routes in zip(packed_preparation[1], ordered_preparation[1], strict=True):
        np.testing.assert_array_equal(packed_routes, ordered_routes)


def test_final_norm_head_pair_vag_matches_separate_weighted_losses() -> None:
    mesh, stage = _tiny_grouped_last_stage("recompute_all", top_k=2)
    batches, hiddens, _ = _tiny_boundary_inputs(mesh)
    with jax.set_mesh(mesh):
        block_results = tuple(
            jax.jit(lambda hidden, mask=batch.attn_mask: stage.block_range(hidden, mask))(hidden)
            for hidden, batch in zip(hiddens, batches, strict=True)
        )
        final_inputs = tuple(result[0] for result in block_results)
        router_metrics = tuple(result[1] for result in block_results)
        joined = jax.jit(
            lambda params, hidden_pair, metrics_pair: joined_final_head_loss_and_grads(
                params,
                hidden_pair,
                batches,
                metrics_pair,
                logsumexp_weight=0.01,
            )
        )(stage, final_inputs, router_metrics)
        ordered = jax.jit(
            lambda params, hidden_pair, metrics_pair: ordered_final_head_loss_and_grads(
                params,
                hidden_pair,
                batches,
                metrics_pair,
                logsumexp_weight=0.01,
            )
        )(stage, final_inputs, router_metrics)

    _assert_tree_rel_l2(joined, ordered)
