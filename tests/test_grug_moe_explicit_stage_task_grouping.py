# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from fray.cluster import ResourceConfig
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.moe import launch_cw_jaxpp_may_d2560
from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe.train import (
    GrugJaxPPConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _accumulate_microbatch_tree,
    _average_microbatch_tree,
    _combine_grouped_router_metrics,
    _compute_stage,
    _grouped_last_stage_loss_and_grads,
    _pack_group_attention_mask,
    _pack_microbatch_pair,
    _sum_microbatch_group,
    _unpack_microbatch_pair,
    explicit_std_1f1b_stage_schedule,
    pack_fp8_pipeline_wire,
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


def _tiny_grouped_last_stage(remat_mode: str):
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=2,
        num_experts_per_token=1,
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

    def ordered_loss_and_grads(params, stage_hiddens, stage_batches):
        losses = []
        qb_betas_next = []
        grads = []
        d_hiddens = []
        for hidden, batch in zip(stage_hiddens, stage_batches, strict=True):

            def loss_fn(stage_params, stage_hidden, batch=batch):
                compute_params = _compute_stage(stage_params, qb_betas, mp)
                output_hidden, router_metrics = compute_params.block_range(stage_hidden, mask=batch.attn_mask)
                final_hidden = compute_params.finalize_hidden(output_hidden)
                loss, metrics = compute_params.hidden_next_token_loss(
                    final_hidden,
                    batch.tokens,
                    batch.loss_weight,
                    router_metrics,
                    reduction="mean",
                    logsumexp_weight=0.01,
                    return_router_metrics=True,
                )
                return loss, metrics["qb_beta_per_layer"]

            (loss, qb_next), (grad, d_hidden) = jax.value_and_grad(
                loss_fn,
                argnums=(0, 1),
                has_aux=True,
            )(params, hidden)
            losses.append(loss)
            qb_betas_next.append(qb_next)
            grads.append(grad)
            d_hiddens.append(d_hidden)
        return (
            _sum_microbatch_group(tuple(losses)),
            _sum_microbatch_group(tuple(qb_betas_next)),
            _sum_microbatch_group(tuple(grads)),
            tuple(d_hiddens),
        )

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
    ordered_fn = jax.jit(ordered_loss_and_grads)
    with jax.set_mesh(mesh):
        actual = grouped_fn(stage, hiddens, batches)
        expected = ordered_fn(stage, hiddens, batches)

    _assert_tree_rel_l2(actual, expected)
