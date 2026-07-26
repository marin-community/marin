# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe import launch_cw_jaxpp_may_d2560
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import (
    GrugJaxPPConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _accumulate_microbatch_tree,
    _average_microbatch_tree,
    _sum_microbatch_group,
    _sum_stacked_microbatch_group,
    _unstack_microbatch_group,
    _vmap_microbatch_group,
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


def test_stacked_vmap_group_matches_ordered_value_and_gradients_under_jit() -> None:
    batches = (
        GrugLmExample.causal(jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)),
        GrugLmExample.causal(jnp.asarray([4, 3, 2, 1], dtype=jnp.int32)),
    )
    hiddens = (
        jnp.asarray([0.25, -0.5, 0.75, -1.0], dtype=jnp.float32),
        jnp.asarray([-1.0, 0.75, -0.5, 0.25], dtype=jnp.float32),
    )
    params = {"weight": jnp.asarray([1.5, -0.75, 0.5, 2.0], dtype=jnp.float32)}
    qb_betas = jnp.asarray([0.125, -0.25, 0.375, -0.5], dtype=jnp.float32)

    def single_loss_and_grads(params, qb_betas, hidden, batch):
        def loss_fn(stage_params, stage_hidden):
            prediction = stage_params["weight"] * stage_hidden + qb_betas
            target = batch.tokens.astype(jnp.float32)
            loss = jnp.sum(jnp.square(prediction - target) * batch.loss_weight)
            return loss, prediction

        (loss, qb_next), (grads, d_hidden) = jax.value_and_grad(
            loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(params, hidden)
        return loss, qb_next, grads, d_hidden

    @jax.jit
    def stacked_group(params, qb_betas, hiddens, batches):
        losses, qb_next, grads, d_hiddens = _vmap_microbatch_group(
            single_loss_and_grads,
            unmapped_args=(params, qb_betas),
            mapped_groups=(hiddens, batches),
        )
        return (
            _sum_stacked_microbatch_group(losses),
            _sum_stacked_microbatch_group(qb_next),
            _sum_stacked_microbatch_group(grads),
            _unstack_microbatch_group(d_hiddens),
        )

    actual = stacked_group(params, qb_betas, hiddens, batches)
    ordered = tuple(
        single_loss_and_grads(params, qb_betas, hidden, batch) for hidden, batch in zip(hiddens, batches, strict=True)
    )
    expected = (
        _sum_microbatch_group(tuple(value[0] for value in ordered)),
        _sum_microbatch_group(tuple(value[1] for value in ordered)),
        _sum_microbatch_group(tuple(value[2] for value in ordered)),
        tuple(value[3] for value in ordered),
    )

    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-6)
