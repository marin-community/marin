# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.expert_merge import convert_one_expert_pair, forward_with_moe_traces
from experiments.grug.moe.gradient_conflict import (
    GRADIENT_PAIR_METRIC_NAMES,
    accumulate_gradient_conflict_batch,
    direct_shared_bank_gradients,
    finalize_gradient_conflict,
    gradient_pair_metrics,
    initial_gradient_conflict_accumulator,
    selected_expert_ids_exact,
)
from experiments.grug.moe.merge_recovery import recovery_forward
from experiments.grug.moe.model import GrugModelConfig, Transformer

_AFFECTED_LAYERS = (1, 2)


def _tiny_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="scatter",
    )


def _teacher_and_nontrivial_student(
    source_to_shared: np.ndarray | None = None,
) -> tuple[Transformer, Transformer]:
    teacher = Transformer.init(_tiny_config(), key=jax.random.key(0))
    if source_to_shared is None:
        source_to_shared = np.arange(teacher.config.num_experts)
    student = convert_one_expert_pair(
        teacher,
        representative_layer=_AFFECTED_LAYERS[0],
        source_layer=_AFFECTED_LAYERS[1],
        source_to_shared=source_to_shared,
    )
    return teacher, eqx.tree_at(
        lambda model: model.expert_banks[1].w_down,
        student,
        student.expert_banks[1].w_down * 0.8,
    )


def _tree_relative_error(actual, expected) -> float:
    error = optax.global_norm(jax.tree.map(lambda left, right: left - right, actual, expected))
    scale = optax.global_norm(expected)
    return float(error / jnp.maximum(scale, 1e-30))


def test_direct_shared_bank_gradients_match_existing_losses_and_joint_gradient() -> None:
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        teacher, student = _teacher_and_nontrivial_student()
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        batch = direct_shared_bank_gradients(
            student,
            teacher,
            tokens,
            affected_layers=_AFFECTED_LAYERS,
            source_to_shared=tuple(range(teacher.config.num_experts)),
        )
        recovery = recovery_forward(
            student,
            teacher,
            tokens,
            affected_layers=_AFFECTED_LAYERS,
            source_to_shared=tuple(range(teacher.config.num_experts)),
        )
        _, traces = forward_with_moe_traces(student, tokens, target_layers=_AFFECTED_LAYERS)
        detached_inputs = tuple(jax.lax.stop_gradient(traces[layer].mlp_input) for layer in _AFFECTED_LAYERS)
        teacher_targets = []
        for layer, mlp_input in zip(_AFFECTED_LAYERS, detached_inputs, strict=True):
            teacher_block = teacher.blocks[layer]
            teacher_bank = teacher.expert_banks[teacher_block.expert_bank_index]
            target = teacher_block.mlp.forward_with_trace(mlp_input, teacher_bank).routed_output
            teacher_targets.append(jax.lax.stop_gradient(target.astype(jnp.float32)))

        def joint_loss(candidate_bank):
            total = jnp.zeros((), dtype=jnp.float32)
            for layer, mlp_input, target in zip(
                _AFFECTED_LAYERS,
                detached_inputs,
                teacher_targets,
                strict=True,
            ):
                block = student.blocks[layer]
                output = block.mlp.forward_with_trace(
                    mlp_input,
                    candidate_bank,
                    block.routed_expert_adapter,
                ).routed_output.astype(jnp.float32)
                total += jnp.mean(jnp.square(output - target)) / (jnp.mean(jnp.square(target)) + 1e-8)
            return total

        joint_gradient = jax.grad(joint_loss)(student.expert_banks[1])
        summed_gradient = jax.tree.map(lambda left, right: left + right, *batch.gradients)

    np.testing.assert_allclose(jnp.sqrt(batch.layer_losses), recovery.moe_output_nrmse, rtol=1e-6, atol=1e-6)
    assert _tree_relative_error(summed_gradient, joint_gradient) <= 1e-5
    np.testing.assert_array_equal(batch.router_top1_agreement_by_layer, 1.0)
    np.testing.assert_array_equal(batch.router_selected_ids_exact_by_layer, 1.0)
    np.testing.assert_array_equal(batch.combine_weight_max_abs_diff_by_layer, 0.0)
    np.testing.assert_array_equal(batch.capacity_overflow_by_layer, 0.0)


def test_gradient_pair_metrics_distinguish_alignment_opposition_and_orthogonality() -> None:
    first = {"value": jnp.asarray([1.0, 0.0])}
    cases = (
        ({"value": jnp.asarray([2.0, 0.0])}, 1.0, 0.0),
        ({"value": jnp.asarray([-1.0, 0.0])}, -1.0, 1.0),
        ({"value": jnp.asarray([0.0, 1.0])}, 0.0, 1.0 - np.sqrt(2.0) / 2.0),
    )
    cosine_index = GRADIENT_PAIR_METRIC_NAMES.index("cosine")
    cancellation_index = GRADIENT_PAIR_METRIC_NAMES.index("cancellation")
    for second, expected_cosine, expected_cancellation in cases:
        metrics = gradient_pair_metrics((first, second))
        np.testing.assert_allclose(metrics[cosine_index], expected_cosine, rtol=0, atol=1e-7)
        np.testing.assert_allclose(metrics[cancellation_index], expected_cancellation, rtol=0, atol=1e-7)


def test_accumulator_uses_mean_gradient_not_mean_batch_cosine() -> None:
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        teacher, student = _teacher_and_nontrivial_student()
        bank = student.expert_banks[1]
        accumulator = initial_gradient_conflict_accumulator(bank, num_batches=2)
        token_batches = (
            jnp.arange(8, dtype=jnp.int32).reshape(1, 8),
            jnp.asarray([[31, 30, 29, 28, 27, 26, 25, 24]], dtype=jnp.int32),
        )
        direct_batches = []
        for batch_index, tokens in enumerate(token_batches):
            accumulator = accumulate_gradient_conflict_batch(
                accumulator,
                student,
                teacher,
                tokens,
                None,
                jnp.asarray(batch_index, dtype=jnp.int32),
                affected_layers=_AFFECTED_LAYERS,
                source_to_shared=tuple(range(teacher.config.num_experts)),
            )
            direct_batches.append(
                direct_shared_bank_gradients(
                    student,
                    teacher,
                    tokens,
                    affected_layers=_AFFECTED_LAYERS,
                    source_to_shared=tuple(range(teacher.config.num_experts)),
                )
            )
        result = finalize_gradient_conflict(accumulator, num_batches=2)
        mean_gradients = tuple(
            jax.tree.map(
                lambda first, second: (first + second) / 2,
                direct_batches[0].gradients[layer_index],
                direct_batches[1].gradients[layer_index],
            )
            for layer_index in range(2)
        )

    np.testing.assert_allclose(result.aggregate_whole_metrics, gradient_pair_metrics(mean_gradients), rtol=1e-6)
    cosine_index = GRADIENT_PAIR_METRIC_NAMES.index("cosine")
    mean_batch_cosine = jnp.mean(result.batch_whole_metrics[:, cosine_index])
    assert not np.isclose(result.aggregate_whole_metrics[cosine_index], mean_batch_cosine)
    np.testing.assert_allclose(
        result.mean_layer_losses,
        jnp.mean(jnp.stack([batch.layer_losses for batch in direct_batches]), axis=0),
        rtol=1e-6,
    )


def test_selected_expert_control_maps_nonidentity_assignment_and_rejects_rank_swap() -> None:
    assignment = np.asarray([1, 0, 3, 2], dtype=np.int32)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        teacher, student = _teacher_and_nontrivial_student(assignment)
        batch = direct_shared_bank_gradients(
            student,
            teacher,
            jnp.arange(8, dtype=jnp.int32).reshape(1, 8),
            affected_layers=_AFFECTED_LAYERS,
            source_to_shared=tuple(int(index) for index in assignment),
        )
    np.testing.assert_array_equal(batch.router_selected_ids_exact_by_layer, 1.0)
    selected = jnp.asarray([[1, 3, 2, 0], [0, 2, 1, 3]], dtype=jnp.int32)
    rank_swapped = selected.at[:, 1].set(selected[:, 2]).at[:, 2].set(selected[:, 1])
    assert float(selected_expert_ids_exact(selected, rank_swapped)) == 0.0
