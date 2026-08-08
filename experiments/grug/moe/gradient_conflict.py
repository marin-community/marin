# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only direct-gradient diagnostics for one shared routed-expert bank."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.model import Block, Transformer

_METRIC_EPSILON = 1e-30
GRADIENT_PAIR_METRIC_NAMES = (
    "first_layer_norm",
    "second_layer_norm",
    "dot",
    "cosine",
    "norm_ratio",
    "cancellation",
)
GRADIENT_PROJECTION_NAMES = ("gate", "up", "down")


class GradientConflictBatch(NamedTuple):
    layer_losses: jax.Array
    gradients: tuple[MoEExpertMlp, MoEExpertMlp]
    routing_counts_by_layer: jax.Array
    capacity_overflow_by_layer: jax.Array
    router_top1_agreement_by_layer: jax.Array
    router_selected_ids_exact_by_layer: jax.Array
    combine_weight_max_abs_diff_by_layer: jax.Array


class GradientConflictAccumulator(NamedTuple):
    gradient_sums: tuple[MoEExpertMlp, MoEExpertMlp]
    batch_layer_losses: jax.Array
    batch_whole_metrics: jax.Array
    batch_projection_metrics: jax.Array
    routing_counts_by_layer: jax.Array
    capacity_overflow_by_layer: jax.Array
    router_top1_agreement_by_layer: jax.Array
    router_selected_ids_exact_by_layer: jax.Array
    combine_weight_max_abs_diff_by_layer: jax.Array


class GradientConflictResult(NamedTuple):
    mean_layer_losses: jax.Array
    aggregate_whole_metrics: jax.Array
    aggregate_projection_metrics: jax.Array
    per_expert_metrics: jax.Array
    batch_layer_losses: jax.Array
    batch_whole_metrics: jax.Array
    batch_projection_metrics: jax.Array
    routing_counts_by_layer: jax.Array
    capacity_overflow_by_layer: jax.Array
    router_top1_agreement_by_layer: jax.Array
    router_selected_ids_exact_by_layer: jax.Array
    combine_weight_max_abs_diff_by_layer: jax.Array


def _tree_dot(left, right) -> jax.Array:
    """Compute a leafwise fp32 dot without flattening explicitly sharded tensors."""
    products = jax.tree.map(
        lambda left_leaf, right_leaf: left_leaf.astype(jnp.float32) * right_leaf.astype(jnp.float32),
        left,
        right,
    )
    return optax.tree_utils.tree_sum(products)


def _pair_metrics_from_dots(left_squared: jax.Array, right_squared: jax.Array, dot: jax.Array) -> jax.Array:
    left_norm = jnp.sqrt(jnp.maximum(left_squared, 0.0))
    right_norm = jnp.sqrt(jnp.maximum(right_squared, 0.0))
    norm_product = left_norm * right_norm
    max_norm = jnp.maximum(left_norm, right_norm)
    norm_sum = left_norm + right_norm
    sum_norm = jnp.sqrt(jnp.maximum(left_squared + right_squared + 2.0 * dot, 0.0))
    cosine = jnp.where(norm_product > _METRIC_EPSILON, dot / norm_product, jnp.nan)
    norm_ratio = jnp.where(max_norm > _METRIC_EPSILON, jnp.minimum(left_norm, right_norm) / max_norm, jnp.nan)
    cancellation = jnp.where(norm_sum > _METRIC_EPSILON, 1.0 - sum_norm / norm_sum, jnp.nan)
    return jnp.stack((left_norm, right_norm, dot, cosine, norm_ratio, cancellation))


def gradient_pair_metrics(gradients: tuple[MoEExpertMlp, MoEExpertMlp]) -> jax.Array:
    """Return whole-bank norm, alignment, and cancellation metrics."""
    left, right = gradients
    return _pair_metrics_from_dots(_tree_dot(left, left), _tree_dot(right, right), _tree_dot(left, right))


def _array_pair_metrics(left: jax.Array, right: jax.Array) -> jax.Array:
    left = left.astype(jnp.float32)
    right = right.astype(jnp.float32)
    return _pair_metrics_from_dots(
        jnp.sum(jnp.square(left)),
        jnp.sum(jnp.square(right)),
        jnp.sum(left * right),
    )


def projection_gradient_pair_metrics(gradients: tuple[MoEExpertMlp, MoEExpertMlp]) -> jax.Array:
    """Return one gradient-pair metric row for each gate/up/down projection."""
    left, right = gradients
    return jnp.stack(
        (
            _array_pair_metrics(left.w_gate, right.w_gate),
            _array_pair_metrics(left.w_up, right.w_up),
            _array_pair_metrics(left.w_down, right.w_down),
        )
    )


def per_expert_gradient_pair_metrics(gradients: tuple[MoEExpertMlp, MoEExpertMlp]) -> jax.Array:
    """Return whole-expert metrics after combining gate/up/down contributions."""
    left, right = gradients
    left_squared = jnp.zeros((left.w_gate.shape[0],), dtype=jnp.float32)
    right_squared = jnp.zeros_like(left_squared)
    dot = jnp.zeros_like(left_squared)
    for left_projection, right_projection in (
        (left.w_gate, right.w_gate),
        (left.w_up, right.w_up),
        (left.w_down, right.w_down),
    ):
        left_projection = left_projection.astype(jnp.float32)
        right_projection = right_projection.astype(jnp.float32)
        left_squared += jnp.sum(jnp.square(left_projection), axis=(1, 2))
        right_squared += jnp.sum(jnp.square(right_projection), axis=(1, 2))
        dot += jnp.sum(left_projection * right_projection, axis=(1, 2))

    left_norm = jnp.sqrt(jnp.maximum(left_squared, 0.0))
    right_norm = jnp.sqrt(jnp.maximum(right_squared, 0.0))
    norm_product = left_norm * right_norm
    max_norm = jnp.maximum(left_norm, right_norm)
    norm_sum = left_norm + right_norm
    sum_norm = jnp.sqrt(jnp.maximum(left_squared + right_squared + 2.0 * dot, 0.0))
    cosine = jnp.where(norm_product > _METRIC_EPSILON, dot / norm_product, jnp.nan)
    norm_ratio = jnp.where(max_norm > _METRIC_EPSILON, jnp.minimum(left_norm, right_norm) / max_norm, jnp.nan)
    cancellation = jnp.where(norm_sum > _METRIC_EPSILON, 1.0 - sum_norm / norm_sum, jnp.nan)
    return jnp.stack((left_norm, right_norm, dot, cosine, norm_ratio, cancellation), axis=-1)


def _mapped_teacher_experts(
    selected_experts: jax.Array,
    source_to_shared: tuple[int, ...] | None,
) -> jax.Array:
    if source_to_shared is None:
        return selected_experts
    assignment = jnp.asarray(source_to_shared, dtype=selected_experts.dtype)
    if get_abstract_mesh().empty:
        return assignment[selected_experts]
    assignment = jax.sharding.reshard(assignment, P(None))
    return assignment.at[selected_experts].get(out_sharding=P(("replica_dcn", "data", "expert"), None))


def _direct_layer_loss(
    candidate_bank: MoEExpertMlp,
    *,
    student_block: Block,
    mlp_input: jax.Array,
    teacher_output: jax.Array,
    normalization_epsilon: float,
) -> jax.Array:
    candidate_trace = student_block.mlp.forward_with_trace(
        mlp_input,
        candidate_bank,
        student_block.routed_expert_adapter,
    )
    student_output = candidate_trace.routed_output.astype(jnp.float32)
    numerator = jnp.mean(jnp.square(student_output - teacher_output))
    denominator = jax.lax.stop_gradient(jnp.mean(jnp.square(teacher_output))) + normalization_epsilon
    return numerator / denominator


def selected_expert_ids_exact(student_selected: jax.Array, teacher_selected: jax.Array) -> jax.Array:
    """Return one only when every selected expert ID agrees at the same top-k rank."""
    if student_selected.shape != teacher_selected.shape:
        raise ValueError(
            f"student and teacher selected-ID shapes differ: {student_selected.shape} != {teacher_selected.shape}"
        )
    return jnp.all(student_selected == teacher_selected).astype(jnp.float32)


def direct_shared_bank_gradients(
    student: Transformer,
    teacher: Transformer,
    token_ids: jax.Array,
    *,
    affected_layers: tuple[int, int],
    source_to_shared: tuple[int, ...] | None,
    mask: AttentionMask | jax.Array | None = None,
    normalization_epsilon: float = 1e-8,
) -> GradientConflictBatch:
    """Measure each affected layer's direct local request to one shared bank."""
    if tuple(sorted(affected_layers)) != affected_layers or len(set(affected_layers)) != 2:
        raise ValueError(f"affected_layers must contain two distinct layers in model order, got {affected_layers}")
    if any(layer < 0 or layer >= len(student.blocks) for layer in affected_layers):
        raise IndexError(f"affected_layers must lie in [0, {len(student.blocks)}), got {affected_layers}")
    if len(student.blocks) != len(teacher.blocks):
        raise ValueError("student and teacher must have the same number of layers")
    bank_indices = tuple(student.blocks[layer].expert_bank_index for layer in affected_layers)
    if len(set(bank_indices)) != 1:
        raise ValueError(f"affected student layers must share one expert bank, got bank IDs {bank_indices}")
    shared_bank_index = bank_indices[0]
    bank_use_count = sum(block.expert_bank_index == shared_bank_index for block in student.blocks)
    if bank_use_count != len(affected_layers):
        raise ValueError(
            f"shared bank {shared_bank_index} must be used by exactly the affected layers, got {bank_use_count} uses"
        )
    if normalization_epsilon <= 0:
        raise ValueError("normalization_epsilon must be positive")
    if mask is None:
        mask = AttentionMask.causal()

    hidden = student.embed_inputs(token_ids)
    student_traces = []
    affected_layer_set = set(affected_layers)
    for layer_index, block in enumerate(student.blocks):
        options = student.block_call_options(mask, layer_index)
        bank = student.expert_banks[block.expert_bank_index]
        trace = block.forward_with_moe_trace(hidden, bank, options)
        hidden = trace.hidden
        if layer_index in affected_layer_set:
            student_traces.append(trace)
        if layer_index == affected_layers[-1]:
            break

    shared_bank = student.expert_banks[shared_bank_index]
    layer_losses = []
    gradients = []
    routing_counts = []
    capacity_overflow = []
    top1_agreement = []
    selected_ids_exact = []
    combine_weight_max_abs_diff = []
    for trace_index, (layer, student_trace) in enumerate(zip(affected_layers, student_traces, strict=True)):
        mlp_input = jax.lax.stop_gradient(student_trace.mlp_input)
        teacher_block = teacher.blocks[layer]
        teacher_bank = teacher.expert_banks[teacher_block.expert_bank_index]
        teacher_trace = teacher_block.mlp.forward_with_trace(mlp_input, teacher_bank)
        teacher_output = jax.lax.stop_gradient(teacher_trace.routed_output.astype(jnp.float32))
        teacher_selected = _mapped_teacher_experts(
            teacher_trace.routing.selected_experts,
            source_to_shared if trace_index == 1 else None,
        )

        student_block = student.blocks[layer]

        loss, gradient = jax.value_and_grad(_direct_layer_loss)(
            shared_bank,
            student_block=student_block,
            mlp_input=mlp_input,
            teacher_output=teacher_output,
            normalization_epsilon=normalization_epsilon,
        )
        layer_losses.append(loss)
        gradients.append(gradient)
        routing_counts.append(student_trace.router_stats["routing_counts"])
        capacity_overflow.append(student_trace.router_stats["capacity_overflow"])
        student_selected = student_trace.selected_experts
        top1_agreement.append(jnp.mean(student_selected[:, 0] == teacher_selected[:, 0]))
        selected_ids_exact.append(selected_expert_ids_exact(student_selected, teacher_selected))
        combine_weight_max_abs_diff.append(
            jnp.max(
                jnp.abs(
                    student_trace.combine_weights.astype(jnp.float32)
                    - teacher_trace.routing.combine_weights.astype(jnp.float32)
                )
            )
        )

    return GradientConflictBatch(
        layer_losses=jnp.stack(layer_losses),
        gradients=(gradients[0], gradients[1]),
        routing_counts_by_layer=jnp.stack(routing_counts),
        capacity_overflow_by_layer=jnp.stack(capacity_overflow),
        router_top1_agreement_by_layer=jnp.stack(top1_agreement),
        router_selected_ids_exact_by_layer=jnp.stack(selected_ids_exact),
        combine_weight_max_abs_diff_by_layer=jnp.stack(combine_weight_max_abs_diff),
    )


def initial_gradient_conflict_accumulator(
    bank: MoEExpertMlp,
    *,
    num_batches: int,
    num_layers: int = 2,
) -> GradientConflictAccumulator:
    """Allocate device-side fp32 gradient and scalar accumulators."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    if num_layers != 2:
        raise ValueError("gradient-conflict accumulation requires exactly two layers")
    gradient_zeros = tuple(
        jax.tree.map(lambda leaf: jnp.zeros_like(leaf, dtype=jnp.float32), bank) for _ in range(num_layers)
    )
    return GradientConflictAccumulator(
        gradient_sums=(gradient_zeros[0], gradient_zeros[1]),
        batch_layer_losses=jnp.zeros((num_batches, num_layers), dtype=jnp.float32),
        batch_whole_metrics=jnp.zeros((num_batches, len(GRADIENT_PAIR_METRIC_NAMES)), dtype=jnp.float32),
        batch_projection_metrics=jnp.zeros(
            (num_batches, len(GRADIENT_PROJECTION_NAMES), len(GRADIENT_PAIR_METRIC_NAMES)),
            dtype=jnp.float32,
        ),
        routing_counts_by_layer=jnp.zeros((num_layers, bank.w_gate.shape[0]), dtype=jnp.float32),
        capacity_overflow_by_layer=jnp.zeros((num_layers,), dtype=jnp.float32),
        router_top1_agreement_by_layer=jnp.ones((num_layers,), dtype=jnp.float32),
        router_selected_ids_exact_by_layer=jnp.ones((num_layers,), dtype=jnp.float32),
        combine_weight_max_abs_diff_by_layer=jnp.zeros((num_layers,), dtype=jnp.float32),
    )


def accumulate_gradient_conflict_batch(
    accumulator: GradientConflictAccumulator,
    student: Transformer,
    teacher: Transformer,
    token_ids: jax.Array,
    mask: AttentionMask | jax.Array | None,
    batch_index: jax.Array,
    *,
    affected_layers: tuple[int, int],
    source_to_shared: tuple[int, ...] | None,
) -> GradientConflictAccumulator:
    """Accumulate one batch without exposing gradient tensors outside the compiled step."""
    batch = direct_shared_bank_gradients(
        student,
        teacher,
        token_ids,
        affected_layers=affected_layers,
        source_to_shared=source_to_shared,
        mask=mask,
    )
    gradient_sums = tuple(
        jax.tree.map(lambda total, gradient: total + gradient.astype(jnp.float32), total_tree, gradient_tree)
        for total_tree, gradient_tree in zip(accumulator.gradient_sums, batch.gradients, strict=True)
    )
    return GradientConflictAccumulator(
        gradient_sums=(gradient_sums[0], gradient_sums[1]),
        batch_layer_losses=accumulator.batch_layer_losses.at[batch_index].set(batch.layer_losses),
        batch_whole_metrics=accumulator.batch_whole_metrics.at[batch_index].set(gradient_pair_metrics(batch.gradients)),
        batch_projection_metrics=accumulator.batch_projection_metrics.at[batch_index].set(
            projection_gradient_pair_metrics(batch.gradients)
        ),
        routing_counts_by_layer=accumulator.routing_counts_by_layer + batch.routing_counts_by_layer,
        capacity_overflow_by_layer=accumulator.capacity_overflow_by_layer + batch.capacity_overflow_by_layer,
        router_top1_agreement_by_layer=jnp.minimum(
            accumulator.router_top1_agreement_by_layer,
            batch.router_top1_agreement_by_layer,
        ),
        router_selected_ids_exact_by_layer=jnp.minimum(
            accumulator.router_selected_ids_exact_by_layer,
            batch.router_selected_ids_exact_by_layer,
        ),
        combine_weight_max_abs_diff_by_layer=jnp.maximum(
            accumulator.combine_weight_max_abs_diff_by_layer,
            batch.combine_weight_max_abs_diff_by_layer,
        ),
    )


def finalize_gradient_conflict(
    accumulator: GradientConflictAccumulator,
    *,
    num_batches: int,
) -> GradientConflictResult:
    """Reduce accumulated gradients to scalar and per-expert summaries."""
    mean_gradients = tuple(
        jax.tree.map(lambda gradient: gradient / num_batches, gradient_sum) for gradient_sum in accumulator.gradient_sums
    )
    return GradientConflictResult(
        mean_layer_losses=jnp.mean(accumulator.batch_layer_losses, axis=0),
        aggregate_whole_metrics=gradient_pair_metrics((mean_gradients[0], mean_gradients[1])),
        aggregate_projection_metrics=projection_gradient_pair_metrics((mean_gradients[0], mean_gradients[1])),
        per_expert_metrics=per_expert_gradient_pair_metrics((mean_gradients[0], mean_gradients[1])),
        batch_layer_losses=accumulator.batch_layer_losses,
        batch_whole_metrics=accumulator.batch_whole_metrics,
        batch_projection_metrics=accumulator.batch_projection_metrics,
        routing_counts_by_layer=accumulator.routing_counts_by_layer,
        capacity_overflow_by_layer=accumulator.capacity_overflow_by_layer,
        router_top1_agreement_by_layer=accumulator.router_top1_agreement_by_layer,
        router_selected_ids_exact_by_layer=accumulator.router_selected_ids_exact_by_layer,
        combine_weight_max_abs_diff_by_layer=accumulator.combine_weight_max_abs_diff_by_layer,
    )
