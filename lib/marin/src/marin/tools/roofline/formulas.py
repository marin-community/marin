# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Formula-level estimates for Grug/MoE semantic roofline rows."""

from __future__ import annotations

from dataclasses import dataclass

from marin.tools.roofline.model_spec import ModelSpec
from marin.tools.roofline.types import RowKind

TRAINING_FORWARD_EQUIVALENTS_WITH_REMAT = 4.0
TRAINING_FORWARD_EQUIVALENTS_NO_REMAT = 3.0


@dataclass(frozen=True)
class FormulaEstimate:
    semantic_op: str
    kind: RowKind
    flops: float
    bytes_accessed: float
    formula: str


def newton_schulz_iteration_flops(rows: int, cols: int) -> float:
    m = min(rows, cols)
    n = max(rows, cols)
    return float(4 * m * m * n + 2 * m * m * m)


def newton_schulz_total_flops(rows: int, cols: int, ns_iters: int) -> float:
    return float(ns_iters) * newton_schulz_iteration_flops(rows, cols)


def newton_schulz_part_flops(rows: int, cols: int) -> dict[str, float]:
    m = min(rows, cols)
    n = max(rows, cols)
    return {
        "gram": float(2 * m * m * n),
        "polynomial": float(2 * m * m * m),
        "apply": float(2 * m * m * n),
    }


def fused_expert_muon_ns_flops(model: ModelSpec) -> float:
    gate_up = newton_schulz_total_flops(model.hidden_dim, 2 * model.intermediate_dim, model.muon_ns_iters)
    down = newton_schulz_total_flops(model.intermediate_dim, model.hidden_dim, model.muon_ns_iters)
    return float(model.num_layers * model.num_experts) * (gate_up + down)


def unfused_expert_muon_ns_flops(model: ModelSpec) -> float:
    one_matrix = newton_schulz_total_flops(model.intermediate_dim, model.hidden_dim, model.muon_ns_iters)
    return float(model.num_layers * model.num_experts * 3) * one_matrix


def fused_expert_muon_ns_part_flops(model: ModelSpec, part: str) -> float:
    gate_up = newton_schulz_part_flops(model.hidden_dim, 2 * model.intermediate_dim)[part]
    down = newton_schulz_part_flops(model.intermediate_dim, model.hidden_dim)[part]
    return float(model.num_layers * model.num_experts * model.muon_ns_iters) * (gate_up + down)


def expert_weight_elements(model: ModelSpec) -> float:
    return float(model.num_layers * model.num_experts * model.hidden_dim * model.intermediate_dim * 3)


def muon_hyperball_flops(model: ModelSpec) -> float:
    return 13.0 * expert_weight_elements(model)


def attention_training_flops(model: ModelSpec) -> float:
    forward_flops = 4.0 * model.global_batch_tokens * model.hidden_dim * attention_context_tokens(model)
    return training_forward_equivalents(model) * forward_flops


def long_attention_layers(model: ModelSpec) -> int:
    if model.long_attention_every is None:
        return 0
    return model.num_layers // model.long_attention_every


def attention_context_tokens(model: ModelSpec) -> int:
    long_layers = long_attention_layers(model)
    short_layers = model.num_layers - long_layers
    short_window = model.sliding_window // 2
    return short_layers * short_window + long_layers * model.sliding_window


def moe_activation_collective_bytes(model: ModelSpec) -> float:
    return (
        training_forward_equivalents(model)
        * model.num_layers
        * model.global_batch_tokens
        * model.hidden_dim
        * model.top_k
        * 2.0
    )


def expert_shard_devices(model: ModelSpec) -> int:
    return model.mesh.replica_dcn * model.mesh.expert * model.mesh.model


def local_activation_shard_devices(model: ModelSpec) -> int:
    return model.mesh.data * model.mesh.expert * model.mesh.model


def per_local_activation_device(value: float, model: ModelSpec) -> float:
    return value / float(local_activation_shard_devices(model))


def per_expert_shard_device(value: float, model: ModelSpec) -> float:
    return value / float(expert_shard_devices(model))


def ring_collective_wire_factor(group_size: int) -> float:
    if group_size <= 1:
        return 0.0
    return float(group_size - 1) / float(group_size)


def expert_shard_ring_collective_bytes(value: float, model: ModelSpec) -> float:
    return per_expert_shard_device(value, model) * ring_collective_wire_factor(expert_shard_devices(model))


def training_forward_equivalents(model: ModelSpec) -> float:
    if model.remat == "recompute_all":
        return TRAINING_FORWARD_EQUIVALENTS_WITH_REMAT
    return TRAINING_FORWARD_EQUIVALENTS_NO_REMAT


def formula_estimates(model: ModelSpec) -> list[FormulaEstimate]:
    token_count = float(model.global_batch_tokens)
    dense_projection_flops = 2.0 * token_count * model.hidden_dim * model.vocab_size
    attention_flops = attention_training_flops(model)
    moe_gemm_flops = (
        training_forward_equivalents(model)
        * 6.0
        * model.num_layers
        * token_count
        * model.top_k
        * model.hidden_dim
        * model.intermediate_dim
    )
    vector_bytes = token_count * model.hidden_dim * 2.0 * 12.0
    expert_weight_bytes = model.num_layers * model.num_experts * model.hidden_dim * model.intermediate_dim * 6.0

    return [
        FormulaEstimate(
            semantic_op="token_embedding_output_projection",
            kind=RowKind.COMPUTE,
            flops=per_local_activation_device(dense_projection_flops, model),
            bytes_accessed=per_local_activation_device(token_count * model.hidden_dim * 2.0, model),
            formula="2 * local_tokens * hidden_dim * vocab_size / local_activation_shard_devices",
        ),
        FormulaEstimate(
            semantic_op="attention_fa4",
            kind=RowKind.COMPUTE,
            flops=per_local_activation_device(attention_flops, model),
            bytes_accessed=per_local_activation_device(token_count * model.hidden_dim * 8.0, model),
            formula=(
                "4 * local_tokens * num_heads * head_dim * "
                "[(layers - long_attention_layers) * sliding_window/2 + "
                "long_attention_layers * sliding_window] * "
                "attention_train_forward_equivalents(remat) / local_activation_shard_devices"
            ),
        ),
        FormulaEstimate(
            semantic_op="moe_expert",
            kind=RowKind.COMPUTE,
            flops=per_local_activation_device(moe_gemm_flops, model),
            bytes_accessed=per_expert_shard_device(expert_weight_bytes, model),
            formula=(
                "6 * layers * local_tokens * top_k * hidden_dim * intermediate_dim * "
                "training_forward_equivalents(remat) / local_activation_shard_devices"
            ),
        ),
        FormulaEstimate(
            semantic_op="xent",
            kind=RowKind.COMPUTE,
            flops=per_local_activation_device(4.0 * token_count * model.vocab_size, model),
            bytes_accessed=per_local_activation_device(token_count * model.vocab_size * 2.0, model),
            formula="4 * local_tokens * vocab_size / local_activation_shard_devices",
        ),
        FormulaEstimate(
            semantic_op="muon_ns_gram",
            kind=RowKind.COMPUTE,
            flops=per_expert_shard_device(fused_expert_muon_ns_part_flops(model, "gram"), model),
            bytes_accessed=per_expert_shard_device(expert_weight_bytes, model),
            formula="L * E * ns_iters * [2*m(D,2I)^2*n(D,2I) + 2*m(I,D)^2*n(I,D)] / expert_shard_devices",
        ),
        FormulaEstimate(
            semantic_op="muon_ns_polynomial",
            kind=RowKind.COMPUTE,
            flops=per_expert_shard_device(fused_expert_muon_ns_part_flops(model, "polynomial"), model),
            bytes_accessed=per_expert_shard_device(expert_weight_bytes, model),
            formula="L * E * ns_iters * [2*m(D,2I)^3 + 2*m(I,D)^3] / expert_shard_devices",
        ),
        FormulaEstimate(
            semantic_op="muon_ns_apply",
            kind=RowKind.COMPUTE,
            flops=per_expert_shard_device(fused_expert_muon_ns_part_flops(model, "apply"), model),
            bytes_accessed=per_expert_shard_device(expert_weight_bytes, model),
            formula="L * E * ns_iters * [2*m(D,2I)^2*n(D,2I) + 2*m(I,D)^2*n(I,D)] / expert_shard_devices",
        ),
        FormulaEstimate(
            semantic_op="muon_hyperball",
            kind=RowKind.COMPUTE,
            flops=per_expert_shard_device(muon_hyperball_flops(model), model),
            bytes_accessed=per_expert_shard_device(expert_weight_bytes, model),
            formula=(
                "13 * L * E * hidden_dim * intermediate_dim * 3 / expert_shard_devices; "
                "lower-bound vector/reduction estimate for MuonH hyperball projection"
            ),
        ),
        FormulaEstimate(
            semantic_op="optimizer_vector_ops",
            kind=RowKind.COMPUTE,
            flops=per_local_activation_device(8.0 * token_count * model.hidden_dim, model),
            bytes_accessed=per_local_activation_device(vector_bytes, model),
            formula="8 * local_tokens * hidden_dim / local_activation_shard_devices",
        ),
        FormulaEstimate(
            semantic_op="fsdp_param_all_gather",
            kind=RowKind.COMM,
            flops=0.0,
            bytes_accessed=per_expert_shard_device(expert_weight_bytes, model),
            formula="expert parameter bytes / expert_shard_devices, first-order communication floor",
        ),
        FormulaEstimate(
            semantic_op="grad_reduce_scatter",
            kind=RowKind.COMM,
            flops=0.0,
            bytes_accessed=per_expert_shard_device(expert_weight_bytes, model),
            formula="expert gradient bytes / expert_shard_devices, first-order communication floor",
        ),
        FormulaEstimate(
            semantic_op="grouped_muon_restore",
            kind=RowKind.COMM,
            flops=0.0,
            bytes_accessed=expert_shard_ring_collective_bytes(expert_weight_bytes, model),
            formula=(
                "expert optimizer-bank restore bytes / expert_shard_devices * "
                "(expert_shard_devices - 1) / expert_shard_devices"
            ),
        ),
        FormulaEstimate(
            semantic_op="expert_all_to_all",
            kind=RowKind.COMM,
            flops=0.0,
            bytes_accessed=per_local_activation_device(moe_activation_collective_bytes(model), model),
            formula=(
                "layers * local_tokens * hidden_dim * top_k * bf16_bytes * "
                "training_forward_equivalents(remat) / local_activation_shard_devices"
            ),
        ),
    ]
