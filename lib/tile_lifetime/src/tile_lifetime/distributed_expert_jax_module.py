# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan a fixed-capacity JAX-owned distributed routed reverse module."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.distributed_expert_backward_ffi import (
    DistributedExpertBackwardTypedFfiComposition,
    compose_distributed_expert_backward_typed_ffi,
)
from tile_lifetime.expert_parallel_training import ExpertParallelTrainingPlan
from tile_lifetime.expert_parallel_training_runtime import execute_distributed_expert_backward_reference
from tile_lifetime.jax_relation_edge_reverse_ffi import call_cuda_relation_edge_reverse_ffi
from tile_lifetime.jax_routed_reverse_ffi import (
    call_cuda_group_batched_contract_ffi,
    call_cuda_segmented_input_adjoint_ffi,
    call_cuda_source_indexed_fold_ffi,
)
from tile_lifetime.relation import RelationPlan, build_fixed_capacity_relation_plan
from tile_lifetime.xla_hlo_recovery import EntryRegionValue
from tile_lifetime.xla_relation_program_recovery import (
    RoutedInputAdjointTypedFfiCodegenPlan,
    RoutedWeightGradientFfiOperand,
    RoutedWeightGradientFfiOperandRole,
    RoutedWeightGradientTypedFfiCodegenPlan,
)
from tile_lifetime.xla_routed_weight_gradient_ffi import (
    GeneratedGroupBatchedContractFfi,
    generate_cuda_group_batched_contract_ffi,
)
from tile_lifetime.xla_segmented_input_adjoint_ffi import (
    GeneratedSegmentedInputAdjointFfi,
    SegmentedInputAdjointFfiPlan,
    SegmentedInputAdjointResourceAudit,
    audit_segmented_input_adjoint_resources,
    generate_cuda_segmented_input_adjoint_ffi,
    plan_segmented_input_adjoint_ffi,
)
from tile_lifetime.xla_source_indexed_fold_ffi import (
    GeneratedSourceIndexedFoldFfi,
    SourceIndexedFoldTypedFfiPlan,
    generate_cuda_source_indexed_fold_ffi,
)


@dataclass(frozen=True)
class DistributedExpertJaxModuleConfig:
    """Static dimensions shared by natural JAX and generated handlers."""

    source_items_per_rank: int
    hidden: int
    intermediate: int


@dataclass(frozen=True)
class JaxPayloadCollectiveBoundary:
    """One payload-only placement transition retained in JAX."""

    name: str
    direction: str
    payload_shape_per_rank: tuple[int, ...]
    mechanism: str = "jax.lax.all_to_all"
    semantics: str = "payload_permutation_only"


@dataclass(frozen=True)
class InputAdjointWeightAbi:
    """Forward storage and generated reverse Contract weight layouts."""

    stored_down: tuple[int, int, int]
    generated_down_input_adjoint: tuple[int, int, int]
    stored_gate_up: tuple[int, int, int]
    generated_gate_up_input_adjoint: tuple[int, int, int]
    transformation: str = "transpose each forward Contract weight within each segment, then concatenate gate/up"


@dataclass(frozen=True)
class RelationReturnMetadata:
    """Rank-local padded rows mapped back to original source-route edges."""

    source_item: np.ndarray
    route_slot: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class GeneratedDistributedExpertHandlers:
    """Instantiated generic reverse handler sources for one rank shape."""

    relation_edge_target: str
    input_adjoint_plan: SegmentedInputAdjointFfiPlan
    input_adjoint: GeneratedSegmentedInputAdjointFfi
    weight_gradient_plans: tuple[RoutedWeightGradientTypedFfiCodegenPlan, RoutedWeightGradientTypedFfiCodegenPlan]
    weight_gradients: tuple[GeneratedGroupBatchedContractFfi, GeneratedGroupBatchedContractFfi]
    source_fold_plan: SourceIndexedFoldTypedFfiPlan
    source_fold: GeneratedSourceIndexedFoldFfi


@dataclass(frozen=True)
class DistributedExpertJaxModulePlan:
    """One transformed natural-JAX boundary with JAX-owned AD and transport."""

    relation: RelationPlan
    composition: DistributedExpertBackwardTypedFfiComposition
    config: DistributedExpertJaxModuleConfig
    local_expert_count: int
    destination_capacity: int
    handlers: GeneratedDistributedExpertHandlers
    input_adjoint_weight_abi: InputAdjointWeightAbi
    input_adjoint_resources: SegmentedInputAdjointResourceAudit
    collectives: tuple[JaxPayloadCollectiveBoundary, ...]
    ad_owner: str = "JAX VJP over ordinary router Contract, top-k values, and normalized route weights"
    runtime_dependencies: tuple[str, ...] = ("JAX/XLA typed FFI", "CUDA runtime", "cuBLAS")


@dataclass(frozen=True)
class DistributedExpertTrainingResult:
    """Natural output and gradients in natural JAX storage layouts."""

    output: np.ndarray
    input_cotangent: np.ndarray
    router_weight_cotangent: np.ndarray
    gate_weight_cotangent: np.ndarray
    up_weight_cotangent: np.ndarray
    down_weight_cotangent: np.ndarray


@dataclass(frozen=True)
class NumericalErrorMetrics:
    """Pointwise absolute, relative, and BF16-representable deviation."""

    maximum_absolute: float
    mean_absolute: float
    maximum_relative: float
    mean_relative: float
    maximum_bf16_ulp: int


def compare_numerical_arrays(expected: np.ndarray, actual: np.ndarray) -> NumericalErrorMetrics:
    """Report pointwise error without hiding near-zero or BF16-scale deviations."""
    expected_f32 = np.asarray(expected, dtype=np.float32)
    actual_f32 = np.asarray(actual, dtype=np.float32)
    if expected_f32.shape != actual_f32.shape:
        raise ValueError(f"comparison shape mismatch: {expected_f32.shape} != {actual_f32.shape}")
    absolute = np.abs(actual_f32 - expected_f32)
    relative = absolute / np.maximum(np.abs(expected_f32), np.finfo(np.float32).tiny)
    expected_bf16 = np.asarray(jnp.asarray(expected_f32, dtype=jnp.bfloat16), dtype=np.float32)
    actual_bf16 = np.asarray(jnp.asarray(actual_f32, dtype=jnp.bfloat16), dtype=np.float32)
    expected_ordered = _ordered_bf16(expected_bf16)
    actual_ordered = _ordered_bf16(actual_bf16)
    return NumericalErrorMetrics(
        maximum_absolute=float(np.max(absolute, initial=0.0)),
        mean_absolute=float(np.mean(absolute)) if absolute.size else 0.0,
        maximum_relative=float(np.max(relative, initial=0.0)),
        mean_relative=float(np.mean(relative)) if relative.size else 0.0,
        maximum_bf16_ulp=int(np.max(np.abs(actual_ordered - expected_ordered), initial=0)),
    )


def _ordered_bf16(values: np.ndarray) -> np.ndarray:
    bits = (np.asarray(values, dtype=np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    magnitude = (bits & np.uint16(0x7FFF)).astype(np.int64)
    return np.where(bits & np.uint16(0x8000), 0x8000 - magnitude, 0x8000 + magnitude)


def plan_distributed_expert_jax_module(
    relation: RelationPlan,
    *,
    config: DistributedExpertJaxModuleConfig,
    input_adjoint_template: RoutedInputAdjointTypedFfiCodegenPlan,
    weight_gradient_templates: tuple[RoutedWeightGradientTypedFfiCodegenPlan, RoutedWeightGradientTypedFfiCodegenPlan],
    source_fold_template: SourceIndexedFoldTypedFfiPlan,
    target_prefix: str,
) -> DistributedExpertJaxModulePlan:
    """Instantiate every generic reverse family at a routing-independent shape."""
    if relation.source_item_count != config.source_items_per_rank * relation.destination_rank_count:
        raise ValueError("global source domain must partition evenly across JAX expert ranks")
    local_experts, remainder = divmod(relation.destination_count, relation.destination_rank_count)
    if remainder:
        raise ValueError("destination groups must partition evenly across JAX expert ranks")
    capacities = np.unique(relation.group_padded_count)
    if capacities.size != 1:
        raise ValueError("distributed JAX module requires one fixed capacity for every destination")
    destination_capacity = int(capacities[0])
    composition = compose_distributed_expert_backward_typed_ffi(
        relation,
        hidden=config.hidden,
        intermediate=config.intermediate,
        target_prefix=f"{target_prefix}.edge",
    )
    edge_shapes = {rank.edge_reverse_plan for rank in composition.ranks}
    if len(edge_shapes) != 1:
        raise ValueError("fixed-capacity relation produced rank-dependent edge handler shapes")
    relation_edge = composition.ranks[0].edge_reverse
    specialized_input = plan_segmented_input_adjoint_ffi(
        input_adjoint_template,
        segment_count=local_experts,
        capacity=destination_capacity,
        input_features=config.hidden,
        intermediate_features=config.intermediate,
    )
    specialized_weights = (
        _specialize_weight_gradient(
            weight_gradient_templates[0],
            groups=local_experts,
            reduction=destination_capacity,
            lhs_features=config.hidden,
            rhs_features=2 * config.intermediate,
        ),
        _specialize_weight_gradient(
            weight_gradient_templates[1],
            groups=local_experts,
            reduction=destination_capacity,
            lhs_features=config.intermediate,
            rhs_features=config.hidden,
        ),
    )
    specialized_source_fold = _specialize_source_fold(
        source_fold_template,
        sources=config.source_items_per_rank,
        edges=config.source_items_per_rank * relation.route_slots,
        features=config.hidden,
    )
    handlers = GeneratedDistributedExpertHandlers(
        relation_edge_target=relation_edge.target,
        input_adjoint_plan=specialized_input,
        input_adjoint=generate_cuda_segmented_input_adjoint_ffi(
            specialized_input,
            target=f"{target_prefix}.input_adjoint",
        ),
        weight_gradient_plans=(specialized_weights[0], specialized_weights[1]),
        weight_gradients=(
            generate_cuda_group_batched_contract_ffi(
                specialized_weights[0],
                target=f"{target_prefix}.weight_gradient.w13",
            ),
            generate_cuda_group_batched_contract_ffi(
                specialized_weights[1],
                target=f"{target_prefix}.weight_gradient.w2",
            ),
        ),
        source_fold_plan=specialized_source_fold,
        source_fold=generate_cuda_source_indexed_fold_ffi(
            specialized_source_fold,
            target=f"{target_prefix}.source_fold",
        ),
    )
    global_sources = relation.source_item_count
    collectives = (
        JaxPayloadCollectiveBoundary(
            "output_adjoint_transport",
            "source owners -> destination owners",
            (relation.destination_rank_count, config.source_items_per_rank, config.hidden),
        ),
        JaxPayloadCollectiveBoundary(
            "input_adjoint_return_transport",
            "destination owners -> source owners",
            (
                relation.destination_rank_count,
                config.source_items_per_rank * relation.route_slots,
                config.hidden,
            ),
        ),
        JaxPayloadCollectiveBoundary(
            "route_weight_return_transport",
            "destination owners -> source owners",
            (global_sources, relation.route_slots),
        ),
    )
    plan = DistributedExpertJaxModulePlan(
        relation=relation,
        composition=composition,
        config=config,
        local_expert_count=local_experts,
        destination_capacity=destination_capacity,
        handlers=handlers,
        input_adjoint_weight_abi=InputAdjointWeightAbi(
            stored_down=(local_experts, config.intermediate, config.hidden),
            generated_down_input_adjoint=(local_experts, config.hidden, config.intermediate),
            stored_gate_up=(local_experts, config.hidden, config.intermediate),
            generated_gate_up_input_adjoint=(local_experts, 2 * config.intermediate, config.hidden),
        ),
        input_adjoint_resources=audit_segmented_input_adjoint_resources(specialized_input),
        collectives=collectives,
    )
    verify_distributed_expert_jax_module(plan)
    return plan


def verify_distributed_expert_jax_module(plan: DistributedExpertJaxModulePlan) -> None:
    """Reject semantic transport, shape drift, and opaque runtime dependencies."""
    if any(boundary.semantics != "payload_permutation_only" for boundary in plan.collectives):
        raise ValueError("JAX transport may not perform a semantic combine")
    if plan.runtime_dependencies != ("JAX/XLA typed FFI", "CUDA runtime", "cuBLAS"):
        raise ValueError("distributed JAX module introduced an unapproved runtime dependency")
    sources = (
        *(rank.edge_reverse.source for rank in plan.composition.ranks),
        plan.handlers.input_adjoint.source,
        *(generated.source for generated in plan.handlers.weight_gradients),
        plan.handlers.source_fold.source,
    )
    forbidden = ("torch", "pybind", "at::tensor", "deep_ep", "mok")
    if any(token in source.lower() for source in sources for token in forbidden):
        raise ValueError("generated distributed module contains an opaque or Torch runtime dependency")
    if any("atomicadd(" in source.lower() for source in sources):
        raise ValueError("generated distributed reverse may not use semantic atomic accumulation")
    edge_shapes = {rank.edge_reverse_plan for rank in plan.composition.ranks}
    if len(edge_shapes) != 1:
        raise ValueError("rank-local relation edge handlers do not share one fixed ABI")
    if len({rank.edge_reverse.semantic_digest for rank in plan.composition.ranks}) != 1:
        raise ValueError("fixed rank handlers do not share one generic semantic program")


def build_relation_return_metadata(plan: DistributedExpertJaxModulePlan) -> RelationReturnMetadata:
    """Invert each rank's runtime route-to-padded-row mapping without changing edge identity."""
    rank_count = plan.relation.destination_rank_count
    local_rows = plan.local_expert_count * plan.destination_capacity
    source_item = np.full((rank_count, local_rows), -1, dtype=np.int32)
    route_slot = np.full((rank_count, local_rows), -1, dtype=np.int32)
    valid = np.zeros((rank_count, local_rows), dtype=np.bool_)
    source_grid = np.broadcast_to(
        np.arange(plan.relation.source_item_count, dtype=np.int32)[:, None],
        (plan.relation.source_item_count, plan.relation.route_slots),
    )
    slot_grid = np.broadcast_to(
        np.arange(plan.relation.route_slots, dtype=np.int32)[None, :],
        (plan.relation.source_item_count, plan.relation.route_slots),
    )
    for rank in plan.composition.ranks:
        padded_rows = rank.metadata.route_padded_rows
        present = padded_rows >= 0
        rows = padded_rows[present]
        if np.unique(rows).size != rows.size:
            raise ValueError(f"rank {rank.metadata.rank} maps multiple relation edges to one padded row")
        source_item[rank.metadata.rank, rows] = source_grid[present]
        route_slot[rank.metadata.rank, rows] = slot_grid[present]
        valid[rank.metadata.rank, rows] = True
    edge_identity = source_item[valid] * plan.relation.route_slots + route_slot[valid]
    if not np.array_equal(np.sort(edge_identity), np.arange(plan.relation.route_count, dtype=np.int32)):
        raise ValueError("rank-local return rows do not preserve every relation edge exactly once")
    return RelationReturnMetadata(source_item=source_item, route_slot=route_slot, valid=valid)


def build_natural_router_relation(
    source: jax.Array,
    router_weight: jax.Array,
    *,
    route_slots: int,
    destination_rank_by_item: np.ndarray,
    destination_local_item_by_item: np.ndarray,
    destination_capacity: int,
) -> RelationPlan:
    """Run ordinary JAX router algebra, then materialize its bounded index plane."""
    logits = source.astype(jnp.float32) @ router_weight.astype(jnp.float32)
    selected_logits, route_indices = jax.lax.top_k(logits, route_slots)
    route_weights = jax.nn.softmax(selected_logits, axis=1)
    return build_fixed_capacity_relation_plan(
        np.asarray(route_indices, dtype=np.int32),
        np.asarray(route_weights, dtype=np.float32),
        destination_rank_by_item=destination_rank_by_item,
        destination_local_item_by_item=destination_local_item_by_item,
        destination_capacity=destination_capacity,
    )


def natural_distributed_expert_program(
    source: jax.Array,
    router_weight: jax.Array,
    gate_weight: jax.Array,
    up_weight: jax.Array,
    down_weight: jax.Array,
    *,
    route_slots: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Express routed expert semantics with only ordinary JAX tensor algebra."""
    logits = source.astype(jnp.float32) @ router_weight.astype(jnp.float32)
    selected_logits, route_indices = jax.lax.top_k(logits, route_slots)
    route_weights = jax.nn.softmax(selected_logits, axis=1)
    selected_gate = gate_weight[route_indices]
    selected_up = up_weight[route_indices]
    gate = jnp.einsum("sh,skhi->ski", source, selected_gate)
    up = jnp.einsum("sh,skhi->ski", source, selected_up)
    hidden = jax.nn.silu(gate) * up
    selected_down = down_weight[route_indices]
    edge_output = jnp.einsum("ski,skih->skh", hidden, selected_down)
    output = jnp.sum(edge_output.astype(jnp.float32) * route_weights[..., None], axis=1)
    return output.astype(jnp.bfloat16), route_indices, route_weights


def evaluate_natural_jax_training(
    source: jax.Array,
    router_weight: jax.Array,
    gate_weight: jax.Array,
    up_weight: jax.Array,
    down_weight: jax.Array,
    output_cotangent: jax.Array,
    *,
    route_slots: int,
) -> DistributedExpertTrainingResult:
    """Use JAX AD as the independent whole-program reference."""

    def output_only(x, router, gate, up, down):
        output, _, _ = natural_distributed_expert_program(
            x,
            router,
            gate,
            up,
            down,
            route_slots=route_slots,
        )
        return output

    output, pullback = jax.vjp(output_only, source, router_weight, gate_weight, up_weight, down_weight)
    gradients = pullback(output_cotangent)
    return DistributedExpertTrainingResult(
        output=np.asarray(output),
        input_cotangent=np.asarray(gradients[0]),
        router_weight_cotangent=np.asarray(gradients[1]),
        gate_weight_cotangent=np.asarray(gradients[2]),
        up_weight_cotangent=np.asarray(gradients[3]),
        down_weight_cotangent=np.asarray(gradients[4]),
    )


def evaluate_decomposed_training_reference(
    plan: DistributedExpertJaxModulePlan,
    training_plan: ExpertParallelTrainingPlan,
    source: jax.Array,
    router_weight: jax.Array,
    gate_weight: jax.Array,
    up_weight: jax.Array,
    down_weight: jax.Array,
    output_cotangent: jax.Array,
) -> DistributedExpertTrainingResult:
    """Execute generated-stage algebra while leaving the router pullback in JAX."""
    gate_up_contract_weight = np.concatenate(
        (
            np.swapaxes(np.asarray(gate_weight, dtype=np.float32), -1, -2),
            np.swapaxes(np.asarray(up_weight, dtype=np.float32), -1, -2),
        ),
        axis=1,
    )
    down_contract_weight = np.swapaxes(np.asarray(down_weight, dtype=np.float32), -1, -2)
    expert_result = execute_distributed_expert_backward_reference(
        plan.relation,
        np.asarray(source, dtype=np.float32),
        gate_up_contract_weight,
        down_contract_weight,
        np.asarray(output_cotangent, dtype=np.float32),
        training_plan,
    )
    route_indices = jnp.asarray(
        plan.relation.destination_item.reshape(plan.relation.source_item_count, plan.relation.route_slots)
    )
    router_input_cotangent, router_weight_cotangent = router_vjp(
        source,
        router_weight,
        route_indices,
        jnp.asarray(expert_result.route_weight_cotangent),
    )
    gate_up_cotangent = expert_result.gate_up_weight_cotangent
    gate_contract_cotangent, up_contract_cotangent = np.split(
        gate_up_cotangent,
        (plan.config.intermediate,),
        axis=1,
    )
    return DistributedExpertTrainingResult(
        output=expert_result.output,
        input_cotangent=expert_result.input_cotangent + np.asarray(router_input_cotangent),
        router_weight_cotangent=np.asarray(router_weight_cotangent),
        gate_weight_cotangent=np.swapaxes(gate_contract_cotangent, -1, -2),
        up_weight_cotangent=np.swapaxes(up_contract_cotangent, -1, -2),
        down_weight_cotangent=np.swapaxes(expert_result.down_weight_cotangent, -1, -2),
    )


def prepare_input_adjoint_weights(
    down_weight: jax.Array,
    gate_weight: jax.Array,
    up_weight: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Transpose natural forward weights into generic reverse Contract operands."""
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate and up forward Contract weights must share one layout")
    down_rhs = jnp.swapaxes(down_weight, -1, -2)
    gate_rhs = jnp.swapaxes(gate_weight, -1, -2)
    up_rhs = jnp.swapaxes(up_weight, -1, -2)
    gate_up_rhs = jnp.concatenate((gate_rhs, up_rhs), axis=1)
    return down_rhs, gate_up_rhs


def rebind_pair_weight_cotangent(
    pair_weight_cotangent: jax.Array,
    *,
    intermediate_features: int,
) -> tuple[jax.Array, jax.Array]:
    """Split a generic concatenated pair-Map Contract result into natural parameter storage."""
    if pair_weight_cotangent.ndim != 3 or pair_weight_cotangent.shape[-1] != 2 * intermediate_features:
        raise ValueError("pair weight cotangent does not match the natural two-operand Map storage")
    return (
        pair_weight_cotangent[..., :intermediate_features],
        pair_weight_cotangent[..., intermediate_features:],
    )


def selected_route_weights(source: jax.Array, router_weight: jax.Array, route_indices: jax.Array) -> jax.Array:
    """Natural router algebra whose reverse remains owned by JAX."""
    logits = source.astype(jnp.float32) @ router_weight.astype(jnp.float32)
    selected = jnp.take_along_axis(logits, route_indices, axis=1)
    return jax.nn.softmax(selected, axis=1)


def router_vjp(
    source: jax.Array,
    router_weight: jax.Array,
    route_indices: jax.Array,
    route_weight_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Differentiate the ordinary selected-weight program with JAX."""
    _, pullback = jax.vjp(lambda x, weight: selected_route_weights(x, weight, route_indices), source, router_weight)
    return pullback(route_weight_cotangent)


def jax_payload_all_to_all(payload: jax.Array, *, axis_name: str) -> jax.Array:
    """Move a fixed-capacity payload without applying semantic accumulation."""
    return jax.lax.all_to_all(
        payload,
        axis_name,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )


def lower_handler_module_stablehlo(plan: DistributedExpertJaxModulePlan) -> str:
    """Lower all generated handlers plus JAX router VJP without compiling CUDA."""
    transformed = _local_handler_function(plan)
    return str(jax.jit(transformed).lower(*_local_handler_arguments(plan)).compiler_ir(dialect="stablehlo"))


def audit_handler_module_stablehlo(plan: DistributedExpertJaxModulePlan, stablehlo: str) -> dict[str, int]:
    """Require one occurrence of every generic family and no opaque kernel."""
    targets = (
        plan.handlers.relation_edge_target,
        plan.handlers.input_adjoint.target,
        *(handler.target for handler in plan.handlers.weight_gradients),
        plan.handlers.source_fold.target,
    )
    occurrences = {target: stablehlo.count(f"@{target}") for target in targets}
    if any(count != 1 for count in occurrences.values()):
        raise ValueError(f"transformed handler occurrences are not one-to-one: {occurrences}")
    lowered = stablehlo.lower()
    if any(token in lowered for token in ("torch", "deep_ep", "mok", "flash_attention")):
        raise ValueError("transformed module contains an opaque semantic custom call")
    if "stablehlo.top_k" in lowered:
        raise ValueError("router selection must precede the transformed reverse boundary")
    if "stablehlo.dot_general" not in lowered:
        raise ValueError("JAX router VJP algebra is absent from the transformed module")
    return occurrences


def lower_shard_mapped_handler_module_stablehlo(
    plan: DistributedExpertJaxModulePlan,
    *,
    mesh: jax.sharding.Mesh,
) -> str:
    """Lower handlers, payload return, source Fold, and router VJP together."""
    rank_count = plan.relation.destination_rank_count
    if mesh.shape != {"expert": rank_count}:
        raise ValueError(f"expert mesh {mesh.shape} != fixed relation rank count {rank_count}")
    destination_handlers = _local_destination_handler_function(plan)
    source_finalize = _local_source_finalize_function(plan)
    local_sources = plan.config.source_items_per_rank
    route_slots = plan.relation.route_slots
    local_experts = plan.local_expert_count
    local_edges = local_sources * route_slots
    hidden = plan.config.hidden

    def local(
        output_cotangent_send,
        route_padded_rows,
        route_weights,
        saved_edge_output,
        row_valid,
        saved_pair,
        padded_source,
        saved_hidden,
        down_weight,
        gate_weight,
        up_weight,
        row_source_item,
        row_route_slot,
        source,
        router_weight,
        route_indices,
    ):
        received_cotangent = jax_payload_all_to_all(output_cotangent_send[0], axis_name="expert").reshape(
            plan.relation.source_item_count,
            hidden,
        )
        edge_input_cotangent, route_weight_cotangent, w13_cotangent, w2_cotangent = destination_handlers(
            received_cotangent,
            route_padded_rows[0],
            route_weights[0],
            saved_edge_output[0],
            row_valid[0],
            saved_pair[0],
            padded_source[0],
            saved_hidden[0],
            down_weight[0],
            gate_weight[0],
            up_weight[0],
        )
        valid = row_valid[0]
        source_rank = jnp.where(valid, row_source_item[0] // local_sources, rank_count)
        source_edge = jnp.where(
            valid,
            (row_source_item[0] % local_sources) * route_slots + row_route_slot[0],
            local_edges,
        )
        packed_input = (
            jnp.zeros((rank_count, local_edges, hidden), dtype=jnp.bfloat16)
            .at[
                source_rank,
                source_edge,
            ]
            .set(edge_input_cotangent, mode="drop")
        )
        received_input = jax_payload_all_to_all(packed_input, axis_name="expert")
        packed_route = route_weight_cotangent.reshape(rank_count, local_sources, route_slots)
        received_route = jax_payload_all_to_all(packed_route, axis_name="expert")
        destination_rank = route_indices[0] // local_experts
        flat_destination = destination_rank.reshape(-1)
        returned_input_edges = received_input[flat_destination, jnp.arange(local_edges, dtype=jnp.int32)]
        returned_route_cotangent = jnp.take_along_axis(
            jnp.transpose(received_route, (1, 2, 0)),
            destination_rank[..., None],
            axis=2,
        )[..., 0]
        input_cotangent, router_weight_cotangent = source_finalize(
            returned_input_edges,
            returned_route_cotangent,
            source[0],
            router_weight,
            route_indices[0],
        )
        gate_cotangent, up_cotangent = rebind_pair_weight_cotangent(
            w13_cotangent,
            intermediate_features=plan.config.intermediate,
        )
        return (
            input_cotangent[None],
            jax.lax.psum(router_weight_cotangent, "expert"),
            gate_cotangent[None],
            up_cotangent[None],
            w2_cotangent[None],
        )

    in_specs = (
        jax.sharding.PartitionSpec("expert", None, None, None),
        jax.sharding.PartitionSpec("expert", None, None),
        jax.sharding.PartitionSpec("expert", None, None),
        jax.sharding.PartitionSpec("expert", None, None),
        jax.sharding.PartitionSpec("expert", None),
        jax.sharding.PartitionSpec("expert", None, None),
        jax.sharding.PartitionSpec("expert", None, None, None),
        jax.sharding.PartitionSpec("expert", None, None, None),
        jax.sharding.PartitionSpec("expert", None, None, None),
        jax.sharding.PartitionSpec("expert", None, None, None),
        jax.sharding.PartitionSpec("expert", None, None, None),
        jax.sharding.PartitionSpec("expert", None),
        jax.sharding.PartitionSpec("expert", None),
        jax.sharding.PartitionSpec("expert", None, None),
        jax.sharding.PartitionSpec(None, None),
        jax.sharding.PartitionSpec("expert", None, None),
    )
    out_specs = (
        jax.sharding.PartitionSpec("expert", None, None),
        jax.sharding.PartitionSpec(None, None),
        jax.sharding.PartitionSpec("expert", None, None, None),
        jax.sharding.PartitionSpec("expert", None, None, None),
        jax.sharding.PartitionSpec("expert", None, None, None),
    )
    transformed = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    return str(jax.jit(transformed).lower(*_shard_mapped_handler_arguments(plan)).compiler_ir(dialect="stablehlo"))


def audit_shard_mapped_handler_module_stablehlo(
    plan: DistributedExpertJaxModulePlan,
    stablehlo: str,
) -> dict[str, object]:
    """Audit the single handler/collective graph before accelerator use."""
    handlers = audit_handler_module_stablehlo(plan, stablehlo)
    all_to_all_count = stablehlo.count("stablehlo.all_to_all")
    all_reduce_count = stablehlo.count("stablehlo.all_reduce")
    if all_to_all_count != 3:
        raise ValueError(f"integrated graph has {all_to_all_count} all-to-all operations, expected 3")
    if all_reduce_count != 1:
        raise ValueError(f"integrated graph has {all_reduce_count} all-reduce operations, expected 1")
    return {
        "handlers": handlers,
        "all_to_all_count": all_to_all_count,
        "all_reduce_count": all_reduce_count,
    }


def _local_handler_function(plan: DistributedExpertJaxModulePlan):
    destination_handlers = _local_destination_handler_function(plan)
    source_finalize = _local_source_finalize_function(plan)

    def transformed(*arguments):
        destination = destination_handlers(*arguments[:11])
        input_cotangent, router_weight_cotangent = source_finalize(*arguments[11:])
        gate_cotangent, up_cotangent = rebind_pair_weight_cotangent(
            destination[2],
            intermediate_features=plan.config.intermediate,
        )
        return (
            input_cotangent,
            router_weight_cotangent,
            destination[0],
            gate_cotangent,
            up_cotangent,
            destination[3],
        )

    return transformed


def _local_destination_handler_function(plan: DistributedExpertJaxModulePlan):
    rank = plan.composition.ranks[0]
    handlers = plan.handlers
    local_rows = rank.edge_reverse_plan.padded_rows
    local_experts = plan.local_expert_count
    capacity = plan.destination_capacity
    hidden = plan.config.hidden
    intermediate = plan.config.intermediate

    def destination_handlers(
        received_cotangent,
        route_padded_rows,
        route_weights,
        saved_edge_output,
        row_valid,
        saved_pair,
        padded_source,
        saved_hidden,
        down_weight,
        gate_weight,
        up_weight,
    ):
        padded_cotangent, route_weight_cotangent = call_cuda_relation_edge_reverse_ffi(
            rank.edge_reverse,
            rank.edge_reverse_plan,
            received_cotangent,
            route_padded_rows,
            route_weights,
            saved_edge_output,
        )
        down_rhs, gate_up_rhs = prepare_input_adjoint_weights(down_weight, gate_weight, up_weight)
        pair_cotangent, segmented_input_cotangent = call_cuda_segmented_input_adjoint_ffi(
            handlers.input_adjoint,
            handlers.input_adjoint_plan,
            padded_cotangent.reshape(local_experts, capacity, hidden),
            saved_pair.reshape(local_experts, capacity, 2 * intermediate),
            row_valid.reshape(local_experts, capacity),
            down_rhs,
            gate_up_rhs,
        )
        w13_cotangent = call_cuda_group_batched_contract_ffi(
            handlers.weight_gradients[0],
            handlers.weight_gradient_plans[0],
            padded_source,
            pair_cotangent,
        )
        w2_cotangent = call_cuda_group_batched_contract_ffi(
            handlers.weight_gradients[1],
            handlers.weight_gradient_plans[1],
            saved_hidden,
            padded_cotangent.reshape(local_experts, capacity, hidden),
        )
        return (
            segmented_input_cotangent.reshape(local_rows, hidden),
            route_weight_cotangent,
            w13_cotangent,
            w2_cotangent,
        )

    return destination_handlers


def _local_source_finalize_function(plan: DistributedExpertJaxModulePlan):
    handlers = plan.handlers
    sources = plan.config.source_items_per_rank
    hidden = plan.config.hidden
    route_slots = plan.relation.route_slots

    def source_finalize(
        returned_input_edges,
        returned_route_cotangent,
        source,
        router_weight,
        route_indices,
    ):
        source_indices = jnp.repeat(jnp.arange(sources, dtype=jnp.int32), route_slots)[:, None]
        input_cotangent = call_cuda_source_indexed_fold_ffi(
            handlers.source_fold,
            handlers.source_fold_plan,
            jnp.zeros((sources, hidden), dtype=jnp.bfloat16),
            source_indices,
            returned_input_edges,
        )
        router_input_cotangent, router_weight_cotangent = router_vjp(
            source,
            router_weight,
            route_indices,
            returned_route_cotangent,
        )
        return input_cotangent + router_input_cotangent.astype(jnp.bfloat16), router_weight_cotangent

    return source_finalize


def _local_handler_arguments(plan: DistributedExpertJaxModulePlan) -> tuple[jax.Array, ...]:
    rank = plan.composition.ranks[0]
    local_rows = rank.edge_reverse_plan.padded_rows
    local_experts = plan.local_expert_count
    capacity = plan.destination_capacity
    hidden = plan.config.hidden
    intermediate = plan.config.intermediate
    sources = plan.config.source_items_per_rank
    route_slots = plan.relation.route_slots
    received_rows = rank.edge_reverse_plan.received_rows
    return (
        jnp.zeros((received_rows, hidden), dtype=jnp.bfloat16),
        jnp.zeros((received_rows, route_slots), dtype=jnp.int32),
        jnp.zeros((received_rows, route_slots), dtype=jnp.float32),
        jnp.zeros((local_rows, hidden), dtype=jnp.bfloat16),
        jnp.zeros((local_rows,), dtype=jnp.bool_),
        jnp.zeros((local_rows, 2 * intermediate), dtype=jnp.bfloat16),
        jnp.zeros((local_experts, capacity, hidden), dtype=jnp.bfloat16),
        jnp.zeros((local_experts, capacity, intermediate), dtype=jnp.bfloat16),
        jnp.zeros((local_experts, intermediate, hidden), dtype=jnp.bfloat16),
        jnp.zeros((local_experts, hidden, intermediate), dtype=jnp.bfloat16),
        jnp.zeros((local_experts, hidden, intermediate), dtype=jnp.bfloat16),
        jnp.zeros((sources * route_slots, hidden), dtype=jnp.bfloat16),
        jnp.zeros((sources, route_slots), dtype=jnp.float32),
        jnp.zeros((sources, hidden), dtype=jnp.bfloat16),
        jnp.zeros((hidden, plan.relation.destination_count), dtype=jnp.bfloat16),
        jnp.zeros((sources, route_slots), dtype=jnp.int32),
    )


def _shard_mapped_handler_arguments(plan: DistributedExpertJaxModulePlan) -> tuple[jax.Array, ...]:
    rank_count = plan.relation.destination_rank_count
    local_rows = plan.composition.ranks[0].edge_reverse_plan.padded_rows
    local_experts = plan.local_expert_count
    capacity = plan.destination_capacity
    hidden = plan.config.hidden
    intermediate = plan.config.intermediate
    sources = plan.config.source_items_per_rank
    route_slots = plan.relation.route_slots
    return_metadata = build_relation_return_metadata(plan)
    route_indices = plan.relation.destination_item.reshape(rank_count, sources, route_slots)
    return (
        jnp.zeros((rank_count, rank_count, sources, hidden), dtype=jnp.bfloat16),
        jnp.asarray(np.stack(tuple(rank.metadata.route_padded_rows for rank in plan.composition.ranks))),
        jnp.asarray(np.stack(tuple(rank.metadata.route_weights for rank in plan.composition.ranks))),
        jnp.zeros((rank_count, local_rows, hidden), dtype=jnp.bfloat16),
        jnp.asarray(return_metadata.valid),
        jnp.zeros((rank_count, local_rows, 2 * intermediate), dtype=jnp.bfloat16),
        jnp.zeros((rank_count, local_experts, capacity, hidden), dtype=jnp.bfloat16),
        jnp.zeros((rank_count, local_experts, capacity, intermediate), dtype=jnp.bfloat16),
        jnp.zeros((rank_count, local_experts, intermediate, hidden), dtype=jnp.bfloat16),
        jnp.zeros((rank_count, local_experts, hidden, intermediate), dtype=jnp.bfloat16),
        jnp.zeros((rank_count, local_experts, hidden, intermediate), dtype=jnp.bfloat16),
        jnp.asarray(return_metadata.source_item),
        jnp.asarray(return_metadata.route_slot),
        jnp.zeros((rank_count, sources, hidden), dtype=jnp.bfloat16),
        jnp.zeros((hidden, plan.relation.destination_count), dtype=jnp.bfloat16),
        jnp.asarray(route_indices),
    )


def _specialize_weight_gradient(
    template: RoutedWeightGradientTypedFfiCodegenPlan,
    *,
    groups: int,
    reduction: int,
    lhs_features: int,
    rhs_features: int,
) -> RoutedWeightGradientTypedFfiCodegenPlan:
    lhs = RoutedWeightGradientFfiOperand(
        RoutedWeightGradientFfiOperandRole.LHS,
        EntryRegionValue("lhs", f"bf16[{groups},{reduction},{lhs_features}]{{2,1,0}}"),
        template.operands[0].parameter_ancestors,
    )
    rhs = RoutedWeightGradientFfiOperand(
        RoutedWeightGradientFfiOperandRole.RHS,
        EntryRegionValue("rhs", f"bf16[{groups},{reduction},{rhs_features}]{{2,1,0}}"),
        template.operands[1].parameter_ancestors,
    )
    contract = replace(template.contract, output_shape=f"bf16[{groups},{lhs_features},{rhs_features}]{{2,1,0}}")
    return replace(
        template,
        region=replace(template.region, contract=contract),
        contract=contract,
        operands=(lhs, rhs),
    )


def _specialize_source_fold(
    template: SourceIndexedFoldTypedFfiPlan,
    *,
    sources: int,
    edges: int,
    features: int,
) -> SourceIndexedFoldTypedFfiPlan:
    output_shape = f"bf16[{sources},{features}]{{1,0}}"
    return replace(
        template,
        initial=EntryRegionValue("fold_initial", output_shape),
        source_indices=EntryRegionValue("source_indices", f"s32[{edges},1]{{1,0}}"),
        contributions=EntryRegionValue("returned_input_edges", f"bf16[{edges},{features}]{{1,0}}"),
        contribution_wrappers=(),
        output_shape=output_shape,
        external_users=(),
    )
