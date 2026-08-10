# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose Torch-free typed-FFI families around exact relation metadata."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.expert_parallel_training_runtime import (
    DistributedExpertBackwardABI,
    derive_distributed_expert_backward_abi,
)
from tile_lifetime.relation import RelationPlan
from tile_lifetime.xla_relation_edge_reverse_ffi import (
    GeneratedRelationEdgeReverseFfi,
    RelationEdgeReverseFfiPlan,
    generate_cuda_relation_edge_reverse_ffi,
)
from tile_lifetime.xla_routed_input_adjoint_ffi import generate_cuda_routed_input_adjoint_ffi
from tile_lifetime.xla_routed_weight_gradient_ffi import generate_cuda_group_batched_contract_ffi
from tile_lifetime.xla_source_indexed_fold_ffi import generate_cuda_source_indexed_fold_ffi


class DistributedBackwardHandlerFamily(StrEnum):
    """Existing generic typed-FFI family assigned to a reverse stage."""

    RELATION_EDGE_REVERSE = "relation_edge_reverse"
    ROUTED_INPUT_ADJOINT = "routed_input_adjoint"
    GROUP_BATCHED_CONTRACT = "group_batched_contract"
    SOURCE_INDEXED_FOLD = "source_indexed_fold"


@dataclass(frozen=True)
class TypedFfiHandlerFamilyBinding:
    """Generic typed-FFI generator and its rank-local data-domain contract."""

    family: DistributedBackwardHandlerFamily
    generator: str
    input_domain: str
    output_domain: str


@dataclass(frozen=True)
class RankRelationReverseMetadata:
    """One rank's exact coalesced-dispatch and edge-return index plane."""

    rank: int
    exchange_source_item: np.ndarray
    route_padded_rows: np.ndarray
    route_weights: np.ndarray
    route_source_item: np.ndarray
    route_source_slot: np.ndarray
    route_edge_identity: np.ndarray
    route_valid: np.ndarray
    global_padded_rows: np.ndarray
    input_adjoint_source_index: np.ndarray


@dataclass(frozen=True)
class RankDistributedBackwardTypedFfi:
    """Rank-local generated edge handler and reusable Contract/Fold families."""

    metadata: RankRelationReverseMetadata
    edge_reverse_plan: RelationEdgeReverseFfiPlan
    edge_reverse: GeneratedRelationEdgeReverseFfi
    input_adjoint: TypedFfiHandlerFamilyBinding
    weight_adjoint: tuple[TypedFfiHandlerFamilyBinding, TypedFfiHandlerFamilyBinding]
    source_fold: TypedFfiHandlerFamilyBinding


@dataclass(frozen=True)
class DistributedExpertBackwardTypedFfiComposition:
    """Torch-free rank handlers with JAX-owned transport and router reverse."""

    abi: DistributedExpertBackwardABI
    ranks: tuple[RankDistributedBackwardTypedFfi, ...]
    relation_digest: str
    transport: str = "JAX collective payload permutation"
    router_vjp: str = "JAX-owned normalized-selection and router Contract VJP"
    runtime_dependencies: tuple[str, ...] = ("JAX/XLA typed FFI", "CUDA runtime", "cuBLAS")


def compose_distributed_expert_backward_typed_ffi(
    relation: RelationPlan,
    *,
    hidden: int,
    intermediate: int,
    target_prefix: str,
) -> DistributedExpertBackwardTypedFfiComposition:
    """Bind exact RelationPlan arrays to existing generic typed-FFI families."""
    abi = derive_distributed_expert_backward_abi(relation, hidden=hidden, intermediate=intermediate)
    ranks = tuple(
        _compose_rank(relation, rank=rank, hidden=hidden, target_prefix=target_prefix)
        for rank in range(relation.destination_rank_count)
    )
    composition = DistributedExpertBackwardTypedFfiComposition(
        abi=abi,
        ranks=ranks,
        relation_digest=_relation_digest(relation),
    )
    verify_distributed_expert_backward_typed_ffi(relation, composition)
    return composition


def verify_distributed_expert_backward_typed_ffi(
    relation: RelationPlan,
    composition: DistributedExpertBackwardTypedFfiComposition,
) -> None:
    """Verify exact edge coverage, ownership, and Torch-free source boundaries."""
    if len(composition.ranks) != relation.destination_rank_count:
        raise ValueError("typed-FFI composition rank count disagrees with RelationPlan")
    seen_routes: list[np.ndarray] = []
    for expected_rank, rank_plan in enumerate(composition.ranks):
        metadata = rank_plan.metadata
        if metadata.rank != expected_rank:
            raise ValueError("rank metadata identity is inconsistent")
        if not np.all(relation.row_destination_rank[metadata.global_padded_rows] == metadata.rank):
            raise ValueError(f"rank {metadata.rank} contains padded rows owned by another rank")
        valid_source = metadata.route_source_item[metadata.route_valid]
        valid_slot = metadata.route_source_slot[metadata.route_valid]
        flat_routes = valid_source * relation.route_slots + valid_slot
        if not np.array_equal(metadata.route_edge_identity[metadata.route_valid], flat_routes):
            raise ValueError(f"rank {metadata.rank} edge identities disagree with source/slot coordinates")
        seen_routes.append(flat_routes)
        if np.unique(flat_routes).size != flat_routes.size:
            raise ValueError(f"rank {metadata.rank} duplicates a logical relation edge")
        lowered = rank_plan.edge_reverse.source.lower()
        if any(token in lowered for token in ("torch", "pybind", "at::tensor", "deep_ep", "mok")):
            raise ValueError(f"rank {metadata.rank} edge reverse source is not Torch-free")
        expected_received = metadata.exchange_source_item.shape[0]
        expected_shape = (expected_received, relation.route_slots)
        rectangular_metadata = (
            metadata.route_padded_rows,
            metadata.route_weights,
            metadata.route_source_item,
            metadata.route_source_slot,
            metadata.route_edge_identity,
            metadata.route_valid,
        )
        if any(value.shape != expected_shape for value in rectangular_metadata):
            raise ValueError(f"rank {metadata.rank} edge metadata has inconsistent rectangular domains")
        if not np.array_equal(
            metadata.input_adjoint_source_index,
            np.arange(metadata.global_padded_rows.size, dtype=np.int32),
        ):
            raise ValueError(f"rank {metadata.rank} input adjoint must preserve one value per padded relation row")
        expected_generators = {
            _qualified_name(generate_cuda_routed_input_adjoint_ffi),
            _qualified_name(generate_cuda_group_batched_contract_ffi),
            _qualified_name(generate_cuda_source_indexed_fold_ffi),
        }
        observed_generators = {
            rank_plan.input_adjoint.generator,
            *(binding.generator for binding in rank_plan.weight_adjoint),
            rank_plan.source_fold.generator,
        }
        if observed_generators != expected_generators:
            raise ValueError(f"rank {metadata.rank} does not bind the approved generic typed-FFI generator families")
    observed = np.sort(np.concatenate(seen_routes))
    expected = np.flatnonzero(relation.edge_valid.reshape(-1))
    if not np.array_equal(observed, expected):
        raise ValueError("typed-FFI rank metadata does not cover every valid relation edge exactly once")
    if composition.runtime_dependencies != ("JAX/XLA typed FFI", "CUDA runtime", "cuBLAS"):
        raise ValueError("typed-FFI composition introduced an unapproved runtime dependency")


def restore_source_route_payload(
    relation: RelationPlan,
    composition: DistributedExpertBackwardTypedFfiComposition,
    rank_payloads: tuple[np.ndarray, ...],
    *,
    fill_value: int | float = 0,
) -> np.ndarray:
    """Return rank-local edge payloads to exact source-item, route-slot order."""
    if len(rank_payloads) != len(composition.ranks):
        raise ValueError("one returned payload is required per rank")
    trailing_shape = rank_payloads[0].shape[2:]
    output = np.full(
        (relation.source_item_count, relation.route_slots, *trailing_shape),
        fill_value,
        dtype=rank_payloads[0].dtype,
    )
    for rank_plan, payload in zip(composition.ranks, rank_payloads, strict=True):
        metadata = rank_plan.metadata
        expected_shape = (*metadata.route_valid.shape, *trailing_shape)
        if payload.shape != expected_shape:
            raise ValueError(f"rank {metadata.rank} returned payload shape {payload.shape} != {expected_shape}")
        output[
            metadata.route_source_item[metadata.route_valid],
            metadata.route_source_slot[metadata.route_valid],
        ] = payload[metadata.route_valid]
    return output


def deterministic_source_slot_fold(
    relation: RelationPlan,
    source_route_payload: np.ndarray,
) -> np.ndarray:
    """Fold returned vectors in source route-slot order without atomics."""
    if source_route_payload.shape[:2] != (relation.source_item_count, relation.route_slots):
        raise ValueError("source route payload disagrees with RelationPlan domain")
    output = np.zeros(
        (relation.source_item_count, *source_route_payload.shape[2:]),
        dtype=np.result_type(source_route_payload.dtype, np.float32),
    )
    for source_item in range(relation.source_item_count):
        for route_slot in range(relation.route_slots):
            if relation.edge_valid[source_item, route_slot]:
                output[source_item] += source_route_payload[source_item, route_slot]
    return output


def source_indexed_fold_operands(
    relation: RelationPlan,
    source_route_payload: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten valid returned slots into deterministic source-indexed Fold operands."""
    if source_route_payload.shape[:2] != (relation.source_item_count, relation.route_slots):
        raise ValueError("source route payload disagrees with RelationPlan domain")
    valid = relation.edge_valid.reshape(-1)
    source_indices = np.repeat(
        np.arange(relation.source_item_count, dtype=np.int32),
        relation.route_slots,
    )[valid]
    contributions = source_route_payload.reshape(-1, *source_route_payload.shape[2:])[valid]
    return source_indices[:, None], contributions


def _compose_rank(
    relation: RelationPlan,
    *,
    rank: int,
    hidden: int,
    target_prefix: str,
) -> RankDistributedBackwardTypedFfi:
    global_padded_rows = np.flatnonzero(relation.row_destination_rank == rank)
    if global_padded_rows.size == 0:
        raise ValueError(f"rank {rank} has no padded relation rows")
    global_to_local = np.full(relation.destination_row_count, -1, dtype=np.int32)
    global_to_local[global_padded_rows] = np.arange(global_padded_rows.size, dtype=np.int32)
    exchange_rows = np.flatnonzero(relation.exchange_destination_rank == rank)
    if exchange_rows.size == 0:
        raise ValueError(f"rank {rank} has no received exchange rows")
    exchange_source_item = relation.exchange_source_item[exchange_rows]
    exchange_local_by_global = np.full(relation.exchange_source_item.shape[0], -1, dtype=np.int32)
    exchange_local_by_global[exchange_rows] = np.arange(exchange_rows.size, dtype=np.int32)
    shape = (exchange_rows.size, relation.route_slots)
    route_padded_rows = np.full(shape, -1, dtype=np.int32)
    route_weights = np.zeros(shape, dtype=np.float32)
    route_source_item = np.full(shape, -1, dtype=np.int32)
    route_source_slot = np.full(shape, -1, dtype=np.int32)
    route_edge_identity = np.full(shape, -1, dtype=np.int32)
    route_valid = np.zeros(shape, dtype=np.bool_)
    for flat_route in np.flatnonzero(relation.edge_valid.reshape(-1)):
        if relation.destination_rank[flat_route] != rank:
            continue
        exchange_row = exchange_local_by_global[relation.route_to_exchange_row[flat_route]]
        route_slot = int(relation.route_slot[flat_route])
        global_padded_row = relation.route_to_destination_row[flat_route]
        route_padded_rows[exchange_row, route_slot] = global_to_local[global_padded_row]
        route_weights[exchange_row, route_slot] = relation.weight.reshape(-1)[flat_route]
        route_source_item[exchange_row, route_slot] = relation.source_item[flat_route]
        route_source_slot[exchange_row, route_slot] = route_slot
        route_edge_identity[exchange_row, route_slot] = flat_route
        route_valid[exchange_row, route_slot] = True
    metadata = RankRelationReverseMetadata(
        rank=rank,
        exchange_source_item=exchange_source_item,
        route_padded_rows=route_padded_rows,
        route_weights=route_weights,
        route_source_item=route_source_item,
        route_source_slot=route_source_slot,
        route_edge_identity=route_edge_identity,
        route_valid=route_valid,
        global_padded_rows=global_padded_rows,
        input_adjoint_source_index=np.arange(global_padded_rows.size, dtype=np.int32),
    )
    edge_plan = RelationEdgeReverseFfiPlan(
        received_rows=exchange_rows.size,
        route_slots=relation.route_slots,
        padded_rows=global_padded_rows.size,
        features=hidden,
    )
    return RankDistributedBackwardTypedFfi(
        metadata=metadata,
        edge_reverse_plan=edge_plan,
        edge_reverse=generate_cuda_relation_edge_reverse_ffi(
            edge_plan,
            target=f"{target_prefix}.rank{rank}.edge_reverse",
        ),
        input_adjoint=TypedFfiHandlerFamilyBinding(
            family=DistributedBackwardHandlerFamily.ROUTED_INPUT_ADJOINT,
            generator=_qualified_name(generate_cuda_routed_input_adjoint_ffi),
            input_domain="padded relation rows with identity source indices",
            output_domain="one input-cotangent payload per padded relation row",
        ),
        weight_adjoint=(
            TypedFfiHandlerFamilyBinding(
                family=DistributedBackwardHandlerFamily.GROUP_BATCHED_CONTRACT,
                generator=_qualified_name(generate_cuda_group_batched_contract_ffi),
                input_domain="destination-group padded rows",
                output_domain="local group-batched W2 weight cotangent",
            ),
            TypedFfiHandlerFamilyBinding(
                family=DistributedBackwardHandlerFamily.GROUP_BATCHED_CONTRACT,
                generator=_qualified_name(generate_cuda_group_batched_contract_ffi),
                input_domain="destination-group padded rows",
                output_domain="local group-batched W13 weight cotangent",
            ),
        ),
        source_fold=TypedFfiHandlerFamilyBinding(
            family=DistributedBackwardHandlerFamily.SOURCE_INDEXED_FOLD,
            generator=_qualified_name(generate_cuda_source_indexed_fold_ffi),
            input_domain="returned valid relation edges in source-item/route-slot order",
            output_domain="one deterministic input cotangent per source item",
        ),
    )


def _relation_digest(relation: RelationPlan) -> str:
    digest = hashlib.sha256()
    for array in (
        relation.edge_valid,
        relation.destination_item,
        relation.destination_rank,
        relation.route_to_destination_row,
        relation.exchange_source_item,
        relation.exchange_destination_rank,
        relation.route_to_exchange_row,
        relation.weight,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _qualified_name(value: object) -> str:
    return f"{value.__module__}.{value.__name__}"
