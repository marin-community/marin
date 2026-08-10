# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a generated training program from an ordinary JAX-owned VJP."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import jax

from tile_lifetime.stablehlo_algebra_import import import_stablehlo_algebra
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_schedule import (
    RecoveredGenericStreamingAttentionTraining,
    select_streaming_attention_training_schedule,
)
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import StreamingAttentionBackwardProvenance

OPAQUE_FRONTEND_PRIMITIVES = frozenset({"custom_call", "ffi_call"})


class JaxAutomaticDifferentiationOwner(StrEnum):
    """Automatic-differentiation system that produced the recovered reverse."""

    JAX_VJP = "jax.vjp"


@dataclass(frozen=True)
class JaxVjpFrontendAudit:
    """Serializable provenance for one accepted natural JAX training graph."""

    source_kind: str
    source_function: str
    automatic_differentiation_owner: JaxAutomaticDifferentiationOwner
    jaxpr_sha256: str
    stablehlo_sha256: str
    recovered_provenance: StreamingAttentionBackwardProvenance
    source_operation_ids: tuple[int, ...]
    generic_algebra_operation_ids: tuple[int, ...]
    contract_operation_ids: tuple[int, ...]
    fold_operation_ids: tuple[int, ...]
    domain_restriction_operation_ids: tuple[int, ...]
    cast_and_view_operation_ids: tuple[int, ...]
    opaque_frontend_primitives: tuple[str, ...]
    workload_dispatch_key: None = None


@dataclass(frozen=True)
class RecoveredJaxVjpStreamingAttentionTraining:
    """Natural JAX source evidence plus its recovered generic program."""

    audit: JaxVjpFrontendAudit
    jaxpr: str
    stablehlo: bytes
    recovered: RecoveredGenericStreamingAttentionTraining


def recover_jax_vjp_streaming_attention_training(
    function: Callable[..., object],
    input_specifications: tuple[jax.ShapeDtypeStruct, ...],
    *,
    input_names: tuple[str, ...],
    schedule: StreamingTileSchedule,
) -> RecoveredJaxVjpStreamingAttentionTraining:
    """Export and recover visible algebra without accepting a workload dispatch key."""
    if not callable(function):
        raise TypeError("natural JAX training frontend requires a callable tensor program")
    if len(input_specifications) != len(input_names):
        raise ValueError("natural JAX training input specifications and names differ")

    closed_jaxpr = jax.make_jaxpr(function)(*input_specifications)
    primitive_names = _jaxpr_primitive_names(closed_jaxpr)
    opaque_primitives = tuple(sorted(OPAQUE_FRONTEND_PRIMITIVES.intersection(primitive_names)))
    if opaque_primitives:
        raise ValueError(
            "natural JAX training frontend must expose tensor algebra before recovery; "
            f"found opaque primitives {opaque_primitives}"
        )
    jaxpr = str(closed_jaxpr)
    stablehlo = jax.export.export(jax.jit(function))(*input_specifications).mlir_module_serialized
    graph = import_stablehlo(stablehlo, input_names=input_names)
    algebra = import_stablehlo_algebra(graph)
    recovered = select_streaming_attention_training_schedule(algebra, schedule=schedule)
    if recovered.program.provenance is not StreamingAttentionBackwardProvenance.JAX_VJP_GENERIC_ALGEBRA_IMPORT:
        raise ValueError(f"unexpected recovered provenance {recovered.program.provenance}")

    source_function = f"{function.__module__}.{function.__qualname__}"
    fold_operation_ids = (
        *recovered.normalized_exponential_fold_operation_ids,
        recovered.maximum_vjp_tie_fold_operation_id,
        *recovered.broadcast_vjp_fold_operation_ids,
    )
    audit = JaxVjpFrontendAudit(
        source_kind="ordinary_jax_tensor_program",
        source_function=source_function,
        automatic_differentiation_owner=JaxAutomaticDifferentiationOwner.JAX_VJP,
        jaxpr_sha256=hashlib.sha256(jaxpr.encode()).hexdigest(),
        stablehlo_sha256=hashlib.sha256(stablehlo).hexdigest(),
        recovered_provenance=recovered.program.provenance,
        source_operation_ids=recovered.source_operation_ids,
        generic_algebra_operation_ids=tuple(operation.source_operation_id for operation in recovered.algebra.operations),
        contract_operation_ids=recovered.contract_operation_ids,
        fold_operation_ids=fold_operation_ids,
        domain_restriction_operation_ids=recovered.domain_restriction_operation_ids,
        cast_and_view_operation_ids=recovered.cast_and_view_operation_ids,
        opaque_frontend_primitives=opaque_primitives,
    )
    return RecoveredJaxVjpStreamingAttentionTraining(audit, jaxpr, stablehlo, recovered)


def _jaxpr_primitive_names(closed_jaxpr: object) -> frozenset[str]:
    names: set[str] = set()
    visited: set[int] = set()

    def visit(value: object) -> None:
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        jaxpr = getattr(value, "jaxpr", None)
        if jaxpr is not None:
            visit(jaxpr)
        equations = getattr(value, "eqns", None)
        if equations is not None:
            for equation in equations:
                names.add(equation.primitive.name)
                visit(equation.params)
            return
        if isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                visit(nested)

    visit(closed_jaxpr)
    return frozenset(names)
