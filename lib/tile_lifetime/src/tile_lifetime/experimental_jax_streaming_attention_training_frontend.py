# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental attention-pattern recovery from ordinary JAX tensor algebra.

JAX owns the exported source VJP. The current selector uses that graph for
structural validation, then regenerates the executable reverse with Shuttle's
symbolic reference VJP. This module is not an accepted plugin frontend.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import jax

from tile_lifetime.experimental_stablehlo_streaming_schedule import (
    ExperimentalRecoveredStreamingAttentionTraining,
    select_experimental_streaming_attention_training_schedule,
)
from tile_lifetime.stablehlo_algebra_import import import_stablehlo_algebra
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import StreamingAttentionBackwardProvenance

OPAQUE_FRONTEND_PRIMITIVES = frozenset({"custom_call", "ffi_call"})


class JaxAutomaticDifferentiationOwner(StrEnum):
    """Automatic-differentiation system that produced the source reverse."""

    JAX_VJP = "jax.vjp"


class GeneratedReverseOwner(StrEnum):
    """System that constructs the executable reverse representation."""

    SHUTTLE_REFERENCE_SYMBOLIC_VJP = "shuttle.reference_symbolic_vjp"


@dataclass(frozen=True)
class ExperimentalJaxVjpFrontendAudit:
    """Serializable provenance for one unaccepted diagnostic graph."""

    source_kind: str
    source_function: str
    source_automatic_differentiation_owner: JaxAutomaticDifferentiationOwner
    generated_reverse_owner: GeneratedReverseOwner
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
class ExperimentalRecoveredJaxVjpStreamingAttentionTraining:
    """Natural JAX source evidence plus a regenerated diagnostic program."""

    audit: ExperimentalJaxVjpFrontendAudit
    jaxpr: str
    stablehlo: bytes
    recovered: ExperimentalRecoveredStreamingAttentionTraining


def recover_experimental_jax_vjp_streaming_attention_training(
    function: Callable[..., object],
    input_specifications: tuple[jax.ShapeDtypeStruct, ...],
    *,
    input_names: tuple[str, ...],
    schedule: StreamingTileSchedule,
) -> ExperimentalRecoveredJaxVjpStreamingAttentionTraining:
    """Export visible algebra and run the experimental attention selector."""
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
    recovered = select_experimental_streaming_attention_training_schedule(algebra, schedule=schedule)
    if (
        recovered.program.provenance
        is not StreamingAttentionBackwardProvenance.EXPERIMENTAL_REGENERATED_REVERSE_FROM_JAX_VJP_ALGEBRA
    ):
        raise ValueError(f"unexpected recovered provenance {recovered.program.provenance}")

    source_function = f"{function.__module__}.{function.__qualname__}"
    fold_operation_ids = (
        *recovered.normalized_exponential_fold_operation_ids,
        recovered.maximum_vjp_tie_fold_operation_id,
        *recovered.broadcast_vjp_fold_operation_ids,
    )
    audit = ExperimentalJaxVjpFrontendAudit(
        source_kind="ordinary_jax_tensor_program_unaccepted_diagnostic",
        source_function=source_function,
        source_automatic_differentiation_owner=JaxAutomaticDifferentiationOwner.JAX_VJP,
        generated_reverse_owner=GeneratedReverseOwner.SHUTTLE_REFERENCE_SYMBOLIC_VJP,
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
    return ExperimentalRecoveredJaxVjpStreamingAttentionTraining(audit, jaxpr, stablehlo, recovered)


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
