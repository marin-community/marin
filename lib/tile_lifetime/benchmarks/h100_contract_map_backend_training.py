# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural H100 Contract/Map backend candidates; execution stays gated."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tile_lifetime.contract_map_backend import (
    ContractMapBackendProgram,
    ContractMapNumericalPolicy,
    build_contract_map_backend_program,
    cubic_mix_expression,
    sigmoid_product_expression,
    tanh_product_expression,
)
from tile_lifetime.contract_map_backend_resources import (
    ContractMapLogicalBoundary,
    expected_contract_map_logical_boundary,
)
from tile_lifetime.cuda_contract_map_backend_codegen import (
    GeneratedCudaContractMapBackendFfi,
    generate_cuda_contract_map_backend_ffi,
)
from tile_lifetime.h100_contract_map_benchmark import (
    BackendVariant,
    MeasurementBoundary,
    ScalarMapFamily,
    StructuralCase,
    default_h100_contract_map_benchmark_plan,
)
from tile_lifetime.tensor_program import ScalarExpression


@dataclass(frozen=True)
class GeneratedContractMapCandidate:
    """One anonymous structural case compiled under one Shuttle policy."""

    case: StructuralCase
    backend: BackendVariant
    program: ContractMapBackendProgram
    generated: GeneratedCudaContractMapBackendFfi
    boundaries: tuple[tuple[MeasurementBoundary, ContractMapLogicalBoundary], ...]


def generated_contract_map_candidates() -> tuple[GeneratedContractMapCandidate, ...]:
    """Generate both Shuttle variants for every reviewed odd-row case."""
    candidates: list[GeneratedContractMapCandidate] = []
    for case in default_h100_contract_map_benchmark_plan().cases:
        expression = _scalar_expression(case.scalar_map)
        for backend, policy in (
            (BackendVariant.SHUTTLE_SOURCE_ORDERED, ContractMapNumericalPolicy.SOURCE_ORDERED),
            (BackendVariant.SHUTTLE_FAST, ContractMapNumericalPolicy.FAST),
        ):
            program = build_contract_map_backend_program(
                rows=case.rows,
                reduction=case.reduction,
                features=case.features,
                scalar_expression=expression,
                numerical_policy=policy,
            )
            generated = generate_cuda_contract_map_backend_ffi(program)
            boundaries = tuple(
                (
                    boundary,
                    expected_contract_map_logical_boundary(
                        generated,
                        kernel_only=boundary is MeasurementBoundary.KERNEL_ONLY,
                    ),
                )
                for boundary in MeasurementBoundary
            )
            candidates.append(
                GeneratedContractMapCandidate(
                    case=case,
                    backend=backend,
                    program=program,
                    generated=generated,
                    boundaries=boundaries,
                )
            )
    return tuple(candidates)


def natural_jax_training_step(
    scalar_map: ScalarMapFamily,
    activation: jax.Array,
    first_weight: jax.Array,
    second_weight: jax.Array,
    output_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the ordinary-JAX forward and JAX-owned VJP oracle."""

    def forward(x: jax.Array, w0: jax.Array, w1: jax.Array) -> jax.Array:
        preactivation = jnp.matmul(x, w0, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        scalar = preactivation.astype(jnp.float32)
        if scalar_map is ScalarMapFamily.SIGMOID_PRODUCT:
            hidden = (scalar * jax.nn.sigmoid(scalar)).astype(jnp.bfloat16)
        elif scalar_map is ScalarMapFamily.TANH_PRODUCT:
            hidden = (scalar * jnp.tanh(scalar)).astype(jnp.bfloat16)
        elif scalar_map is ScalarMapFamily.CUBIC_MIX:
            square = scalar * scalar
            hidden = (scalar + square * scalar).astype(jnp.bfloat16)
        else:
            raise ValueError(f"unsupported structural scalar Map {scalar_map!r}")
        return jnp.matmul(hidden, w1, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    output, pullback = jax.vjp(forward, activation, first_weight, second_weight)
    input_adjoint, first_weight_adjoint, second_weight_adjoint = pullback(output_cotangent)
    return output, input_adjoint, first_weight_adjoint, second_weight_adjoint


def _scalar_expression(scalar_map: ScalarMapFamily) -> ScalarExpression:
    return {
        ScalarMapFamily.SIGMOID_PRODUCT: sigmoid_product_expression,
        ScalarMapFamily.TANH_PRODUCT: tanh_product_expression,
        ScalarMapFamily.CUBIC_MIX: cubic_mix_expression,
    }[scalar_map]()
