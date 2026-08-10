# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from shuttle.ir import DType
from tile_lifetime import (
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    TransformSkeleton,
)
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.ir import TensorGraph
from tile_lifetime.plan import AttachmentSite, NumericalEquivalence
from tile_lifetime.swiglu import compile_reference_swiglu_region

TOKENS = 8
HIDDEN = 16
INTERMEDIATE = 24


def _separate_swiglu_region(*, observe_gate: bool = False) -> TensorGraph:
    graph = TensorGraph()
    x = graph.input("x", shape=(TOKENS, HIDDEN), dtype=DType.BF16)
    gate_weight = graph.parameter("gate_weight", shape=(HIDDEN, INTERMEDIATE), dtype=DType.BF16)
    up_weight = graph.parameter("up_weight", shape=(HIDDEN, INTERMEDIATE), dtype=DType.BF16)
    down_weight = graph.parameter("down_weight", shape=(INTERMEDIATE, HIDDEN), dtype=DType.BF16)

    gate = graph.linear(x, gate_weight, name="gate", accumulation_dtype=DType.FP32)
    up = graph.linear(x, up_weight, name="up", accumulation_dtype=DType.FP32)
    activated = graph.swiglu(gate, up, name="activated")
    graph.linear(activated, down_weight, name="output", accumulation_dtype=DType.FP32)
    if observe_gate:
        residual = graph.input("gate_residual", shape=gate.shape, dtype=gate.dtype)
        graph.residual_add(gate, residual, name="observed_gate")
    return graph


def _combined_swiglu_region() -> TensorGraph:
    graph = TensorGraph()
    x = graph.input("x", shape=(TOKENS, HIDDEN), dtype=DType.BF16)
    gate_up_weight = graph.parameter(
        "gate_up_weight",
        shape=(HIDDEN, 2 * INTERMEDIATE),
        dtype=DType.BF16,
    )
    down_weight = graph.parameter("down_weight", shape=(INTERMEDIATE, HIDDEN), dtype=DType.BF16)

    gate_up = graph.linear(x, gate_up_weight, name="gate_up", accumulation_dtype=DType.FP32)
    activated = graph.pairwise_swiglu(gate_up, name="activated")
    graph.linear(activated, down_weight, name="output", accumulation_dtype=DType.FP32)
    return graph


def test_separate_gate_up_projections_fuse_pairwise_swiglu() -> None:
    plan = compile_reference_swiglu_region(
        _separate_swiglu_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert [type(skeleton) for skeleton in plan.skeletons] == [GemmSkeleton, GemmSkeleton]
    gate_up = plan.skeletons[0]
    assert isinstance(gate_up, GemmSkeleton)
    assert gate_up.shape == (TOKENS, 2 * INTERMEDIATE, HIDDEN)
    assert gate_up.backend == GENERIC_H100_GEMM_BACKEND
    assert gate_up.output_layout == "row_major_mn_pair_reduced"
    assert gate_up.weight == "interleave_adjacent(gate_weight,up_weight)"
    assert [(attachment.operation, attachment.site) for attachment in gate_up.epilogue] == [
        ("pairwise_swiglu", AttachmentSite.GEMM_EPILOGUE)
    ]
    assert gate_up.epilogue[0].inputs == ("gate", "up")
    assert gate_up.output == "activated"
    assert plan.materialization("gate").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("up").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("activated").disposition is MaterializationDisposition.MATERIALIZE
    assert plan.activation_materializations[0].value == "activated"
    assert plan.rewrites[0].applied
    assert plan.rewrites[0].numerical_equivalence is NumericalEquivalence.ALGEBRAICALLY_EXACT


def test_combined_adjacent_pair_projection_fuses_pairwise_swiglu() -> None:
    plan = compile_reference_swiglu_region(
        _combined_swiglu_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    gate_up = plan.skeletons[0]
    assert isinstance(gate_up, GemmSkeleton)
    assert gate_up.weight == "gate_up_weight"
    assert gate_up.shape == (TOKENS, 2 * INTERMEDIATE, HIDDEN)
    assert gate_up.epilogue[0].inputs == ("gate_up",)
    assert plan.materialization("gate_up").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert not any(isinstance(skeleton, TransformSkeleton) for skeleton in plan.skeletons)


def test_bitwise_policy_retains_materialized_swiglu() -> None:
    plan = compile_reference_swiglu_region(
        _separate_swiglu_region(),
        numerical_policy=NumericalPolicy.BITWISE_EXACT,
    )

    assert any(isinstance(skeleton, TransformSkeleton) for skeleton in plan.skeletons)
    assert plan.materialization("gate").disposition is MaterializationDisposition.MATERIALIZE
    assert plan.materialization("up").disposition is MaterializationDisposition.MATERIALIZE
    assert not plan.rewrites[0].applied
    assert any("bitwise-exact" in reason for reason in plan.rewrites[0].rejection_reasons)


def test_observed_gate_retains_materialized_swiglu() -> None:
    plan = compile_reference_swiglu_region(
        _separate_swiglu_region(observe_gate=True),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert not plan.rewrites[0].applied
    assert plan.materialization("gate").disposition is MaterializationDisposition.MATERIALIZE
    assert any("gate has 2 consumers" in reason for reason in plan.rewrites[0].rejection_reasons)


def test_adjacent_pair_projection_is_equivalent_over_real_numbers() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(TOKENS, HIDDEN))
    gate_weight = rng.normal(size=(HIDDEN, INTERMEDIATE))
    up_weight = rng.normal(size=(HIDDEN, INTERMEDIATE))
    down_weight = rng.normal(size=(INTERMEDIATE, HIDDEN))

    gate = x @ gate_weight
    up = x @ up_weight
    reference = ((gate / (1.0 + np.exp(-gate))) * up) @ down_weight

    adjacent_weight = np.stack((gate_weight, up_weight), axis=-1).reshape(HIDDEN, 2 * INTERMEDIATE)
    pairs = (x @ adjacent_weight).reshape(TOKENS, INTERMEDIATE, 2)
    fused = ((pairs[..., 0] / (1.0 + np.exp(-pairs[..., 0]))) * pairs[..., 1]) @ down_weight

    np.testing.assert_allclose(fused, reference, rtol=1e-12, atol=1e-12)
