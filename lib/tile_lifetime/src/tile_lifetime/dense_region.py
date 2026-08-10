# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a connected dense region after frontend names erase to Flow."""

from tile_lifetime.compiler import RowScalePlacement
from tile_lifetime.dense_flow import erase_dense_transformer_semantics
from tile_lifetime.dense_flow_planner import compile_erased_dense_transformer_region
from tile_lifetime.ir import TensorGraph
from tile_lifetime.plan import NumericalPolicy, RegionPlan


def compile_dense_transformer_region(
    graph: TensorGraph,
    *,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RowScalePlacement = RowScalePlacement.CONSUMER_PROLOGUE,
) -> RegionPlan:
    """Compile one connected dense region exclusively from erased Flow algebra."""
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("the dense structural prototype requires the rounding-reorder numerical policy")
    erased = erase_dense_transformer_semantics(graph)
    return compile_erased_dense_transformer_region(
        erased,
        row_scale_placement=rms_scale_placement,
    )
