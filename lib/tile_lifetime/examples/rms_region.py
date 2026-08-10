# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print the first tile-lifetime execution plan."""

import json
from dataclasses import asdict

from shuttle.ir import DType
from tile_lifetime.compiler import compile_reference_region
from tile_lifetime.ir import TensorGraph
from tile_lifetime.plan import NumericalPolicy


def build_example_graph() -> TensorGraph:
    """Build a small residual/RMSNorm/GEMM semantic region."""
    graph = TensorGraph()
    x = graph.input("x", shape=(128, 512), dtype=DType.BF16)
    residual = graph.input("residual", shape=(128, 512), dtype=DType.BF16)
    weight_0 = graph.parameter("weight_0", shape=(512, 512), dtype=DType.BF16)
    gamma = graph.parameter("gamma", shape=(512,), dtype=DType.BF16)
    weight_1 = graph.parameter("weight_1", shape=(512, 1408), dtype=DType.BF16)

    projected = graph.linear(x, weight_0, name="projected", accumulation_dtype=DType.FP32)
    residual_sum = graph.residual_add(projected, residual, name="residual_sum")
    normalized = graph.rms_norm(
        residual_sum,
        gamma,
        name="normalized",
        axis=-1,
        epsilon=1e-6,
        reduction_dtype=DType.FP32,
    )
    graph.linear(normalized, weight_1, name="output", accumulation_dtype=DType.FP32)
    return graph


def main() -> None:
    plan = compile_reference_region(build_example_graph(), numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER)
    print(json.dumps(asdict(plan), indent=2))


if __name__ == "__main__":
    main()
