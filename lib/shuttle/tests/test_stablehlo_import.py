# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from shuttle.ir import DType
from shuttle.stablehlo_import import DotAttributes, import_stablehlo


def test_import_stablehlo_detaches_typed_dataflow_from_mlir_context() -> None:
    def linear_map(left: jax.Array, right: jax.Array) -> jax.Array:
        return (left @ right) * jnp.asarray(0.5, dtype=jnp.bfloat16)

    exported = jax.export.export(jax.jit(linear_map))(
        jax.ShapeDtypeStruct((2, 4), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, 3), jnp.bfloat16),
    )
    graph = import_stablehlo(exported.mlir_module_serialized, input_names=("left", "right"))

    assert [
        (graph.value(value_id).name, graph.value(value_id).shape, graph.value(value_id).dtype)
        for value_id in graph.inputs
    ] == [
        ("left", (2, 4), DType.BF16),
        ("right", (4, 3), DType.BF16),
    ]
    dot = next(operation for operation in graph.operations if operation.kind == "dot_general")
    assert isinstance(dot.attributes, DotAttributes)
    assert dot.attributes.lhs_contracting_dimensions == (1,)
    assert dot.attributes.rhs_contracting_dimensions == (0,)
    assert graph.producer(graph.outputs[0]) is not None
