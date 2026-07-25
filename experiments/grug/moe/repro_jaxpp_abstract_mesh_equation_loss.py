# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce JaxPP dropping a shard_map equation that uses AbstractMesh.

Run this file with NVIDIA/jaxpp at ``JAXPP_REVISION`` on ``PYTHONPATH``. The
reproducer uses one CPU, one scalar input, and one shard_map equation.
"""

import dataclasses
import json

import jax
import jaxpp.core as jaxpp_core
import numpy as np
from jax.extend import core as jax_core
from jax.sharding import AbstractMesh, Mesh, PartitionSpec

JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"


@dataclasses.dataclass(frozen=True)
class ReproductionResult:
    original_equations: int
    rewritten_equations: int
    validation_error: str | None


def abstract_mesh_shard_map_jaxpr() -> jax_core.ClosedJaxpr:
    abstract_mesh = AbstractMesh((1,), ("x",))

    @jax.shard_map(
        mesh=abstract_mesh,
        in_specs=PartitionSpec(),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    def add_one(value):
        return value + 1

    value = jax.ShapeDtypeStruct((), jax.numpy.float32)
    return jax.make_jaxpr(add_one)(value)


def reproduce_equation_loss() -> ReproductionResult:
    original = abstract_mesh_shard_map_jaxpr()
    jax_core.check_jaxpr(original.jaxpr)

    cpu_mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("x",))
    rewritten = jaxpp_core.replace_captured_meshes(original, cpu_mesh)

    try:
        jax_core.check_jaxpr(rewritten.jaxpr)
    except jax_core.JaxprTypeError as error:
        validation_error = str(error)
    else:
        validation_error = None

    return ReproductionResult(
        original_equations=len(original.eqns),
        rewritten_equations=len(rewritten.eqns),
        validation_error=validation_error,
    )


def main() -> int:
    result = reproduce_equation_loss()
    print(json.dumps(dataclasses.asdict(result), indent=2))

    reproduced = (
        result.original_equations == 1
        and result.rewritten_equations == 0
        and result.validation_error is not None
        and "not defined" in result.validation_error
    )
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
