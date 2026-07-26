# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce JaxPP rejecting equivalent replicated output shardings.

Run this file with NVIDIA/jaxpp at ``JAXPP_REVISION`` on ``PYTHONPATH``. The
reproducer uses one CPU and covers the rank-1 and rank-2 assertion payloads
observed after automatic eager-1F1B execution.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxpp.array import MpmdArray
from jaxpp.mesh import MpmdMesh
from jaxpp.types import MpmdSharding

JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"


@dataclasses.dataclass(frozen=True)
class OutputSpecCase:
    shape: tuple[int, ...]
    actual_spec: str
    target_spec: str
    raw_equal: bool
    semantically_equivalent: bool
    constructor_error: str | None


def _one_cpu_mpmd_mesh() -> MpmdMesh:
    device = jax.devices("cpu")[0]
    return MpmdMesh(Mesh(np.asarray([device], dtype=object), ("stage",)), "stage")


def reproduce_output_spec_case(shape: tuple[int, ...]) -> OutputSpecCase:
    mpmd_mesh = _one_cpu_mpmd_mesh()
    actual_spec = P(*([None] * len(shape)))
    target_spec = P()
    local_mesh = mpmd_mesh.unstack[0]
    array = jax.device_put(
        jnp.zeros(shape, dtype=jnp.float32),
        NamedSharding(local_mesh, actual_spec),
    )

    try:
        MpmdArray([array], MpmdSharding(mpmd_mesh, {0}, target_spec))
    except AssertionError as error:
        constructor_error = str(error)
    else:
        constructor_error = None

    actual_sharding = NamedSharding(local_mesh, actual_spec)
    target_sharding = NamedSharding(local_mesh, target_spec)
    return OutputSpecCase(
        shape=shape,
        actual_spec=str(actual_spec),
        target_spec=str(target_spec),
        raw_equal=actual_spec == target_spec,
        semantically_equivalent=actual_sharding.is_equivalent_to(target_sharding, len(shape)),
        constructor_error=constructor_error,
    )


def reproduce_output_spec_mismatch() -> tuple[OutputSpecCase, ...]:
    return (
        reproduce_output_spec_case((2,)),
        reproduce_output_spec_case((2, 3)),
    )


def main() -> int:
    results = reproduce_output_spec_mismatch()
    print(json.dumps([dataclasses.asdict(result) for result in results], indent=2))

    reproduced = all(
        result.constructor_error is not None and not result.raw_equal and result.semantically_equivalent
        for result in results
    )
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
