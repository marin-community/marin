# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contracts for the unbuilt ABI 7 VJP Host proof."""

import json

from shuttle_jaxlib_cpu_ffi_vjp_acceptance import (
    BOUNDARIES,
    PIPELINE_ABI_VERSION,
    SHAPE,
    arrays,
    boundary_function,
    fixed_inputs,
    ready,
    subject_options,
)


def test_vjp_host_driver_binds_abi7_cpu_mode() -> None:
    assert PIPELINE_ABI_VERSION == 7
    payload = json.loads(subject_options()["xla_shuttle_options"])
    assert payload == {
        "execution_mode": "cpu_executable_bundle",
        "numerics": "source_ordered",
        "pipeline_abi_version": 7,
        "schema_version": 1,
        "tuning": {
            "cluster_shape": [],
            "materialization": "automatic",
            "maximum_candidates": 1,
            "pipeline_stages": 1,
            "tile_sizes": [],
        },
    }


def test_vjp_host_driver_preserves_public_jax_result_order() -> None:
    assert BOUNDARIES == ("backward", "composed")
    backward = arrays(ready(boundary_function("backward")(*fixed_inputs(SHAPE, "backward"))))
    composed = arrays(ready(boundary_function("composed")(*fixed_inputs(SHAPE, "composed"))))
    assert [value.shape for value in backward] == [(7, 13), (13,)]
    assert [value.shape for value in composed] == [(7, 13), (7, 13), (13,)]
    assert composed[1].tobytes() == backward[0].tobytes()
    assert composed[2].tobytes() == backward[1].tobytes()
