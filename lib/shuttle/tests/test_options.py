# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from shuttle import ExecutionMode, Materialization, Numerics, Tuning, compiler_options, options_digest


def _tuning() -> Tuning:
    return Tuning(
        tile_sizes=(64, 128),
        cluster_shape=(2, 1, 1),
        pipeline_stages=3,
        materialization=Materialization.PREFER_FUSION,
        maximum_candidates=16,
    )


def test_compiler_options_have_canonical_closed_wire_format() -> None:
    options = compiler_options(
        execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=_tuning(),
    )

    assert options == {
        "xla_shuttle_enable": True,
        "xla_shuttle_options": (
            '{"execution_mode":"stablehlo_round_trip","numerics":"source_ordered",'
            '"pipeline_abi_version":10,"schema_version":1,'
            '"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion",'
            '"maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}}'
        ),
    }
    assert (
        options_digest(
            execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
            numerics=Numerics.SOURCE_ORDERED,
            tuning=_tuning(),
        )
        == "bc351e7e440ce6c1a7f6998231358cde98eda6a7035822856488d89f0aa43ffb"
    )


def test_numerical_policies_have_distinct_option_payloads_and_digests() -> None:
    tuning = _tuning()
    source_ordered = compiler_options(
        execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    )
    fast = compiler_options(
        execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
        numerics=Numerics.FAST,
        tuning=tuning,
    )

    assert source_ordered != fast
    assert options_digest(
        execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    ) != options_digest(
        execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
        numerics=Numerics.FAST,
        tuning=tuning,
    )


def test_execution_modes_have_distinct_cache_identity() -> None:
    tuning = _tuning()
    roundtrip = compiler_options(
        execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    )
    cpu_bundle = compiler_options(
        execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    )

    gpu_bundle = compiler_options(
        execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    )

    assert (
        len(
            {
                roundtrip["xla_shuttle_options"],
                cpu_bundle["xla_shuttle_options"],
                gpu_bundle["xla_shuttle_options"],
            }
        )
        == 3
    )
    assert options_digest(
        execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    ) != options_digest(
        execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=tuning,
    )

    assert gpu_bundle["xla_shuttle_options"] == (
        '{"execution_mode":"gpu_executable_bundle","numerics":"source_ordered",'
        '"pipeline_abi_version":10,"schema_version":1,'
        '"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion",'
        '"maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}}'
    )


def test_gpu_executable_mode_rejects_fast_policy() -> None:
    with pytest.raises(ValueError, match="GPU executable bundle requires source_ordered"):
        compiler_options(
            execution_mode=ExecutionMode.GPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.FAST,
            tuning=_tuning(),
        )


def test_cpu_executable_mode_has_distinct_source_ordered_and_fast_cache_identities() -> None:
    source_ordered = compiler_options(
        execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.SOURCE_ORDERED,
        tuning=_tuning(),
    )
    fast = compiler_options(
        execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
        numerics=Numerics.FAST,
        tuning=_tuning(),
    )

    assert source_ordered["xla_shuttle_options"] == (
        '{"execution_mode":"cpu_executable_bundle","numerics":"source_ordered",'
        '"pipeline_abi_version":10,"schema_version":1,'
        '"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion",'
        '"maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}}'
    )
    assert fast["xla_shuttle_options"] == (
        '{"execution_mode":"cpu_executable_bundle","numerics":"fast",'
        '"pipeline_abi_version":10,"schema_version":1,'
        '"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion",'
        '"maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}}'
    )
    assert (
        options_digest(
            execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.SOURCE_ORDERED,
            tuning=_tuning(),
        )
        == "58c083e68c83cc4e0693e6416d2fe2cf5cfc46f3543fb6fee4c0d228490f25a6"
    )
    assert (
        options_digest(
            execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
            numerics=Numerics.FAST,
            tuning=_tuning(),
        )
        == "4d7046a19bf860e24d74a425d579cb5d801046370baf8b041d3218a88115c223"
    )


def test_empty_shape_hints_leave_physical_search_unconstrained() -> None:
    tuning = Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )
    options = compiler_options(
        execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
        numerics=Numerics.FAST,
        tuning=tuning,
    )

    assert options["xla_shuttle_options"] == (
        '{"execution_mode":"stablehlo_round_trip","numerics":"fast",'
        '"pipeline_abi_version":10,"schema_version":1,'
        '"tuning":{"cluster_shape":[],"materialization":"automatic",'
        '"maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}}'
    )


def test_tuning_rejects_nonpositive_search_bounds() -> None:
    tuning = _tuning()
    with pytest.raises(ValueError):
        replace(tuning, tile_sizes=(64, 0))
    with pytest.raises(ValueError):
        replace(tuning, cluster_shape=(1, -1, 1))
    with pytest.raises(ValueError):
        replace(tuning, pipeline_stages=0)
    with pytest.raises(ValueError):
        replace(tuning, maximum_candidates=0)


def test_options_reject_mutable_non_native_and_wrong_enum_values() -> None:
    tuning = _tuning()
    with pytest.raises(TypeError):
        replace(tuning, tile_sizes=cast(tuple[int, ...], [64, 128]))
    with pytest.raises(TypeError):
        replace(tuning, pipeline_stages=cast(int, True))
    with pytest.raises(TypeError):
        replace(tuning, materialization=cast(Materialization, "automatic"))
    with pytest.raises(TypeError):
        compiler_options(
            execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
            numerics=cast(Numerics, "fast"),
            tuning=tuning,
        )
    with pytest.raises(TypeError):
        compiler_options(
            execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
            numerics=Numerics.FAST,
            tuning=cast(Tuning, object()),
        )
    with pytest.raises(TypeError):
        compiler_options(
            execution_mode=cast(ExecutionMode, "cpu_executable_bundle"),
            numerics=Numerics.FAST,
            tuning=tuning,
        )


def test_tuning_rejects_values_outside_the_native_wire_bounds() -> None:
    tuning = _tuning()
    with pytest.raises(ValueError):
        replace(tuning, tile_sizes=(1,) * 9)
    with pytest.raises(ValueError):
        replace(tuning, cluster_shape=(1,) * 4)
    with pytest.raises(ValueError):
        replace(tuning, maximum_candidates=2**31)


def test_stock_jaxlib_rejects_shuttle_options_instead_of_ignoring_them() -> None:
    compiled = jax.jit(
        lambda value: value + 1,
        compiler_options=compiler_options(
            execution_mode=ExecutionMode.STABLEHLO_ROUND_TRIP,
            numerics=Numerics.SOURCE_ORDERED,
            tuning=_tuning(),
        ),
    )

    with pytest.raises(jax.errors.JaxRuntimeError, match="No such compile option: 'xla_shuttle_enable'"):
        compiled(jnp.int32(1))
