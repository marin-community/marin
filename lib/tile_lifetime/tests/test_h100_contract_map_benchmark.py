# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tile_lifetime.h100_contract_map_benchmark import (
    ArchitectureStatus,
    BackendVariant,
    ExternalComparator,
    MeasurementBoundary,
    RepeatabilityMode,
    StructuralFeature,
    comparator_decision,
    default_h100_contract_map_benchmark_plan,
    require_gpu_execution_ready,
    staging_manifest,
)

_CANONICAL_REVISION = "ca2091a4b27a366c4f3625cd339b21e139886450"


def test_default_plan_requires_three_backends_two_boundaries_and_anonymous_irregular_cases() -> None:
    plan = default_h100_contract_map_benchmark_plan()

    assert plan.architecture_status is ArchitectureStatus.NONCONFORMING
    assert plan.backends == tuple(BackendVariant)
    assert plan.boundaries == tuple(MeasurementBoundary)
    assert len(plan.cases) == 4
    assert len({case.case_id for case in plan.cases}) == 4
    assert all(case.case_id.startswith("contract_map_") for case in plan.cases)
    assert all(case.rows % 2 == 1 and case.reduction % 8 == 0 and case.features % 8 == 0 for case in plan.cases)


def test_plan_rejects_missing_resource_or_logical_boundary_evidence() -> None:
    plan = default_h100_contract_map_benchmark_plan()

    with pytest.raises(ValueError, match="resource evidence cannot omit"):
        replace(plan.resources, ptx=False)
    with pytest.raises(ValueError, match="logical-boundary evidence cannot omit"):
        replace(plan.logical_boundary, saved_state_names_and_bytes=False)


def test_dense_contract_map_plan_excludes_attention_and_routed_comparators() -> None:
    plan = default_h100_contract_map_benchmark_plan()

    fa4 = comparator_decision(ExternalComparator.FA4, plan.features)
    grug = comparator_decision(ExternalComparator.GRUG, plan.features)

    assert not fa4.admitted
    assert fa4.missing_features == (
        StructuralFeature.ATTENTION_SCORE,
        StructuralFeature.NORMALIZED_EXP,
        StructuralFeature.FOLD,
    )
    assert not grug.admitted
    assert grug.missing_features == (
        StructuralFeature.SEGMENTED_CONTRACT,
        StructuralFeature.RELATION,
        StructuralFeature.TRANSPORT,
    )


def test_staging_manifest_is_structural_and_records_every_counterbalanced_order() -> None:
    manifest = staging_manifest(shuttle_revision=_CANONICAL_REVISION)
    serialized = json.dumps(manifest, sort_keys=True)

    assert manifest["kind"] == "staged_plan_no_gpu_evidence"
    assert not manifest["execution_allowed"]
    assert len(manifest["counterbalanced_orders"]) == 6
    assert len({tuple(order) for order in manifest["counterbalanced_orders"]}) == 6
    assert "workload" not in serialized
    assert "model_name" not in serialized
    assert all(not decision["admitted"] for decision in manifest["external_comparators"])


def test_gpu_execution_refuses_before_backend_or_collector_wiring() -> None:
    with pytest.raises(RuntimeError, match=r"architecture-nonconforming.*ordinary_xla.*shuttle_fast"):
        require_gpu_execution_ready()


def test_plan_rejects_missing_backend_or_measurement_boundary() -> None:
    plan = default_h100_contract_map_benchmark_plan()

    with pytest.raises(ValueError, match="every plan must compare"):
        replace(plan, backends=plan.backends[:-1])
    with pytest.raises(ValueError, match="every plan must measure"):
        replace(plan, boundaries=plan.boundaries[:-1])


def test_timing_protocol_requires_full_permutation_balance_and_raw_cache_evidence() -> None:
    plan = default_h100_contract_map_benchmark_plan()
    timing = plan.timing

    with pytest.raises(ValueError, match="backend permutations"):
        replace(timing, steady_state_repeats=25)
    with pytest.raises(ValueError, match="cache isolation"):
        replace(timing, isolate_persistent_cache_roots=False)
    with pytest.raises(ValueError, match="raw samples"):
        replace(timing, retain_raw_samples=False)


def test_numerical_floor_rejects_posthoc_or_inconsistent_accuracy_bounds() -> None:
    floor = default_h100_contract_map_benchmark_plan().numerical_floors[1]
    assert floor.backend is BackendVariant.SHUTTLE_SOURCE_ORDERED
    assert floor.repeatability is RepeatabilityMode.BITWISE

    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(floor, maximum_absolute_error=-1.0)
    with pytest.raises(ValueError, match="mean absolute error"):
        replace(floor, mean_absolute_error=floor.maximum_absolute_error + 1.0)
    with pytest.raises(ValueError, match="mean ULP distance"):
        replace(floor, mean_ulp_distance=floor.maximum_ulp_distance + 1.0)
    with pytest.raises(ValueError, match="bitwise repeatability"):
        replace(floor, repeat_maximum_absolute_error=0.01)
