# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.grug.moe.benchmark_nccl_ubx import (
    BenchmarkConfig,
    admission_result,
    build_route_plan,
    reference_maps,
    ring_assignment_indices,
)


def _small_config(routing: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        tokens_per_rank=8,
        hidden_dim=32,
        num_experts=16,
        top_k=4,
        routing=routing,
        warmup=0,
        iterations=2,
    )


def test_balanced_route_plan_preserves_all_assignments() -> None:
    config = _small_config("balanced")
    plan = build_route_plan(config)

    assert int(plan.accepted_counts.sum()) == config.global_assignments
    assert np.array_equal(plan.original_counts, plan.accepted_counts)
    assert np.array_equal(plan.drops_by_expert_rank, np.zeros(8, dtype=np.int64))
    assert np.all(plan.routing.sum(axis=1) == config.top_k)


def test_learned_skew_route_plan_matches_ring_prefix_cap() -> None:
    config = _small_config("learned_skew")
    plan = build_route_plan(config)

    assert int(plan.accepted_counts.sum() + plan.drops_by_expert_rank.sum()) == config.global_assignments
    accepted_by_rank = plan.accepted_counts.reshape(8, config.experts_per_rank).sum(axis=1)
    assert np.all(accepted_by_rank <= config.capacity_per_expert_rank)
    assert np.all(plan.routing.sum(axis=1) <= config.top_k)
    assert np.any(plan.drops_by_expert_rank > 0)


def test_reference_maps_and_ring_indices_cover_the_same_accepted_routes() -> None:
    config = _small_config("learned_skew")
    plan = build_route_plan(config)

    for rank in range(8):
        maps = reference_maps(plan, config, rank)
        assignment_indices, valid = ring_assignment_indices(plan, config, rank)
        expert_slice = slice(rank * config.experts_per_rank, (rank + 1) * config.experts_per_rank)
        accepted_on_rank = int(plan.accepted_counts[expert_slice].sum())

        assert int(maps.inverse_map[:, 3].sum()) == accepted_on_rank
        assert maps.valid_slots.size == accepted_on_rank
        assert int(valid.sum()) == accepted_on_rank
        selected = plan.selected_experts.reshape(-1)[assignment_indices[valid]]
        assert np.all((selected >= expert_slice.start) & (selected < expert_slice.stop))


def test_admission_requires_exactness_relative_l2_and_speedup() -> None:
    passing = admission_result(
        route_exact=True,
        candidate_relative_l2={"ubx_vs_fp32_identity_reference": 0.0019},
        ring_p50_ms=11.0,
        ubx_p50_ms=10.0,
        relative_l2_limit=0.002,
        required_speedup=1.10,
    )
    bad_route = admission_result(
        route_exact=False,
        candidate_relative_l2={"ubx_vs_fp32_identity_reference": 0.0},
        ring_p50_ms=12.0,
        ubx_p50_ms=10.0,
        relative_l2_limit=0.002,
        required_speedup=1.10,
    )
    bad_output = admission_result(
        route_exact=True,
        candidate_relative_l2={"ubx_vs_fp32_identity_reference": 0.0021},
        ring_p50_ms=12.0,
        ubx_p50_ms=10.0,
        relative_l2_limit=0.002,
        required_speedup=1.10,
    )
    bad_speed = admission_result(
        route_exact=True,
        candidate_relative_l2={"ubx_vs_fp32_identity_reference": 0.0},
        ring_p50_ms=10.99,
        ubx_p50_ms=10.0,
        relative_l2_limit=0.002,
        required_speedup=1.10,
    )

    assert passing["passed"]
    assert passing["candidate_reference"] == "fp32_identity"
    assert not bad_route["passed"]
    assert not bad_output["passed"]
    assert not bad_speed["passed"]
