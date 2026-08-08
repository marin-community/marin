# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-static behavior tests for the MiniMax MSA oracle adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent / "backends"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sm100_minimax_msa_oracle import (  # noqa: E402
    CorrectnessTolerance,
    alternating_launch_orders,
    compare_matched_boundaries,
    validate_q2k_indices,
)


def _valid_q2k() -> np.ndarray:
    return np.array(
        [
            [[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1]],
            [[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1]],
        ],
        dtype=np.int32,
    )


def test_validate_q2k_indices_accepts_canonical_relation() -> None:
    validate_q2k_indices(
        _valid_q2k(),
        key_value_heads=2,
        total_queries=3,
        top_k=4,
        maximum_key_value_blocks=3,
    )


@pytest.mark.parametrize(
    "invalid",
    (
        np.array([[[1, 0, -1, -1]]], dtype=np.int32),
        np.array([[[0, -1, 1, -1]]], dtype=np.int32),
        np.array([[[0, 0, -1, -1]]], dtype=np.int32),
        np.array([[[3, -1, -1, -1]]], dtype=np.int32),
    ),
)
def test_validate_q2k_indices_rejects_ambiguous_or_invalid_relation(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_q2k_indices(
            invalid,
            key_value_heads=1,
            total_queries=1,
            top_k=4,
            maximum_key_value_blocks=3,
        )


def test_alternating_launch_orders_counterbalances_first_position() -> None:
    assert alternating_launch_orders(("generated", "oracle"), 4) == [
        ("generated", "oracle"),
        ("oracle", "generated"),
        ("generated", "oracle"),
        ("oracle", "generated"),
    ]


def test_compare_matched_boundaries_records_raw_samples_and_shared_route() -> None:
    q2k = _valid_q2k()

    def generated_payload(route: np.ndarray) -> np.ndarray:
        return route.astype(np.float32).sum(axis=-1)

    def oracle_payload(route: np.ndarray) -> np.ndarray:
        return route.astype(np.float32).sum(axis=-1)

    def common_route() -> np.ndarray:
        return q2k.copy()

    elapsed = iter(float(index) for index in range(1, 17))

    record = compare_matched_boundaries(
        generated_payload=generated_payload,
        msa_payload=oracle_payload,
        precomputed_q2k=q2k,
        common_route=common_route,
        semantic_reference=generated_payload,
        tolerance=CorrectnessTolerance(maximum_absolute_error=0.0, mean_absolute_error=0.0),
        oracle_manifest={
            "included_per_payload_call": ("q2k-to-k2q", "attention", "combine"),
            "excluded_from_payload_call": ("router",),
        },
        warmups=0,
        repeats=4,
        measure_one=lambda operation: (operation(), next(elapsed))[1],
        synchronize=lambda: None,
    )

    payload = record["boundaries"]["payload"]
    full = record["boundaries"]["natural_full_route"]
    assert payload["generated_shuttle"]["samples_ms"] == [1.0, 4.0, 5.0, 8.0]
    assert payload["matched_msa_oracle"]["samples_ms"] == [2.0, 3.0, 6.0, 7.0]
    assert full["generated_shuttle"]["samples_ms"] == [9.0, 12.0, 13.0, 16.0]
    assert full["matched_msa_oracle"]["samples_ms"] == [10.0, 11.0, 14.0, 15.0]
    assert record["relation"]["natural_route_deterministic"]
    assert payload["correctness"]["generated_deterministic"]
    assert payload["correctness"]["msa_deterministic"]
    assert record["acceptance_boundary"] == "natural_full_route"


def test_compare_matched_boundaries_rejects_numerical_mismatch() -> None:
    q2k = _valid_q2k()

    with pytest.raises(ValueError, match="maximum absolute error"):
        compare_matched_boundaries(
            generated_payload=lambda route: route.astype(np.float32),
            msa_payload=lambda route: route.astype(np.float32) + 1.0,
            precomputed_q2k=q2k,
            common_route=lambda: q2k.copy(),
            semantic_reference=lambda route: route.astype(np.float32),
            tolerance=CorrectnessTolerance(maximum_absolute_error=0.0, mean_absolute_error=0.0),
            oracle_manifest={
                "included_per_payload_call": ("q2k-to-k2q",),
                "excluded_from_payload_call": ("router",),
            },
            warmups=0,
            repeats=2,
            measure_one=lambda operation: 1.0,
            synchronize=lambda: None,
        )
