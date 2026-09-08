# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    evaluate_wsd80_selected_policy_noninferiority_20260815 as noninferiority,
)

CANDIDATE = noninferiority.PolicyCoordinate(0.06, 0.4875)
REFERENCE = noninferiority.PolicyCoordinate(0.10, 0.50)


def _observations(
    candidate_values: list[float],
    reference_values: list[float],
    *,
    seeds: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    if seeds is None:
        seeds = tuple(range(100, 100 + len(candidate_values)))
    return (
        pd.DataFrame(
            [
                {
                    "phase_0_starcoder": coordinate.phase_0_starcoder,
                    "phase_1_starcoder": coordinate.phase_1_starcoder,
                    "data_seed": seed,
                    "wsd80_bpb": value,
                }
                for coordinate, values in ((CANDIDATE, candidate_values), (REFERENCE, reference_values))
                for seed, value in zip(seeds, values, strict=True)
            ]
        ),
        seeds,
    )


def _evaluate(
    observations: pd.DataFrame,
    seeds: tuple[int, ...],
) -> noninferiority.NonInferiorityResult:
    return noninferiority.paired_noninferiority(
        observations,
        CANDIDATE,
        REFERENCE,
        margin_bpb=0.002,
        alpha=0.05,
        expected_seeds=seeds,
    )


def test_noninferiority_passes_only_when_upper_bound_is_below_margin() -> None:
    observations, seeds = _observations(
        [0.9300, 0.9303, 0.9299, 0.9304, 0.9301, 0.9302, 0.9300, 0.9303, 0.9299, 0.9304, 0.9301, 0.9302],
        [0.9300] * 12,
    )

    result = _evaluate(observations, seeds)

    assert result.evaluable
    assert result.passed
    assert result.one_sided_upper_confidence_bound_bpb < 0.002
    assert result.one_sided_noninferiority_p <= result.alpha


def test_five_realistic_pairs_do_not_certify_an_equal_mean() -> None:
    paired_differences = [-0.002758, -0.001379, 0.0, 0.001379, 0.002758]
    observations, seeds = _observations(
        [0.9300 + difference for difference in paired_differences],
        [0.9300] * 5,
    )

    result = _evaluate(observations, seeds)

    assert result.evaluable
    assert not result.passed
    assert result.mean_candidate_minus_reference_bpb == pytest.approx(0.0)
    assert result.sample_sd_bpb == pytest.approx(0.00218039, abs=1e-8)
    assert result.one_sided_upper_confidence_bound_bpb > 0.002
    assert result.one_sided_noninferiority_p > result.alpha


def test_noninferiority_is_not_evaluable_without_the_exact_seed_manifest() -> None:
    observations, seeds = _observations(
        [0.9300, 0.9301, 0.9302, 0.9303, 0.9304],
        [0.9299, 0.9300, 0.9301, 0.9302, 0.9303],
    )
    observations.loc[0, "data_seed"] = 999

    result = _evaluate(observations, seeds)

    assert not result.evaluable
    assert not result.passed
    assert result.status == "not_evaluable_seed_manifest_mismatch"
    assert result.missing_candidate_seeds == (100,)
    assert result.unexpected_candidate_seeds == (999,)
    assert result.candidate_mean_bpb is None
    assert result.reference_mean_bpb is None


def test_noninferiority_rejects_duplicate_policy_seed_rows() -> None:
    observations, seeds = _observations(
        [0.9300, 0.9301, 0.9302, 0.9303, 0.9304],
        [0.9299, 0.9300, 0.9301, 0.9302, 0.9303],
    )
    observations = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate rows"):
        _evaluate(observations, seeds)


def test_noninferiority_rejects_identical_candidate_and_reference() -> None:
    observations, seeds = _observations(
        [0.9300, 0.9301, 0.9302, 0.9303, 0.9304],
        [0.9299, 0.9300, 0.9301, 0.9302, 0.9303],
    )

    with pytest.raises(ValueError, match="must differ"):
        noninferiority.paired_noninferiority(
            observations,
            CANDIDATE,
            CANDIDATE,
            margin_bpb=0.002,
            alpha=0.05,
            expected_seeds=seeds,
        )


def test_noninferiority_does_not_auto_pass_zero_paired_variance() -> None:
    observations, seeds = _observations([0.9301] * 5, [0.9300] * 5)

    result = _evaluate(observations, seeds)

    assert not result.evaluable
    assert not result.passed
    assert result.status == "not_evaluable_zero_paired_variance"


def test_noninferiority_rejects_out_of_simplex_coordinates() -> None:
    observations, seeds = _observations([0.9301] * 5, [0.9300] * 5)

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        noninferiority.paired_noninferiority(
            observations,
            noninferiority.PolicyCoordinate(-0.01, 0.49),
            REFERENCE,
            margin_bpb=0.002,
            alpha=0.05,
            expected_seeds=seeds,
        )


def test_cli_rejects_policy_coordinate_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observations = tmp_path / "observations.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wsd80_selected_policy_noninferiority_20260815.py",
            "--observations",
            str(observations),
            "--candidate-phase-0",
            "0.5",
        ],
    )

    with pytest.raises(SystemExit):
        noninferiority._parse_args()
