# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14"]
# ///
"""Evaluate the frozen WSD80 selected policy with paired non-inferiority.

Lower BPB is better. For same-seed differences

    d_s = BPB_s(candidate) - BPB_s(reference),

the candidate passes only when the one-sided confidence upper bound for the
mean difference is at most the predeclared practical margin. This is stronger
than failing to reject equality: an underpowered comparison cannot pass.

The command-line contract is fixed to the preregistered protocol. It does not
accept overrides for the margin, alpha, repeat count, or seed manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PHASE_0_COLUMN = "phase_0_starcoder"
PHASE_1_COLUMN = "phase_1_starcoder"
SEED_COLUMN = "data_seed"
METRIC_COLUMN = "wsd80_bpb"
PROTOCOL_ID = "wsd80-selected-policy-noninferiority-20260815-v2"
PREREGISTERED_ALPHA = 0.05
PREREGISTERED_MARGIN_BPB = 0.002
PREREGISTERED_SEEDS = tuple(range(20_260_816, 20_260_828))
PREREGISTERED_CANDIDATE = (0.0575, 0.4900)
PREREGISTERED_REFERENCE = (0.1000, 0.5000)
COORDINATE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PolicyCoordinate:
    """A two-phase StarCoder mixture coordinate."""

    phase_0_starcoder: float
    phase_1_starcoder: float


@dataclass(frozen=True)
class NearestCoordinate:
    """Nearest observed coordinate when an exact policy is absent."""

    coordinate: PolicyCoordinate
    euclidean_distance: float
    row_count: int


@dataclass(frozen=True)
class NonInferiorityResult:
    """Machine-readable result of one paired policy comparison."""

    protocol_id: str
    observations_sha256: str | None
    status: str
    evaluable: bool
    passed: bool
    candidate: PolicyCoordinate
    reference: PolicyCoordinate
    margin_bpb: float
    alpha: float
    expected_seeds: tuple[int, ...]
    candidate_rows: int
    reference_rows: int
    pair_count: int
    paired_seeds: tuple[int, ...]
    missing_candidate_seeds: tuple[int, ...]
    missing_reference_seeds: tuple[int, ...]
    unexpected_candidate_seeds: tuple[int, ...]
    unexpected_reference_seeds: tuple[int, ...]
    nearest_candidate: NearestCoordinate | None
    nearest_reference: NearestCoordinate | None
    candidate_mean_bpb: float | None
    reference_mean_bpb: float | None
    mean_candidate_minus_reference_bpb: float | None
    sample_sd_bpb: float | None
    standard_error_bpb: float | None
    one_sided_upper_confidence_bound_bpb: float | None
    two_sided_ci_low_bpb: float | None
    two_sided_ci_high_bpb: float | None
    one_sided_noninferiority_p: float | None
    candidate_win_count: int | None


def _validate_contract(
    candidate: PolicyCoordinate,
    reference: PolicyCoordinate,
    margin_bpb: float,
    alpha: float,
    expected_seeds: tuple[int, ...],
    coordinate_tolerance: float,
) -> None:
    coordinate_values = (
        candidate.phase_0_starcoder,
        candidate.phase_1_starcoder,
        reference.phase_0_starcoder,
        reference.phase_1_starcoder,
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in coordinate_values):
        raise ValueError("policy coordinates must be finite values in [0, 1]")
    if not math.isfinite(margin_bpb) or margin_bpb < 0.0:
        raise ValueError("margin_bpb must be finite and nonnegative")
    if not math.isfinite(alpha) or not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if len(expected_seeds) < 2 or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected_seeds must contain at least two unique seeds")
    if not math.isfinite(coordinate_tolerance) or coordinate_tolerance < 0.0:
        raise ValueError("coordinate_tolerance must be finite and nonnegative")
    if (
        abs(candidate.phase_0_starcoder - reference.phase_0_starcoder) <= coordinate_tolerance
        and abs(candidate.phase_1_starcoder - reference.phase_1_starcoder) <= coordinate_tolerance
    ):
        raise ValueError("candidate and reference coordinates must differ")


def _rows_at_coordinate(
    observations: pd.DataFrame,
    coordinate: PolicyCoordinate,
    coordinate_tolerance: float,
) -> pd.DataFrame:
    required = {PHASE_0_COLUMN, PHASE_1_COLUMN, SEED_COLUMN, METRIC_COLUMN}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    phase_0 = pd.to_numeric(observations[PHASE_0_COLUMN], errors="raise").to_numpy(dtype=float)
    phase_1 = pd.to_numeric(observations[PHASE_1_COLUMN], errors="raise").to_numpy(dtype=float)
    selected = observations.loc[
        np.isclose(phase_0, coordinate.phase_0_starcoder, rtol=0.0, atol=coordinate_tolerance)
        & np.isclose(phase_1, coordinate.phase_1_starcoder, rtol=0.0, atol=coordinate_tolerance)
    ].copy()
    if not selected.empty:
        selected[SEED_COLUMN] = pd.to_numeric(selected[SEED_COLUMN], errors="raise").astype(int)
        selected[METRIC_COLUMN] = pd.to_numeric(selected[METRIC_COLUMN], errors="raise").astype(float)
    if selected[SEED_COLUMN].duplicated().any():
        duplicates = sorted(selected.loc[selected[SEED_COLUMN].duplicated(keep=False), SEED_COLUMN].unique().tolist())
        raise ValueError(f"Coordinate {coordinate} has duplicate rows for seeds {duplicates}")
    if not selected.empty and not np.isfinite(selected[METRIC_COLUMN].to_numpy()).all():
        raise ValueError(f"Coordinate {coordinate} has non-finite BPB values")
    return selected


def _nearest_coordinate(observations: pd.DataFrame, coordinate: PolicyCoordinate) -> NearestCoordinate | None:
    if observations.empty:
        return None
    coordinates = observations[[PHASE_0_COLUMN, PHASE_1_COLUMN]].apply(pd.to_numeric, errors="raise")
    unique = coordinates.drop_duplicates().to_numpy(dtype=float)
    target = np.array([coordinate.phase_0_starcoder, coordinate.phase_1_starcoder])
    distances = np.linalg.norm(unique - target[None, :], axis=1)
    nearest = unique[int(np.argmin(distances))]
    row_count = int(
        (
            np.isclose(coordinates[PHASE_0_COLUMN], nearest[0], rtol=0.0, atol=COORDINATE_TOLERANCE)
            & np.isclose(coordinates[PHASE_1_COLUMN], nearest[1], rtol=0.0, atol=COORDINATE_TOLERANCE)
        ).sum()
    )
    return NearestCoordinate(
        coordinate=PolicyCoordinate(float(nearest[0]), float(nearest[1])),
        euclidean_distance=float(np.min(distances)),
        row_count=row_count,
    )


def _not_evaluable(
    *,
    status: str,
    observations_sha256: str | None,
    candidate: PolicyCoordinate,
    reference: PolicyCoordinate,
    margin_bpb: float,
    alpha: float,
    expected_seeds: tuple[int, ...],
    candidate_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    observations: pd.DataFrame,
) -> NonInferiorityResult:
    expected = set(expected_seeds)
    candidate_seeds = set(candidate_rows[SEED_COLUMN].tolist())
    reference_seeds = set(reference_rows[SEED_COLUMN].tolist())
    return NonInferiorityResult(
        protocol_id=PROTOCOL_ID,
        observations_sha256=observations_sha256,
        status=status,
        evaluable=False,
        passed=False,
        candidate=candidate,
        reference=reference,
        margin_bpb=margin_bpb,
        alpha=alpha,
        expected_seeds=expected_seeds,
        candidate_rows=len(candidate_rows),
        reference_rows=len(reference_rows),
        pair_count=len(candidate_seeds & reference_seeds & expected),
        paired_seeds=tuple(sorted(candidate_seeds & reference_seeds & expected)),
        missing_candidate_seeds=tuple(sorted(expected - candidate_seeds)),
        missing_reference_seeds=tuple(sorted(expected - reference_seeds)),
        unexpected_candidate_seeds=tuple(sorted(candidate_seeds - expected)),
        unexpected_reference_seeds=tuple(sorted(reference_seeds - expected)),
        nearest_candidate=_nearest_coordinate(observations, candidate),
        nearest_reference=_nearest_coordinate(observations, reference),
        candidate_mean_bpb=None,
        reference_mean_bpb=None,
        mean_candidate_minus_reference_bpb=None,
        sample_sd_bpb=None,
        standard_error_bpb=None,
        one_sided_upper_confidence_bound_bpb=None,
        two_sided_ci_low_bpb=None,
        two_sided_ci_high_bpb=None,
        one_sided_noninferiority_p=None,
        candidate_win_count=None,
    )


def paired_noninferiority(
    observations: pd.DataFrame,
    candidate: PolicyCoordinate,
    reference: PolicyCoordinate,
    *,
    margin_bpb: float,
    alpha: float,
    expected_seeds: tuple[int, ...],
    coordinate_tolerance: float = COORDINATE_TOLERANCE,
    observations_sha256: str | None = None,
) -> NonInferiorityResult:
    """Test whether candidate regret relative to reference is below ``margin_bpb``."""
    _validate_contract(candidate, reference, margin_bpb, alpha, expected_seeds, coordinate_tolerance)
    candidate_rows = _rows_at_coordinate(observations, candidate, coordinate_tolerance)
    reference_rows = _rows_at_coordinate(observations, reference, coordinate_tolerance)
    if candidate_rows.empty:
        status = "not_evaluable_missing_candidate_coordinate"
    elif reference_rows.empty:
        status = "not_evaluable_missing_reference_coordinate"
    else:
        expected = set(expected_seeds)
        candidate_seeds = set(candidate_rows[SEED_COLUMN].tolist())
        reference_seeds = set(reference_rows[SEED_COLUMN].tolist())
        status = (
            "evaluable" if candidate_seeds == reference_seeds == expected else "not_evaluable_seed_manifest_mismatch"
        )

    if status != "evaluable":
        return _not_evaluable(
            status=status,
            observations_sha256=observations_sha256,
            candidate=candidate,
            reference=reference,
            margin_bpb=margin_bpb,
            alpha=alpha,
            expected_seeds=expected_seeds,
            candidate_rows=candidate_rows,
            reference_rows=reference_rows,
            observations=observations,
        )

    candidate_by_seed = candidate_rows.set_index(SEED_COLUMN)[METRIC_COLUMN].sort_index()
    reference_by_seed = reference_rows.set_index(SEED_COLUMN)[METRIC_COLUMN].sort_index()
    values = (candidate_by_seed - reference_by_seed).to_numpy(dtype=float)
    pair_count = len(values)
    mean = float(values.mean())
    sample_sd = float(values.std(ddof=1))
    if sample_sd == 0.0:
        return _not_evaluable(
            status="not_evaluable_zero_paired_variance",
            observations_sha256=observations_sha256,
            candidate=candidate,
            reference=reference,
            margin_bpb=margin_bpb,
            alpha=alpha,
            expected_seeds=expected_seeds,
            candidate_rows=candidate_rows,
            reference_rows=reference_rows,
            observations=observations,
        )

    standard_error = sample_sd / math.sqrt(pair_count)
    degrees_of_freedom = pair_count - 1
    upper_bound = mean + float(stats.t.ppf(1.0 - alpha, degrees_of_freedom)) * standard_error
    half_width = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)) * standard_error
    statistic = (mean - margin_bpb) / standard_error
    p_value = float(stats.t.cdf(statistic, degrees_of_freedom))
    passed = upper_bound <= margin_bpb
    if passed != (p_value <= alpha):
        raise AssertionError("Non-inferiority UCB and p-value decisions disagree")

    return NonInferiorityResult(
        protocol_id=PROTOCOL_ID,
        observations_sha256=observations_sha256,
        status="passed" if passed else "failed",
        evaluable=True,
        passed=passed,
        candidate=candidate,
        reference=reference,
        margin_bpb=margin_bpb,
        alpha=alpha,
        expected_seeds=expected_seeds,
        candidate_rows=len(candidate_rows),
        reference_rows=len(reference_rows),
        pair_count=pair_count,
        paired_seeds=expected_seeds,
        missing_candidate_seeds=(),
        missing_reference_seeds=(),
        unexpected_candidate_seeds=(),
        unexpected_reference_seeds=(),
        nearest_candidate=None,
        nearest_reference=None,
        candidate_mean_bpb=float(candidate_by_seed.mean()),
        reference_mean_bpb=float(reference_by_seed.mean()),
        mean_candidate_minus_reference_bpb=mean,
        sample_sd_bpb=sample_sd,
        standard_error_bpb=standard_error,
        one_sided_upper_confidence_bound_bpb=upper_bound,
        two_sided_ci_low_bpb=mean - half_width,
        two_sided_ci_high_bpb=mean + half_width,
        one_sided_noninferiority_p=p_value,
        candidate_win_count=int(np.sum(values < 0.0)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = args.observations.read_bytes()
    observations = pd.read_csv(args.observations)
    result = paired_noninferiority(
        observations,
        PolicyCoordinate(*PREREGISTERED_CANDIDATE),
        PolicyCoordinate(*PREREGISTERED_REFERENCE),
        margin_bpb=PREREGISTERED_MARGIN_BPB,
        alpha=PREREGISTERED_ALPHA,
        expected_seeds=PREREGISTERED_SEEDS,
        observations_sha256=hashlib.sha256(payload).hexdigest(),
    )
    rendered = json.dumps(asdict(result), indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
