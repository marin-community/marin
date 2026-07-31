# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the structural claims in the HPR audit against the design matrix the model actually builds.

The audit makes four claims that are decidable by arithmetic rather than by fitting: the pooled
family base column is a linear combination of its members' excess columns, singleton families emit
two identical harm columns, the benefit link eventually outgrows the replay penalty for every
admissible exponent, and the tied-policy restriction leaves the transition parameters active. Each is
checked here directly on the built design so the fix is aimed at a confirmed defect.

Rank and condition number are reported per dataset because the severity is dataset-dependent: a
partition with many singleton families duplicates more columns than one with three broad families.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "hpr_identifiability_audit_20260727"
VARIANT = bench.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.THREE_HUNDRED_M_TABLE9,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)
DUPLICATE_TOLERANCE = 1e-12
COLLINEARITY_TOLERANCE = 1e-9


def duplicate_columns(values: np.ndarray, names: tuple[str, ...]) -> list[tuple[str, str, float]]:
    """Column pairs that are numerically identical, which no fit can separate."""
    scale = np.maximum(np.abs(values).max(axis=0), 1e-30)
    normalized = values / scale[None, :]
    found = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            gap = float(np.abs(normalized[:, left] - normalized[:, right]).max())
            if gap < DUPLICATE_TOLERANCE:
                found.append((names[left], names[right], gap))
    return found


def pooled_base_collinearity(
    values: np.ndarray,
    names: tuple[str, ...],
    dataset,
) -> list[tuple[str, int, float]]:
    """Each pooled family base column against the sum of that family's own bucket-excess columns.

    The base is built as the summed bucket signal over a family's members, and every one of those
    members also receives its own excess column, so the exact relation to test is
    ``base_f == sum_{i in f} excess_i`` rather than mere membership of the excess span.
    """
    excess = {name.split(":", 1)[1]: index for index, name in enumerate(names) if name.startswith("bucket_excess")}
    by_family = {
        family: tuple(dataset.domains[member] for member in members)
        for family, members in zip(dataset.family_names, dataset.family_members, strict=True)
    }
    findings = []
    for index, name in enumerate(names):
        if not name.startswith("pooled_base_signal:"):
            continue
        family = name.split(":", 1)[1]
        member_columns = [excess[domain] for domain in by_family.get(family, ()) if domain in excess]
        if not member_columns:
            continue
        summed = values[:, member_columns].sum(axis=1)
        gap = float(np.abs(summed - values[:, index]).max())
        findings.append((name, len(member_columns), gap / max(float(np.abs(values[:, index]).max()), 1e-30)))
    return findings


def link_crossover(exponent: float, threshold: float) -> float | None:
    """Exposure at which the power benefit overtakes the log-squared replay harm, if it does."""
    grid = np.logspace(0.0, 12.0, 4000)
    benefit = bench.power_response(grid, exponent)
    harm = bench.overexposure_harm(grid[None, :], threshold)[0]
    dominated = benefit > harm
    if not dominated.any():
        return None
    return float(grid[int(np.argmax(dominated))])


def tied_restriction_active(dataset) -> dict[str, float]:
    """Whether the exposure of a tied policy still depends on the transition parameters.

    A clean single-phase restriction of a two-phase model would make the phase boundary irrelevant
    when both phases carry the same mixture. Retained exposure is evaluated on a tied policy under
    two transition shapes to see whether it does.
    """
    tied = bench.proportional_weights(dataset)
    stacked = np.stack([tied, tied], axis=0)[None, :, :]
    probe = replace(dataset, weights=stacked, target=np.zeros(1, dtype=float))
    neutral = bench.family_grp.Shape(0.5, 1.0, 0.0, 5.0)
    fitted = bench.family_grp.Shape(0.5, 6.627794351309641, 2.28, 5.0)
    return {
        "tied_exposure_neutral_sum": float(bench.retained_exposure(probe, neutral).sum()),
        "tied_exposure_fitted_sum": float(bench.retained_exposure(probe, fitted).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shape = bench.family_grp.Shape(
        exponent=0.33989885260566105,
        late_multiplier=6.627794351309641,
        forgetting_rate=6.14421235332821e-06,
        penalty_threshold=5.136810831800622,
    )
    rows = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        config = bench.Config(VARIANT, 0, shape, 0.0, 1.0, 0.0, 0.0)
        design = bench.build_design(dataset, config)
        values, names = design.values, design.names
        rank = int(np.linalg.matrix_rank(values))
        centered = values - values.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        positive = singular[singular > singular.max() * 1e-14]
        duplicates = duplicate_columns(values, names)
        collinear = pooled_base_collinearity(values, names, dataset)
        singletons = sum(1 for members in dataset.family_members if len(members) == 1)
        rows.append(
            {
                "dataset": dataset_id.value,
                "rows": values.shape[0],
                "columns": values.shape[1],
                "rank": rank,
                "deficiency": values.shape[1] - rank,
                "condition_number": float(positive.max() / positive.min()),
                "families": len(dataset.family_names),
                "singleton_families": singletons,
                "duplicate_pairs": len(duplicates),
                "pooled_base_worst_residual": max((gap for _n, _k, gap in collinear), default=float("nan")),
            }
        )
        print(f"\n{dataset_id.value}")
        print(f"  design {values.shape[0]}x{values.shape[1]}  rank {rank}  deficiency {values.shape[1] - rank}")
        print(f"  condition number (centered) {positive.max() / positive.min():.4g}")
        print(f"  families {len(dataset.family_names)}  singletons {singletons}")
        for left, right, gap in duplicates:
            print(f"  DUPLICATE  {left}  ==  {right}   (max abs gap {gap:.2e})")
        for name, member_count, gap in collinear:
            verdict = "EXACT SUM of its excess columns" if gap < COLLINEARITY_TOLERANCE else "independent"
            print(f"  {name}: {member_count} members, relative gap {gap:.2e} -> {verdict}")
        for key, value in tied_restriction_active(dataset).items():
            print(f"  {key}: {value:.6f}")

    print("\n" + "=" * 96)
    print("LINK ASYMPTOTICS: exposure at which power benefit overtakes log-squared replay harm")
    print("=" * 96)
    for exponent in (0.08, 0.2, 0.34, 0.5, 0.8, 1.0, 1.2):
        crossover = link_crossover(exponent, shape.penalty_threshold)
        location = "never within 1e12" if crossover is None else f"{crossover:.4g}"
        print(f"  exponent {exponent:<6} benefit dominates from exposure {location}")

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "identifiability_audit.csv", index=False)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
