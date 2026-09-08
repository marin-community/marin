# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit SUR-049 and the proposed retained-mass/composition follow-up.

This script does not fit a surrogate. It checks whether the frozen retained
mass has the variation and invariances required to motivate a follow-up that
factors retained state into total mass and normalized composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "reference_outputs" / "effective_budget_equivalence_20260731"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "sur049_retained_composition_audit_20260731"
PAIR_PREDICTIONS = SOURCE_DIR / "pair_predictions.csv"
PROTOCOL = SOURCE_DIR / "protocol.json"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260731
PRIMARY_MIN_AGGREGATE = 0.10
PRIMARY_MAX_AGGREGATE = 0.35
SELECTED_ASYMMETRIC_ROWS = (12, 16, 17, 19, 20, 21)
FIBER_COMPARISONS = {
    "0.18": ((24, 66), (31, 63)),
    "0.30": ((62, 143), (146, 179)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def prediction_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    observed = frame["observed_phase_delta"].to_numpy(dtype=float)
    predicted = frame["predicted_effective_budget_delta"].to_numpy(dtype=float)
    zero_rmse = float(np.sqrt(np.mean(observed**2)))
    model_rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    centered_predicted = predicted - predicted.mean()
    denominator = float(centered_predicted @ centered_predicted)
    calibration = float(centered_predicted @ (observed - observed.mean()) / denominator)
    return zero_rmse - model_rmse, calibration


def clustered_bootstrap(frame: pd.DataFrame, key: str, seed: int) -> dict[str, Any]:
    groups = [group for _, group in frame.groupby(key, sort=True)]
    if len(groups) < 3:
        raise ValueError(f"cluster key {key} has fewer than three groups")

    rng = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAP_DRAWS, 2), dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = [groups[index] for index in rng.integers(0, len(groups), size=len(groups))]
        draws[draw] = prediction_metrics(pd.concat(sampled, ignore_index=True))

    return {
        "groups": len(groups),
        "rmse_improvement_ci95_median": interval(draws[:, 0]),
        "calibration_slope_ci95_median": interval(draws[:, 1]),
    }


def row_values(frame: pd.DataFrame, asymmetric_row: int) -> dict[str, float | int]:
    matches = frame[frame["asymmetric_row"].eq(asymmetric_row)]
    if len(matches) != 1:
        raise ValueError(f"expected one pair for asymmetric row {asymmetric_row}, found {len(matches)}")
    row = matches.iloc[0]
    return {
        "asymmetric_row": asymmetric_row,
        "aggregate": float(row["aggregate_starcoder_nominal"]),
        "retained_mass": float(row["retained_mass_asymmetric"]),
        "log_retained_mass_ratio": float(row["log_retained_mass_ratio"]),
        "predicted_delta": float(row["predicted_effective_budget_delta"]),
        "observed_delta": float(row["observed_phase_delta"]),
    }


def fiber_summary(frame: pd.DataFrame, aggregate: float) -> dict[str, Any]:
    fiber = frame[np.isclose(frame["aggregate_starcoder_nominal"], aggregate)].copy()
    if fiber.empty:
        raise ValueError(f"no rows for aggregate {aggregate}")

    comparisons = [
        [row_values(fiber, left), row_values(fiber, right)] for left, right in FIBER_COMPARISONS[f"{aggregate:.2f}"]
    ]
    correlation = spearmanr(fiber["retained_mass_asymmetric"], fiber["observed_phase_delta"])
    return {
        "pairs": len(fiber),
        "retained_mass_range": [
            float(fiber["retained_mass_asymmetric"].min()),
            float(fiber["retained_mass_asymmetric"].max()),
        ],
        "observed_delta_range": [
            float(fiber["observed_phase_delta"].min()),
            float(fiber["observed_phase_delta"].max()),
        ],
        "retained_mass_observed_delta_spearman": float(correlation.statistic),
        "comparisons": comparisons,
    }


def render_report(summary: dict[str, Any]) -> str:
    selected = pd.DataFrame(summary["selected_near_fixed_mass_rows"])
    selected_table = selected.to_markdown(index=False, floatfmt=".6f")
    aggregate_bootstrap = summary["cluster_bootstrap"]["aggregate"]
    tied_bootstrap = summary["cluster_bootstrap"]["tied_row"]
    fiber_018 = summary["fibers"]["0.18"]
    fiber_030 = summary["fibers"]["0.30"]

    return f"""# SUR-049 Independent Audit and SUR-050 Admissibility Decision

## Decision

- SUR-049's narrow rejection of effective-budget equivalence is preserved.
- The post-outcome suggestion that retained mass may be an even asymmetry-damage proxy is withdrawn.
- The proposed mass/composition follow-up is **blocked before fit** as WSD80-SUR-050.

## Recomputed evidence

The tied retained mass has numerical range
`{summary["tied_mass_range"][0]:.16f}` to `{summary["tied_mass_range"][1]:.16f}`.
It is constant across all {summary["rows"]} pairs, so the retained-mass ratio
is only a shifted version of asymmetric retained mass.

Six policies with nearly fixed retained-mass prediction have observed deltas
spanning `{summary["selected_observed_spread"]:.6f}` BPB while their predictions
span only `{summary["selected_predicted_spread"]:.6f}` BPB:

{selected_table}

At aggregate 0.18, where the observed two-phase optimum lies, retained mass
and observed phase delta have Spearman correlation
`{fiber_018["retained_mass_observed_delta_spearman"]:.3f}` and the same retained
mass occurs with opposite-sign phase deltas. At aggregate 0.30, retained mass
instead tracks even damage strongly (`rho={fiber_030["retained_mass_observed_delta_spearman"]:.3f}`).
The coordinate therefore changes meaning with aggregate and is not a sufficient
phase statistic in either direction.

The original pairwise bootstrap was not cluster-aware. Recomputing by rounded
aggregate gives an RMSE-improvement interval/median of
`{aggregate_bootstrap["rmse_improvement_ci95_median"]}`; recomputing by reused
tied control gives `{tied_bootstrap["rmse_improvement_ci95_median"]}`. Both
remain strictly negative, so the rejection survives while the original
pairwise interval overstates precision.

## Protocol erratum

The frozen protocol says the nominal-versus-realized aggregate correction uses
a "1B tied-only WSD80 BPB" curve. The implementation actually uses tied rows
from the complete WSD80 panel. The implementation is the correct source for
the correction. The frozen protocol is preserved byte-for-byte; the discrepancy
is recorded separately in `protocol_erratum.json`.

## Why SUR-050 is blocked

Let the frozen RPL state be `S`, total retained mass `M=sum(S)`, and normalized
composition `pi=S/M`.

1. `(M, pi)` is a one-to-one coordinate factorization of the same outcome-selected
   RPL state, not a new latent state or transition.
2. At fixed bucketwise aggregate, `S` is a deterministic function of the
   38-dimensional phase contrast. Factoring it into one mass coordinate and
   38 composition coordinates creates no independent intervention that can
   identify mass versus composition.
3. `pi` is exactly the retained-state normalization blocked by SUR-047. Adding
   aggregate conditioning reduces the proposal to the already exposed
   aggregate-fiber regressions PMVT/PWD.
4. The frozen state uses retention and late-multiplier values selected on the
   same 300M outcomes, both at grid boundaries. A positive result would not
   identify the transition law and is nearly guaranteed by SUR-046's existing
   phase-blind ablation.

Reopen only with mass and composition states that are separately physically
measured or independently identified, use units covariant under bucket
refinement, and share their meaning across WSD80 and the 39-bucket panel.
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIR_PREDICTIONS)
    required = {
        "asymmetric_row",
        "aggregate_starcoder_nominal",
        "retained_mass_tied",
        "retained_mass_asymmetric",
        "log_retained_mass_ratio",
        "observed_phase_delta",
        "predicted_effective_budget_delta",
        "tied_row",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pair predictions missing columns: {sorted(missing)}")

    primary = pairs[
        pairs["aggregate_starcoder_nominal"].between(
            PRIMARY_MIN_AGGREGATE - 1e-12,
            PRIMARY_MAX_AGGREGATE + 1e-12,
        )
    ].copy()
    primary["aggregate_cluster"] = primary["aggregate_starcoder_nominal"].round(6)

    selected = pd.DataFrame([row_values(primary, row) for row in SELECTED_ASYMMETRIC_ROWS])
    summary: dict[str, Any] = {
        "decision": "block_sur050_before_fit",
        "rows": len(pairs),
        "primary_rows": len(primary),
        "source_hashes": {
            str(PAIR_PREDICTIONS.relative_to(SCRIPT_DIR)): file_sha256(PAIR_PREDICTIONS),
            str(PROTOCOL.relative_to(SCRIPT_DIR)): file_sha256(PROTOCOL),
        },
        "tied_mass_range": [
            float(pairs["retained_mass_tied"].min()),
            float(pairs["retained_mass_tied"].max()),
        ],
        "selected_near_fixed_mass_rows": selected.to_dict(orient="records"),
        "selected_predicted_spread": float(selected["predicted_delta"].max() - selected["predicted_delta"].min()),
        "selected_observed_spread": float(selected["observed_delta"].max() - selected["observed_delta"].min()),
        "fibers": {
            "0.18": fiber_summary(pairs, 0.18),
            "0.30": fiber_summary(pairs, 0.30),
        },
        "cluster_bootstrap": {
            "aggregate": clustered_bootstrap(primary, "aggregate_cluster", BOOTSTRAP_SEED),
            "tied_row": clustered_bootstrap(primary, "tied_row", BOOTSTRAP_SEED + 1),
        },
    }

    erratum = {
        "frozen_protocol_hash": json.loads(PROTOCOL.read_text())["protocol_hash"],
        "frozen_protocol_file_sha256": file_sha256(PROTOCOL),
        "field": "aggregate_mismatch_correction",
        "recorded_text": (
            "piecewise-linear interpolation of 1B tied-only WSD80 BPB; subtract "
            "F(a_realized_asymmetric)-F(a_realized_tied) from each nominally matched pair delta"
        ),
        "implemented_behavior": (
            "piecewise-linear interpolation of tied rows from the complete WSD80 panel; "
            "subtract F(a_realized_asymmetric)-F(a_realized_tied)"
        ),
        "decision": "implementation is correct; preserve the frozen protocol and record this erratum",
    }

    selected.to_csv(output_dir / "selected_near_fixed_mass_rows.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "protocol_erratum.json").write_text(json.dumps(erratum, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
