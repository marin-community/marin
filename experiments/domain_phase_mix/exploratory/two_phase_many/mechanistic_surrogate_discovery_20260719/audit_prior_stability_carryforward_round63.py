# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "tabulate>=0.9"]
# ///
"""Carry forward prior-drive complexity and stability evidence with provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
REFERENCE_ROOT = TWO_PHASE_ROOT / "reference_outputs"
SOURCE_DIR = REFERENCE_ROOT / "mechanistic_surrogate_discovery_20260717/final_synthesis"
OUTPUT_ROOT = REFERENCE_ROOT / "mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round63_prior_stability_carryforward"

SOURCE_FILES = (
    "parameter_identifiability.csv",
    "hyperparameter_cross_panel_stability.csv",
    "candidate_active_set_complexity.csv",
    "raw_optimum_convex_support.csv",
    "raw_optimum_crossfit_summary.csv",
    "heldout_calibration_bootstrap.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = []
    frames: dict[str, pd.DataFrame] = {}
    for name in SOURCE_FILES:
        source = SOURCE_DIR / name
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source)
        if frame.empty:
            raise ValueError(f"Prior stability artifact is empty: {source}")
        destination = ROUND_DIR / name
        frame.to_csv(destination, index=False)
        frames[name] = frame
        source_rows.append(
            {
                "artifact": name,
                "source_path": str(source.relative_to(TWO_PHASE_ROOT)),
                "source_sha256": sha256(source),
                "carried_path": str(destination.relative_to(TWO_PHASE_ROOT)),
                "carried_sha256": sha256(destination),
                "rows": len(frame),
                "columns": len(frame.columns),
                "use": "inherited diagnostic evidence only; no model or threshold selected",
            }
        )

    hyperparameters = frames["hyperparameter_cross_panel_stability.csv"]
    complexity = frames["candidate_active_set_complexity.csv"]
    support = frames["raw_optimum_convex_support.csv"]
    crossfit = frames["raw_optimum_crossfit_summary.csv"]

    if len(hyperparameters) != 89 or len(complexity) != 4 or len(support) != 4 or len(crossfit) != 4:
        raise ValueError("Prior stability artifacts do not match the frozen source cardinalities")
    if int((~hyperparameters["cross_panel_stable"].astype(bool)).sum()) != 60:
        raise ValueError("Prior cross-panel stability count changed")
    if not crossfit["fraction_below_observed_frontier"].eq(1.0).all():
        raise ValueError("Prior raw-optimum cross-fit conclusion changed")
    if not support["distance_over_fit_p95"].gt(2.0).all():
        raise ValueError("Prior raw optima are no longer beyond twice the fit-support radius")

    summary = {
        "source_drive": "mechanistic_surrogate_discovery_20260717",
        "use_boundary": "inherited diagnostic evidence only; no model or threshold selected",
        "nonlinear_parameter_pairs": len(hyperparameters),
        "cross_panel_unstable_pairs": int((~hyperparameters["cross_panel_stable"].astype(bool)).sum()),
        "boundary_selected_pairs_at_least_half": int(hyperparameters["boundary_selection_fraction"].ge(0.5).sum()),
        "active_set_effective_df_min": float(complexity["effective_degrees_of_freedom"].min()),
        "active_set_effective_df_max": float(complexity["effective_degrees_of_freedom"].max()),
        "penalized_condition_number_min": float(complexity["penalized_condition_number"].min()),
        "penalized_condition_number_max": float(complexity["penalized_condition_number"].max()),
        "raw_optimum_min_distance_over_fit_p95": float(support["distance_over_fit_p95"].min()),
        "raw_optimum_max_distance_over_fit_p95": float(support["distance_over_fit_p95"].max()),
        "raw_optimum_crossfit_refits_per_policy": int(crossfit["refits"].min()),
        "raw_optimum_crossfit_fraction_below_frontier_min": float(crossfit["fraction_below_observed_frontier"].min()),
        "raw_optimum_crossfit_prediction_sd_min": float(crossfit["sd_predicted_bpb"].min()),
        "raw_optimum_crossfit_prediction_sd_max": float(crossfit["sd_predicted_bpb"].max()),
    }
    pd.DataFrame(source_rows).to_csv(ROUND_DIR / "source_manifest.csv", index=False)
    (ROUND_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = "\n".join(
        [
            "# Round 63: prior stability evidence carry-forward",
            "",
            "This round copies no conclusion by assertion. It verifies and hashes the prior drive's source-level complexity, identifiability, bootstrap, convex-support, and raw-optimum cross-fit tables. The artifacts are inherited diagnostics only; they do not select a new model, hyperparameter, or acceptance threshold and they read no sealed confirmation outcome.",
            "",
            "## Findings",
            "",
            f"- Cross-panel nonlinear hyperparameters classified unstable: {summary['cross_panel_unstable_pairs']}/{summary['nonlinear_parameter_pairs']}.",
            f"- Hyperparameter pairs selecting a grid boundary in at least half of panels: {summary['boundary_selected_pairs_at_least_half']}/{summary['nonlinear_parameter_pairs']}.",
            f"- Active-set ridge effective degrees of freedom: {summary['active_set_effective_df_min']:.2f}--{summary['active_set_effective_df_max']:.2f}; this is a lower bound because nonlinear hyperparameter selection is excluded.",
            f"- Penalized active-set condition numbers: {summary['penalized_condition_number_min']:.3e}--{summary['penalized_condition_number_max']:.3e}.",
            f"- All four frozen raw optima lie {summary['raw_optimum_min_distance_over_fit_p95']:.04f}--{summary['raw_optimum_max_distance_over_fit_p95']:.04f} times beyond the fit-panel 95th-percentile support radius.",
            f"- Across {summary['raw_optimum_crossfit_refits_per_policy']} refits per target and policy class, every refit predicts its raw optimum below the observed frontier; prediction SD is only {summary['raw_optimum_crossfit_prediction_sd_min']:.04f}--{summary['raw_optimum_crossfit_prediction_sd_max']:.04f} BPB.",
            "",
            "## Interpretation",
            "",
            "The optimum failure is not explained by ordinary refit variance. The fitted value assigned to unsupported optima is stable while the policies remain outside empirical support and many nonlinear transition parameters are weakly identified. This is structural extrapolation with parameter equifinality, not a bootstrap-stable mechanistic law.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
