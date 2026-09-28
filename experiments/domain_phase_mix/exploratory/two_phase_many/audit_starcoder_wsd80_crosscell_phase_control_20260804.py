# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///
"""Adversarially audit the exposed WSD80 cross-cell clock diagnostic.

The frozen v3 protocol compared its nested clock selector with a scale-blind
phase model. This post-outcome audit leaves that artifact untouched and asks
the stricter questions that became apparent after independent review: whether
the selector beats a zero-phase predictor, whether the largest-TPP cell
controls the conclusion, and whether total and non-embedding TPP are resolved.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "reference_outputs" / "wsd80_crosscell_phase_control_v3_20260804"
DISCOVERY_CSV = (
    SCRIPT_DIR
    / "reference_outputs"
    / "starcoder_wsd80_matched_nd_stage1_20260731"
    / "stage3_dense_surface_results_20260802"
    / "combined_discovery_observations.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "wsd80_crosscell_phase_control_adversarial_audit_20260804"

ZERO_MODEL = "zero_phase"
BASELINE_MODEL = "lr_dose_plus_taylor"
SELECTOR_MODEL = "nested_clock_selector"
TOTAL_TPP_MODEL = "lr_dose_plus_taylor_total_tpp"
NONEMBEDDING_TPP_MODEL = "lr_dose_plus_taylor_nonembedding_tpp"


def exact_sign_flip_p(differences: np.ndarray, alternative: str) -> float:
    """Return an exact paired sign-flip p-value for the mean difference."""
    observed = float(np.mean(differences))
    statistics = np.asarray(
        [np.mean(differences * signs) for signs in itertools.product((-1.0, 1.0), repeat=len(differences))]
    )
    tolerance = 1e-15
    if alternative == "less":
        return float(np.mean(statistics <= observed + tolerance))
    if alternative == "two-sided":
        return float(np.mean(np.abs(statistics) >= abs(observed) - tolerance))
    raise ValueError(f"Unknown alternative {alternative!r}")


def load_cell_metadata() -> pd.DataFrame:
    """Load one physical metadata row per cell."""
    observations = pd.read_csv(DISCOVERY_CSV)
    metadata = observations.groupby("cell_id", sort=True).first()
    clocks = pd.read_csv(SOURCE_DIR / "optimizer_clock_diagnostics.csv").set_index("cell_id")
    return clocks.join(
        metadata[["materialized_tokens", "total_parameters", "non_embedding_parameters"]],
        how="left",
    )


def correlation_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    """Summarize how strongly total TPP aliases other scale coordinates."""
    rows = []
    for column in (
        "materialized_tokens",
        "total_parameters",
        "non_embedding_parameters",
        "total_steps",
        "nonembedding_tpp",
    ):
        result = spearmanr(metadata["total_tpp"], metadata[column])
        rows.append({"coordinate": column, "spearman_with_total_tpp": float(result.statistic)})
    return pd.DataFrame(rows)


def build_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compute the post-outcome comparisons and decision."""
    cell_metrics = pd.read_csv(SOURCE_DIR / "cell_metrics.csv")
    wide = cell_metrics.pivot(index="cell_id", columns="model", values="rmse")
    required = (ZERO_MODEL, BASELINE_MODEL, SELECTOR_MODEL, TOTAL_TPP_MODEL, NONEMBEDDING_TPP_MODEL)
    missing = set(required) - set(wide.columns)
    if missing:
        raise ValueError(f"Missing models in v3 cell metrics: {sorted(missing)}")

    table = wide.loc[:, list(required)].copy()
    table["selector_minus_zero"] = table[SELECTOR_MODEL] - table[ZERO_MODEL]
    table["selector_over_zero"] = table[SELECTOR_MODEL] / table[ZERO_MODEL]
    table["total_minus_nonembedding"] = table[TOTAL_TPP_MODEL] - table[NONEMBEDDING_TPP_MODEL]
    table = table.reset_index()

    selector_zero = table["selector_minus_zero"].to_numpy()
    total_nonembedding = table["total_minus_nonembedding"].to_numpy()
    worst = table.loc[table["selector_over_zero"].idxmax()]

    pooled = pd.read_csv(SOURCE_DIR / "phase_model_metrics.csv").set_index("model")
    optima = pd.read_csv(SOURCE_DIR / "optimum_diagnostics.csv")
    worst_optimum = optima[(optima["cell_id"] == worst["cell_id"]) & (optima["model"] == SELECTOR_MODEL)]
    if len(worst_optimum) != 1:
        raise ValueError("Expected exactly one selector optimum for the worst cell")
    optimum = worst_optimum.iloc[0]

    metadata = load_cell_metadata()
    correlations = correlation_rows(metadata)
    result = {
        "status": "post_outcome_adversarial_audit",
        "frozen_v3_gate_unchanged": True,
        "selector_beats_zero_cells": int((selector_zero < 0.0).sum()),
        "cells": len(table),
        "selector_minus_zero_mean_cell_rmse": float(selector_zero.mean()),
        "selector_vs_zero_exact_one_sided_sign_flip_p": exact_sign_flip_p(selector_zero, "less"),
        "selector_pooled_rmse": float(pooled.loc[SELECTOR_MODEL, "rmse"]),
        "zero_pooled_rmse": float(pooled.loc[ZERO_MODEL, "rmse"]),
        "selector_pooled_rmse_change_vs_zero": float(
            pooled.loc[SELECTOR_MODEL, "rmse"] / pooled.loc[ZERO_MODEL, "rmse"] - 1.0
        ),
        "worst_cell": str(worst["cell_id"]),
        "worst_cell_selector_rmse": float(worst[SELECTOR_MODEL]),
        "worst_cell_zero_rmse": float(worst[ZERO_MODEL]),
        "worst_cell_selector_over_zero": float(worst["selector_over_zero"]),
        "worst_cell_predicted_gain": float(optimum["predicted_gain"]),
        "worst_cell_reference_gain": float(optimum["reference_gain"]),
        "worst_cell_gain_ratio": float(optimum["predicted_gain"] / optimum["reference_gain"]),
        "worst_cell_optimum_on_support_boundary": bool(optimum["optimum_on_support_boundary"]),
        "total_minus_nonembedding_mean_cell_rmse": float(total_nonembedding.mean()),
        "total_vs_nonembedding_exact_two_sided_sign_flip_p": exact_sign_flip_p(total_nonembedding, "two-sided"),
        "total_tpp_distinct_values": int(metadata["total_tpp"].nunique()),
        "decision": "mixed_diagnostic_not_a_model_license",
        "allowed_inference": (
            "A scale coordinate improves phase-residual prediction in nine of ten cells over the zero-phase null, "
            "but gain magnitude and the highest-TPP extrapolation are not controlled."
        ),
        "forbidden_inference": (
            "Do not encode total TPP in a surrogate or distinguish total from non-embedding TPP from this panel alone."
        ),
    }
    return table, correlations, result


def write_report(
    output_dir: Path,
    cell_table: pd.DataFrame,
    correlations: pd.DataFrame,
    result: dict[str, object],
) -> None:
    """Persist a compact, auditable post-outcome report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_table.to_csv(output_dir / "cell_comparison.csv", index=False)
    correlations.to_csv(output_dir / "tpp_confound_correlations.csv", index=False)
    (output_dir / "decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    display = cell_table[
        ["cell_id", ZERO_MODEL, BASELINE_MODEL, SELECTOR_MODEL, "selector_minus_zero", "selector_over_zero"]
    ].copy()
    lines = [
        "# WSD80 Cross-Cell Phase-Control Adversarial Audit",
        "",
        "This is a post-outcome audit of the frozen v3 diagnostic. It does not alter its preregistration or gate.",
        "",
        "## Decision",
        "",
        "The v3 selector remains evidence that scale matters, but it does **not** license "
        "total TPP as a surrogate term.",
        "",
        f"- It beats the zero-phase predictor in {result['selector_beats_zero_cells']}/{result['cells']} cells.",
        (
            "- Against zero phase, the exact one-sided sign-flip p-value is "
            f"`{result['selector_vs_zero_exact_one_sided_sign_flip_p']:.6f}`."
        ),
        (
            "- Pooled RMSE is "
            f"`{result['selector_pooled_rmse']:.6f}` versus `{result['zero_pooled_rmse']:.6f}` for zero phase "
            f"({100 * result['selector_pooled_rmse_change_vs_zero']:+.1f}%)."
        ),
        (
            f"- The worst cell is `{result['worst_cell']}`: selector RMSE is "
            f"`{result['worst_cell_selector_rmse']:.6f}` versus `{result['worst_cell_zero_rmse']:.6f}` for zero phase."
        ),
        (
            "- That cell's selected optimum is on the observed-support boundary and predicts gain "
            f"`{result['worst_cell_predicted_gain']:.6f}` versus reference `{result['worst_cell_reference_gain']:.6f}` "
            f"({result['worst_cell_gain_ratio']:.1f}x)."
        ),
        (
            "- Total and non-embedding TPP are not distinguished: exact paired two-sided sign-flip "
            f"`p={result['total_vs_nonembedding_exact_two_sided_sign_flip_p']:.6f}`."
        ),
        "",
        "## Cellwise comparison",
        "",
        display.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## TPP confounding",
        "",
        correlations.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Total TPP is strongly aliased with token horizon, optimizer steps, model size, and non-embedding TPP. "
        "A matched-overlap design is required before assigning a causal or transferable clock interpretation.",
        "",
        "## Consequence",
        "",
        str(result["allowed_inference"]),
        "",
        str(result["forbidden_inference"]),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    cell_table, correlations, result = build_audit()
    write_report(args.output_dir, cell_table, correlations, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
