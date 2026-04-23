# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate structured epoch-law surrogates on many-domain and Starcoder packets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    STARCODER_TARGET,
    PENALTY_KIND_GROUP_LOG_THRESHOLD,
    PREMIUM_MODE_GLOBAL,
    SIGNAL_KIND_RETAINED_TOTAL,
    SIGNAL_KIND_THRESHOLD_RETAINED_TOTAL,
    SIGNAL_KIND_THRESHOLD_TOTAL,
    SIGNAL_KIND_TOTAL_LOG,
    evaluate_cc_model,
    load_two_phase_many_packet,
    load_two_phase_starcoder_packet,
    optimize_starcoder_family,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "structured_epoch_family_results.csv"
SUMMARY_MD = SCRIPT_DIR / "structured_epoch_family_summary.md"


def evaluate_many_domain() -> pd.DataFrame:
    """Evaluate the many-domain CC-aware models."""
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    cc_threshold_params = {
        "signal_kind": SIGNAL_KIND_THRESHOLD_TOTAL,
        "alpha": 8.0,
        "eta": 5.0,
        "lam": 0.0,
        "sig_tau": 0.25,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "tau": 2.0,
        "reg": 0.01,
    }
    cc_retained_params = {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "alpha": 8.0,
        "eta": 3.0,
        "lam": 1.0,
        "sig_tau": 0.0,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "tau": 1.0,
        "reg": 0.01,
    }
    cc_threshold_row, _ = evaluate_cc_model(data, "CCGlobalPremium-Threshold", cc_threshold_params)
    cc_retained_row, _ = evaluate_cc_model(data, "CCGlobalPremium-RetainedTotal", cc_retained_params)
    return pd.DataFrame([cc_threshold_row, cc_retained_row]).assign(
        dataset="two_phase_many_domains", target=MANY_DOMAIN_TARGET
    )


def evaluate_starcoder() -> pd.DataFrame:
    """Fit and evaluate the 2-domain Starcoder models."""
    data = load_two_phase_starcoder_packet(target=STARCODER_TARGET)
    families = [
        ("TotalLog", SIGNAL_KIND_TOTAL_LOG, False),
        ("RetainedTotal", SIGNAL_KIND_RETAINED_TOTAL, False),
        ("ThresholdTotal", SIGNAL_KIND_THRESHOLD_TOTAL, True),
        ("ThresholdRetainedTotal", SIGNAL_KIND_THRESHOLD_RETAINED_TOTAL, True),
    ]
    rows: list[dict[str, object]] = []
    for name, signal_kind, use_sig_tau in families:
        params = optimize_starcoder_family(data, signal_kind, use_sig_tau=use_sig_tau, seed=0)
        row, _ = evaluate_cc_model(data, name, params)
        # Keep apples-to-apples count with the downloaded Starcoder table.
        row["reported_n_params"] = row["linear_coefficients"]
        row["legacy_reported_n_params"] = row["linear_coefficients"]
        row["dataset"] = "two_phase_starcoder"
        row["target"] = STARCODER_TARGET
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame) -> str:
    """Build a short markdown summary."""
    many = results[results["dataset"] == "two_phase_many_domains"].sort_values("cv_rmse")
    star = results[results["dataset"] == "two_phase_starcoder"].sort_values("cv_rmse")
    return "\n".join(
        [
            "# Structured epoch-law surrogate results",
            "",
            "## Many-domain",
            many[
                [
                    "model",
                    "reported_n_params",
                    "linear_coefficients",
                    "global_shape_parameters",
                    "total_with_shapes",
                    "train_rmse",
                    "cv_rmse",
                    "cv_regret_at_1",
                    "cv_foldmean_regret_at_1",
                ]
            ].to_markdown(index=False),
            "",
            "## Starcoder",
            star[
                [
                    "model",
                    "reported_n_params",
                    "linear_coefficients",
                    "global_shape_parameters",
                    "total_with_shapes",
                    "train_rmse",
                    "cv_rmse",
                    "cv_regret_at_1",
                    "cv_foldmean_regret_at_1",
                ]
            ].to_markdown(index=False),
            "",
            "Reported parameter count matches the legacy tables only for the many-domain rows.",
            "For Starcoder, the legacy scripts counted only fitted linear coefficients and excluded the intercept.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    args = parser.parse_args()

    results = pd.concat([evaluate_many_domain(), evaluate_starcoder()], ignore_index=True)
    results.to_csv(args.output_csv, index=False)
    summary = build_summary(results)
    args.summary_md.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
