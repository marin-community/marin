# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "plotly>=6.0", "scipy>=1.15", "tabulate>=0.9"]
# ///
"""Quantify support-based abstention as a deployment constraint, not a surrogate fix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719.audit_partial_identification_round53 import (  # noqa: E402
    heldout_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719.freeze_pareto_gate import (  # noqa: E402
    BASELINE_MODELS,
    DEFAULT_DASHBOARD,
    metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round68_support_abstention"
COVERAGE_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def coverage_metrics(frame: pd.DataFrame, target: str, model: str, coverage: float) -> dict[str, object]:
    count = max(3, int(np.ceil(coverage * len(frame))))
    retained = frame.nsmallest(count, "support_distance", keep="first")
    observed = retained["observed"].to_numpy(dtype=float)
    predicted = retained[f"prediction::{model}"].to_numpy(dtype=float)
    summary, _ = metrics(observed, predicted)
    selected = int(np.argmin(predicted))
    return {
        "target": target,
        "model": model,
        "coverage_fraction": coverage,
        "retained_rows": len(retained),
        "maximum_support_distance": float(retained["support_distance"].max()),
        **summary,
        "global_archive_regret_at_1": float(observed[selected] - frame["observed"].min()),
    }


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(DEFAULT_DASHBOARD.read_text())
    rows = []
    for target in ("uncheatable", "table9"):
        frame = heldout_frame(bundle, target).sort_values("support_distance", kind="stable")
        for model in BASELINE_MODELS:
            for coverage in COVERAGE_FRACTIONS:
                rows.append(coverage_metrics(frame, target, model, coverage))
    results = pd.DataFrame(rows)
    results.to_csv(ROUND_DIR / "support_abstention_metrics.csv", index=False)

    safe = (
        results.loc[results["optimism_gt_0p05_count"].eq(0)]
        .sort_values("coverage_fraction")
        .groupby(["target", "model"], as_index=False)
        .last()[
            [
                "target",
                "model",
                "coverage_fraction",
                "retained_rows",
                "maximum_support_distance",
                "rmse",
                "regret_at_1",
                "global_archive_regret_at_1",
                "worst_optimism",
            ]
        ]
        .rename(columns={"coverage_fraction": "maximum_tested_coverage_without_severe_optimism"})
    )
    safe.to_csv(ROUND_DIR / "maximum_safe_coverage.csv", index=False)

    figure = px.line(
        results,
        x="coverage_fraction",
        y="rmse",
        color="model",
        markers=True,
        facet_col="target",
        title="Support abstention reduces error but does not identify the response surface",
        labels={"coverage_fraction": "Nearest-support archive coverage", "rmse": "Heldout RMSE"},
    )
    figure.update_layout(template="plotly_white", height=560, width=1200)
    figure.write_html(ROUND_DIR / "support_abstention_rmse.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    figure = px.line(
        results,
        x="coverage_fraction",
        y="global_archive_regret_at_1",
        color="model",
        markers=True,
        facet_col="target",
        title="Abstention trades extrapolation risk for missed global opportunities",
        labels={
            "coverage_fraction": "Nearest-support archive coverage",
            "global_archive_regret_at_1": "Selected regret against full archive",
        },
    )
    figure.update_layout(template="plotly_white", height=560, width=1200)
    figure.write_html(ROUND_DIR / "support_abstention_regret.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report_rows = results.loc[results["coverage_fraction"].isin([0.25, 0.50, 1.00])]
    report = "\n".join(
        [
            "# Round 68: support-abstention tradeoff",
            "",
            "This diagnostic freezes the existing support distance and baseline predictions, retains the nearest fixed fractions of the 710-run archive, and recomputes metrics. It fits no response term, tunes no threshold, and reads no sealed confirmation outcome.",
            "",
            "## Fixed coverage levels",
            "",
            report_rows[
                [
                    "target",
                    "model",
                    "coverage_fraction",
                    "retained_rows",
                    "rmse",
                    "calibration_slope_observed_on_predicted",
                    "optimism_gt_0p05_count",
                    "worst_optimism",
                    "regret_at_1",
                    "global_archive_regret_at_1",
                ]
            ].to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Maximum tested coverage without optimism above 0.05 BPB",
            "",
            safe.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Interpretation",
            "",
            "Nearest-support abstention lowers RMSE and usually removes catastrophic optimism, but it does not repair calibration or identify the global optimum. Restricting coverage can increase regret against the full archive because strong observed policies may themselves be out of support. The admissible use is therefore operational: report an abstention envelope or constrain deployment. Support distance must not be inserted into the BPB response equation or counted as evidence that a mechanistic surrogate is correct.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
