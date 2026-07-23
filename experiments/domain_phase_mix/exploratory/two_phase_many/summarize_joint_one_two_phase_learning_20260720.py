# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
# ]
# ///
"""Summarize identification-aware joint learning of one- and two-phase policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "joint_one_two_phase_learning_20260720"
FAMILY_DIR = REFERENCE_OUTPUTS / "family_state_phase_surrogate_20260720"
FISHER_DIR = REFERENCE_OUTPUTS / "hierarchical_fisher_phase_field_20260720"
SHARED_DIR = REFERENCE_OUTPUTS / "shared_latent_fisher_phase_field_20260720"
INFORMATION_DIR = REFERENCE_OUTPUTS / "phase_identification_information_budget_20260720"
COMPOSITION_DIR = REFERENCE_OUTPUTS / "delphi_3e18_fixed_budget_frontier_composition_20260719"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def candidate_metrics() -> pd.DataFrame:
    family = pd.read_csv(FAMILY_DIR / "stage1_metrics.csv")
    family = family.loc[family["candidate"].eq("family_state_order_information")].copy()
    family["candidate"] = "family_state_5d"
    family = family[
        [
            "target",
            "candidate",
            "pair_rmse",
            "pair_zero_rmse",
            "pair_rmse_ratio",
            "fiber_rmse",
            "fiber_zero_rmse",
            "fiber_rmse_ratio",
            "coefficient_cosine_mean",
        ]
    ].rename(columns={"coefficient_cosine_mean": "stability"})
    family["effective_df"] = 5.0
    family["status"] = "rejected_stage1"

    fisher = pd.read_csv(FISHER_DIR / "stage1_metrics.csv")
    fisher["candidate"] = "hierarchical_fisher_bucket"
    fisher = fisher.rename(columns={"phase_prediction_cosine_mean": "stability"})
    fisher["status"] = "rejected_stage1"

    shared = pd.read_csv(SHARED_DIR / "stage1_metrics.csv")
    shared["candidate"] = "shared_latent_fisher"
    shared = shared.rename(columns={"shared_direction_absolute_cosine_mean": "stability"})
    shared["effective_df"] = 47.0
    shared["status"] = "rejected_stage1_overlaps_prior_jlpt"

    columns = [
        "target",
        "candidate",
        "pair_rmse",
        "pair_zero_rmse",
        "pair_rmse_ratio",
        "fiber_rmse",
        "fiber_zero_rmse",
        "fiber_rmse_ratio",
        "stability",
        "effective_df",
        "status",
    ]
    return pd.concat([family[columns], fisher[columns], shared[columns]], ignore_index=True)


def fixed_budget_summary() -> pd.DataFrame:
    composition = pd.read_csv(COMPOSITION_DIR / "common_archive_comparison.csv")
    selected = {
        "b280_baseline",
        "b140_s70_f70_matched",
        "b100_s80_f100_both",
        "s140_f140_both",
        "b42_s238",
    }
    return composition.loc[composition["allocation"].isin(selected)].copy()


def write_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    ratios = metrics.melt(
        id_vars=["target", "candidate"],
        value_vars=["pair_rmse_ratio", "fiber_rmse_ratio"],
        var_name="evidence_block",
        value_name="rmse_ratio",
    )
    figure = px.bar(
        ratios,
        x="candidate",
        y="rmse_ratio",
        color="target",
        facet_col="evidence_block",
        barmode="group",
        color_discrete_map={"uncheatable": "#2f855a", "table9": "#c53030"},
        title="Phase models improve broad pairs but fail local frontier transfer",
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#243746")
    figure.update_layout(template="plotly_white", xaxis_title="", yaxis_title="nested OOF RMSE / zero-phase RMSE")
    figure.write_html(output_dir / "candidate_phase_rmse_ratios.html", include_plotlyjs=True, config=PLOT_CONFIG)

    budget = pd.DataFrame(
        [
            {"wave": "Wave 1", "block": "one-phase aggregate policies", "checkpoints": 140},
            {"wave": "Wave 2", "block": "exact two-phase counterparts", "checkpoints": 70},
            {"wave": "Wave 2", "block": "signed frontier probes", "checkpoints": 64},
            {"wave": "Wave 2", "block": "frontier center controls", "checkpoints": 6},
        ]
    )
    budget.to_csv(output_dir / "two_wave_budget.csv", index=False)
    colors = {
        "one-phase aggregate policies": "#1f5a75",
        "exact two-phase counterparts": "#d97706",
        "signed frontier probes": "#6b8e23",
        "frontier center controls": "#8c6d5a",
    }
    chart = go.Figure()
    for block, group in budget.groupby("block", sort=False):
        chart.add_bar(
            x=group["wave"],
            y=group["checkpoints"],
            name=block,
            marker_color=colors[block],
            text=group["checkpoints"],
            textposition="inside",
        )
    chart.update_layout(
        template="plotly_white",
        barmode="stack",
        title="Strict 280-checkpoint, two-wave identification design",
        yaxis_title="checkpoints",
    )
    chart.write_html(output_dir / "two_wave_budget.html", include_plotlyjs=True, config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = candidate_metrics()
    composition = fixed_budget_summary()
    signal = pd.read_csv(INFORMATION_DIR / "phase_signal_noise.csv")
    metrics.to_csv(output_dir / "candidate_stage1_metrics.csv", index=False)
    composition.to_csv(output_dir / "fixed_budget_composition_summary.csv", index=False)
    signal.to_csv(output_dir / "phase_signal_noise.csv", index=False)
    write_plots(metrics, output_dir)

    design = {
        "total_checkpoints": 280,
        "sequential_training_waves": 2,
        "wave_1": {
            "one_phase_policies": 140,
            "sampling": (
                "space-filling aggregate-simplex design with deliberate lower-tail and support-boundary coverage"
            ),
        },
        "wave_2": {
            "exact_two_phase_counterparts": 70,
            "counterpart_selection": (
                "D-optimal over phase features, stratified by one-phase performance and support distance; "
                "include but do not exclusively target the frontier"
            ),
            "frontier_probe_checkpoints": 64,
            "frontier_centers": 6,
            "anchors": 2,
            "signed_directions_per_anchor": 16,
            "radii": (
                "two feasible radii approximately 2.6x and 4.1x the current domain-vs-rest radius, "
                "clipped before simplex boundaries"
            ),
            "direction_allocation_per_anchor": {
                "family_axes_at_two_radii": 4,
                "D_optimal_within_family_axes_at_two_radii": 8,
                "mixed_family_bucket_axes_at_large_radius": 4,
            },
        },
        "estimating_equations": {
            "aggregate": "one-phase absolute BPB identifies F_t(a)",
            "global_phase": "same-seed two-minus-one-phase differences identify Delta_t(a,d)",
            "local_odd": "signed plus-minus half-differences identify order benefit",
            "local_even": "signed-pair mean minus same-seed center identifies path curvature or fatigue",
        },
        "dimensionality_rule": (
            "Begin with family-level phase effects; unlock a within-family random-effect block only "
            "when nested held-direction prediction and bootstrap stability pass."
        ),
    }
    (output_dir / "two_wave_design.json").write_text(json.dumps(design, indent=2, sort_keys=True) + "\n")

    registry = pd.DataFrame(
        [
            {
                "id": "FSOP",
                "family": "Family-state orthogonal phase law",
                "new_mechanism": "five family/state/information phase coordinates fit only to pair and fiber moments",
                "status": "rejected_stage1",
                "evidence": "Pair RMSE ratios 0.991/0.949; fiber ratios 0.975/1.000 for Uncheatable/Table-9.",
            },
            {
                "id": "HF-PF",
                "family": "Hierarchical Fisher phase field",
                "new_mechanism": "Fisher-orthogonal bucket random effects with training-fold variance selection",
                "status": "rejected_stage1",
                "evidence": "Pair ratios 0.886/0.855, but Table-9 fiber ratio 1.052 and bucket effective DoF 28.1.",
            },
            {
                "id": "SLF-PF",
                "family": "Shared latent Fisher phase field",
                "new_mechanism": "one cross-target within-family curriculum state",
                "status": "rejected_stage1_overlaps_prior_jlpt",
                "evidence": "Pair ratios 0.850/0.883, but Uncheatable fiber ratio 1.149; special case of prior JLPT.",
            },
            {
                "id": "TWO-WAVE-ID",
                "family": "Two-wave aggregate and phase-contrast acquisition",
                "new_mechanism": "identification design rather than an additional response term",
                "status": "proposed_future_design",
                "evidence": (
                    "Current signed phase SNR is 0.48-0.78 and bucket effective rank is about 8 despite nominal rank 38."
                ),
            },
        ]
    )
    registry.to_csv(output_dir / "approach_registry.csv", index=False)
    ledger = pd.DataFrame(
        [
            {
                "round": "family_state",
                "outcomes_opened": "exact pairs and frontier fibers",
                "historical_absolute_opened": False,
                "adversarial_absolute_opened": False,
                "optimization_run": False,
            },
            {
                "round": "hierarchical_fisher",
                "outcomes_opened": "prior stage1 plus exact pairs and frontier fibers",
                "historical_absolute_opened": False,
                "adversarial_absolute_opened": False,
                "optimization_run": False,
            },
            {
                "round": "shared_latent_fisher",
                "outcomes_opened": "prior stage1 plus exact pairs and frontier fibers",
                "historical_absolute_opened": False,
                "adversarial_absolute_opened": False,
                "optimization_run": False,
            },
        ]
    )
    ledger.to_csv(output_dir / "data_use_ledger.csv", index=False)

    lines = [
        "# Joint one- and two-phase learning",
        "",
        "## Verdict",
        "",
        "The promising direction is an orthogonal aggregate/phase decomposition, but the current phase-fiber design "
        "does not identify a deterministic bucket-level phase law. Two new partial-pooling variants materially "
        "improve broad exact-pair prediction, yet both fail frozen local-fiber transfer. No new model is promoted, "
        "and historical/adversarial absolute heldouts were not opened for these variants.",
        "",
        "The model class should be written as",
        "",
        "$$a=\\alpha w^{(0)}+(1-\\alpha)w^{(1)},\\qquad d=\\alpha(1-\\alpha)(w^{(1)}-w^{(0)}),$$",
        "",
        "$$Y_t(w^{(0)},w^{(1)})=F_t(a)+\\Delta_t(a,d),\\qquad \\Delta_t(a,0)=0.$$",
        "`F_t` is fit from one-phase absolute levels. `Delta_t` is fit from exact pair differences and signed "
        "frontier moments. This cleanly separates aggregate quality from phase order and makes the independently "
        "fitted one-phase model the exact restriction.",
        "",
        "## New local falsification results",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The hierarchical Fisher model retains 10.2 bucket effective DoF on Uncheatable and 28.1 on Table-9. "
        "The shared-latent model has a stable direction and improves pair differences further, but its 1.149 "
        "Uncheatable fiber ratio falsifies the expected local transfer. It also substantially overlaps the prior "
        "JLPT mechanism and is not counted as a new surviving family.",
        "",
        "## Why the current data cannot settle the bucket phase law",
        "",
        signal.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Signed phase effects have SNR below one under the conservative independent-run noise estimate. The 39 "
        "domain-vs-rest probes span rank 38, but their entropy effective rank is only about 8.3 and there is one "
        "observation per direction. A free bucket gradient can interpolate these probes without being identified.",
        "",
        "## What heterogeneous composition already tells us",
        "",
        composition.to_markdown(index=False, floatfmt=".6f"),
        "",
        "At a fixed 280-checkpoint budget, mixed designs can improve broad archive RMSE, but they do not reliably "
        "improve Regret@1 and their raw optima remain unstable. Heterogeneity is useful only when the fitting "
        "equations preserve what each intervention identifies.",
        "",
        "## Recommended two-wave design",
        "",
        "Wave 1 trains 140 one-phase policies to locate and uncertainty-quantify the aggregate frontier. Wave 2 "
        "uses 70 exact two-phase counterparts distributed across performance/support strata, plus 32 signed "
        "frontier directions and six center controls. This is still only two sequential waves: all 140 second-wave "
        "checkpoints are chosen after Wave 1.",
        "",
        "The signed probes use two radii approximately 2.6x and 4.1x the current radius, clipped for simplex "
        "feasibility. Per anchor, use four family-axis probes, eight D-optimal within-family probes, and four mixed "
        "directions. Odd half-differences estimate order benefit; even pair-means relative to centers estimate path "
        "cost. Start with 2-5 family phase DoF and unlock bucket residuals only after held-direction prediction and "
        "bootstrap stability pass.",
        "",
        "This is the main innovation justified by the evidence: learn the one-phase surface and the two-phase lift "
        "with different sufficient statistics, while designing the second wave to make the lift identifiable. It is "
        "preferable to another flexible global surrogate trained on exchangeable absolute rows.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(metrics.to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
