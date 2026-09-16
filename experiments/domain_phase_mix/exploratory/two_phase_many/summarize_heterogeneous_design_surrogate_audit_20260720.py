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
"""Summarize the frozen heterogeneous-design HPR batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "heterogeneous_design_surrogate_audit_20260720"
PAIR_DIR = REFERENCE_OUTPUTS / "paired_random_effects_hpr_20260720"
BLOCK_DIR = REFERENCE_OUTPUTS / "heterogeneous_block_gls_hpr_20260720"
PLASTICITY_DIR = REFERENCE_OUTPUTS / "paired_family_plasticity_hpr_20260720"
REGULARIZATION_DIR = REFERENCE_OUTPUTS / "paired_likelihood_regularized_hpr_20260720"
OPTIMUM_DIR = REFERENCE_OUTPUTS / "paired_random_effects_hpr_optimum_audit_20260720"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    local = metrics.loc[
        metrics["candidate"].isin(["pooled_identity", "paired_random_effects_shared"])
        & metrics["scope"].isin(["train_oof", "common_all", "adversarial_target_matched"])
    ].copy()
    local["calibration_distance"] = np.abs(local["calibration_slope"] - 1.0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for keys, group in local.groupby(["target", "allocation", "scope"], sort=True):
        pivot = group.pivot(index="seed", columns="candidate")
        for metric in ("rmse", "calibration_distance", "regret_at_1", "optimism_gt_0p05", "worst_optimism"):
            difference = (pivot[metric]["paired_random_effects_shared"] - pivot[metric]["pooled_identity"]).to_numpy(
                dtype=float
            )
            samples = difference[rng.integers(0, len(difference), size=(BOOTSTRAP_SAMPLES, len(difference)))].mean(
                axis=1
            )
            rows.append(
                {
                    "target": keys[0],
                    "allocation": keys[1],
                    "scope": keys[2],
                    "metric": metric,
                    "paired_minus_pooled": float(difference.mean()),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def optimum_summary(optima: pd.DataFrame) -> pd.DataFrame:
    selected = optima.loc[optima["candidate"].isin(["pooled_identity", "paired_random_effects_shared"])]
    return (
        selected.groupby(["target", "candidate"], sort=True)
        .agg(
            replicates=("seed", "size"),
            predicted_bpb=("predicted_bpb", "mean"),
            max_bucket_weight=("max_bucket_weight", "mean"),
            max_simulated_epochs_mean=("max_simulated_epochs", "mean"),
            max_simulated_epochs_max=("max_simulated_epochs", "max"),
            phase_total_variation=("phase_total_variation", "mean"),
            fit_support_distance=("fit_support_distance", "mean"),
            nearest_common_policy_tv=("nearest_common_policy_tv", "mean"),
            convergence_rate=("optimizer_converged", "mean"),
        )
        .reset_index()
    )


def route_comparison(
    pair_metrics: pd.DataFrame,
    block_metrics: pd.DataFrame,
    plasticity_metrics: pd.DataFrame,
    regularization_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    pair_summary = pair_metrics.loc[
        pair_metrics["scope"].eq("common_all")
        & pair_metrics["allocation"].eq("p140")
        & pair_metrics["candidate"].isin(["pooled_identity", "paired_random_effects_shared"])
    ]
    for target, group in pair_summary.groupby("target"):
        means = group.groupby("candidate")["rmse"].mean()
        relative = means["paired_random_effects_shared"] / means["pooled_identity"] - 1.0
        rows.append(
            {
                "route": "PHPR-GLS",
                "target": target,
                "status": "fitting_only",
                "evidence": f"p140 common RMSE change {relative:+.1%}; raw optimization gate failed",
            }
        )
    block = block_metrics.loc[
        block_metrics["scope"].eq("common_all")
        & block_metrics["allocation"].eq("p90_f100_matched")
        & block_metrics["candidate"].isin(["pair_only_random_effects", "unified_block_random_effects"])
    ]
    for target, group in block.groupby("target"):
        means = group.groupby("candidate")["rmse"].mean()
        change = means["unified_block_random_effects"] - means["pair_only_random_effects"]
        rows.append(
            {
                "route": "HBGLS-HPR",
                "target": target,
                "status": "rejected",
                "evidence": f"fiber-block covariance changes common RMSE by {change:+.6f} BPB and not Regret@1",
            }
        )
    plasticity = plasticity_metrics.loc[
        plasticity_metrics["scope"].eq("common_all")
        & plasticity_metrics["allocation"].eq("p140")
        & plasticity_metrics["candidate"].isin(
            ["paired_random_effects_shared", "paired_random_effects_family_coverage_gain"]
        )
    ]
    for target, group in plasticity.groupby("target"):
        means = group.groupby("candidate")["rmse"].mean()
        change = means["paired_random_effects_family_coverage_gain"] - means["paired_random_effects_shared"]
        rows.append(
            {
                "route": "FP-HPR",
                "target": target,
                "status": "rejected",
                "evidence": f"stable family-plasticity terms worsen common RMSE by {change:+.6f} BPB",
            }
        )
    regularization = regularization_metrics.loc[
        regularization_metrics["scope"].eq("common_all")
        & regularization_metrics["target"].eq("uncheatable")
        & regularization_metrics["candidate"].isin(["legacy_regularization", "paired_likelihood_regularization"])
    ]
    for allocation, group in regularization.groupby("allocation"):
        means = group.groupby("candidate")["rmse"].mean()
        change = means["paired_likelihood_regularization"] - means["legacy_regularization"]
        rows.append(
            {
                "route": "PLR-HPR",
                "target": f"uncheatable/{allocation}",
                "status": "rejected",
                "evidence": f"training-selected regularization worsens common RMSE by {change:+.6f} BPB",
            }
        )
    return pd.DataFrame(rows)


def acceptance_gate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route": "PHPR-GLS",
                "criterion": "material heldout failure improvement",
                "status": "pass",
                "evidence": "Common RMSE and optimism improve, especially Table-9.",
            },
            {
                "route": "PHPR-GLS",
                "criterion": "OOF RMSE within 5 percent",
                "status": "pass",
                "evidence": "The paired likelihood preserves grouped OOF accuracy.",
            },
            {
                "route": "PHPR-GLS",
                "criterion": "Regret@1 degradation <= 0.002 BPB",
                "status": "pass",
                "evidence": "p140 Regret@1 is unchanged on both targets.",
            },
            {
                "route": "PHPR-GLS",
                "criterion": "calibration moves toward one on both targets",
                "status": "fail",
                "evidence": "Table-9 improves, but Uncheatable slopes move farther above one.",
            },
            {
                "route": "PHPR-GLS",
                "criterion": "plausible bootstrap-stable raw optimum",
                "status": "fail",
                "evidence": (
                    "No optimization converged; Table-9 reaches 391 epochs in one resample and all optima remain "
                    "far outside support."
                ),
            },
            {
                "route": "HBGLS-HPR",
                "criterion": "incremental benefit beyond pair GLS",
                "status": "fail",
                "evidence": (
                    "Fiber-block covariance has negligible or adverse changes and does not improve decision regret."
                ),
            },
            {
                "route": "FP-HPR",
                "criterion": "mechanism transfers beyond paired deltas",
                "status": "fail",
                "evidence": "Coefficients are stable but broad and adversarial heldouts worsen on both targets.",
            },
            {
                "route": "PLR-HPR",
                "criterion": "training-only hyperparameter gain transfers",
                "status": "fail",
                "evidence": "OOF improves slightly while common/adversarial RMSE and t42+p119 regret worsen.",
            },
        ]
    )


def approach_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "PHPR-GLS",
                "family": "Paired random-effects HPR",
                "new_invariant": (
                    "Exact aggregate-matched policies sharing a data seed share a random observation effect."
                ),
                "equation": "Y[j,r] = f_HPR(w[j,r]) + u[j] + epsilon[j,r]",
                "additional_dof": 0,
                "status": "fitting_only",
                "evidence": (
                    "Material RMSE/optimism gain, but calibration is mixed and raw optima fail stability/support gates."
                ),
            },
            {
                "id": "HBGLS-HPR",
                "family": "Heterogeneous block-GLS HPR",
                "new_invariant": "Same-seed phase-fiber observations share the center seed effect.",
                "equation": "Y[b,r] = f_HPR(w[b,r]) + u[b] + epsilon[b,r]",
                "additional_dof": 0,
                "status": "rejected",
                "evidence": "No material increment over pair-only GLS on unused fibers or broad heldouts.",
            },
            {
                "id": "FP-HPR",
                "family": "Family-plasticity HPR",
                "new_invariant": "Phase order rescales retained family coverage relative to its tied counterfactual.",
                "equation": "Y = f_HPR(w) + sum_f delta[f] (C[f](w)-C[f](w_tied))",
                "additional_dof": "1 global or 3 family coefficients",
                "status": "rejected",
                "evidence": "Stable active coefficients fit pair deltas but fail broad and adversarial transfer.",
            },
            {
                "id": "PLR-HPR",
                "family": "Paired-likelihood-selected HPR regularization",
                "new_invariant": "Coefficient shrinkage is selected under the same paired likelihood used for fitting.",
                "equation": "Same PHPR-GLS equation; only ridge scales change.",
                "additional_dof": 0,
                "status": "rejected",
                "evidence": "Training-only gain does not transfer to exposed development outcomes.",
            },
        ]
    )


def data_use_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "round": 1,
                "candidate": "PHPR-GLS",
                "selection_data": "Exact-pair structure and nested fit-panel OOF residual covariance",
                "development_outcomes_inspected_before": "Prior historical/adversarial baseline failures",
                "outcomes_inspected_after_freeze": (
                    "Historical archive, adversarial panel, unused matched pairs, raw optima"
                ),
                "decision": "Retain as fitting-only improvement; block as headline surrogate",
            },
            {
                "round": 2,
                "candidate": "FP-HPR",
                "selection_data": "Matched-pair equations; family coverage mechanism preregistered",
                "development_outcomes_inspected_before": "PHPR-GLS outcomes",
                "outcomes_inspected_after_freeze": "Historical archive, adversarial panel, unused matched pairs",
                "decision": "Reject",
            },
            {
                "round": 3,
                "candidate": "HBGLS-HPR",
                "selection_data": "Pair covariance plus same-seed fiber block identity",
                "development_outcomes_inspected_before": "PHPR-GLS and prior fiber outcomes",
                "outcomes_inspected_after_freeze": "Historical archive, adversarial panel, unused fibers",
                "decision": "Reject",
            },
            {
                "round": 4,
                "candidate": "PLR-HPR",
                "selection_data": "Eight p140 training-only paired-likelihood OOF resamples",
                "development_outcomes_inspected_before": (
                    "All previous rounds; no heldout value used for hyperparameter selection"
                ),
                "outcomes_inspected_after_freeze": "Historical archive and adversarial panel",
                "decision": "Reject",
            },
        ]
    )


def render_effects(effects: pd.DataFrame, output_dir: Path) -> None:
    local = effects.loc[
        effects["metric"].eq("rmse") & effects["scope"].isin(["common_all", "adversarial_target_matched"])
    ]
    figure = px.bar(
        local,
        x="paired_minus_pooled",
        y="allocation",
        color="paired_minus_pooled",
        facet_row="target",
        facet_col="scope",
        orientation="h",
        color_continuous_scale="RdYlGn_r",
        title="Pair random-effects HPR minus pooled HPR RMSE",
    )
    figure.add_vline(x=0.0, line_dash="dot", line_color="#333")
    figure.update_layout(template="plotly_white", width=1500, height=800)
    figure.write_html(output_dir / "paired_rmse_effects.html", include_plotlyjs="cdn")


def write_report(
    effects: pd.DataFrame,
    optima: pd.DataFrame,
    routes: pd.DataFrame,
    gate: pd.DataFrame,
    output_dir: Path,
) -> None:
    common_rmse = effects.loc[effects["scope"].eq("common_all") & effects["metric"].eq("rmse")]
    lines = [
        "# Heterogeneous-design surrogate audit",
        "",
        "## Verdict",
        "",
        "The exact same-seed pair structure is useful, but only as an observation model. Paired random-effects GLS ",
        "is the sole surviving fitting improvement: it removes shared seed noise without adding response degrees of ",
        "freedom. It is not a new headline surrogate. Its raw HPR optimum remains unsupported and unstable, so no ",
        "candidate from this batch is recommended for validation or deployment.",
        "",
        "The frontier-fiber blocks add almost no transferable information through covariance alone. Explicit family ",
        "plasticity fits the designed contrasts but worsens independent policies, and likelihood-selected coefficient ",
        "regularization repeats the familiar OOF-to-heldout reversal.",
        "",
        "## Pair-GLS effects",
        "",
        "Negative values favor paired GLS. Intervals are paired bootstraps over the eight fixed-budget resamples.",
        "",
        common_rmse.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optimization audit",
        "",
        optima.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Route decisions",
        "",
        routes.to_markdown(index=False),
        "",
        "## Frozen gate",
        "",
        gate.to_markdown(index=False),
        "",
        "## Modeling implication",
        "",
        "A heterogeneous acquisition design should retain exact one-/two-phase pairs and same-seed blocks in the ",
        "likelihood. However, the current fibers do not identify a phase-response mechanism that transfers away from ",
        "their anchors. The next scientifically justified step is not another coefficient correction: it requires a ",
        "new latent transition law whose phase response is identified on both StarCoder schedules before any further ",
        "3e18 development evaluation.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_metrics = pd.read_csv(PAIR_DIR / "metric_runs.csv")
    block_metrics = pd.read_csv(BLOCK_DIR / "metric_runs.csv")
    plasticity_metrics = pd.read_csv(PLASTICITY_DIR / "metric_runs.csv")
    regularization_metrics = pd.read_csv(REGULARIZATION_DIR / "metric_runs.csv")
    optima = optimum_summary(pd.read_csv(OPTIMUM_DIR / "raw_optima.csv"))
    effects = paired_effects(pair_metrics)
    routes = route_comparison(pair_metrics, block_metrics, plasticity_metrics, regularization_metrics)
    gate = acceptance_gate()

    effects.to_csv(args.output_dir / "paired_bootstrap_effects.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optimum_summary.csv", index=False)
    routes.to_csv(args.output_dir / "route_comparison.csv", index=False)
    gate.to_csv(args.output_dir / "acceptance_gate_evaluation.csv", index=False)
    approach_registry().to_csv(args.output_dir / "approach_registry.csv", index=False)
    data_use_ledger().to_csv(args.output_dir / "data_use_ledger.csv", index=False)
    render_effects(effects, args.output_dir)
    write_report(effects, optima, routes, gate, args.output_dir)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
