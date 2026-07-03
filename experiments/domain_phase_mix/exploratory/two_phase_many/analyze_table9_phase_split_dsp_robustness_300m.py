# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Robustness diagnostics for simple phase-separated Table-9 DSP variants.

This script is intentionally narrower than
``analyze_table9_phase_split_dsp_300m.py``. It checks whether the phase-split
variant survives three low-cost stress tests before any 3e18 validation run:

1. OLMoBaseEval Table-9 proportional-repeat noise floor.
2. Multi-seed nested CV for selected tied-vs-split DSP settings.
3. A gamma-saturation profile for the split model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_split_dsp_300m as phase_split,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "reference_outputs"
    / "table9_dsp_phase_functional_form_20260630"
    / "robustness"
)
MACRO_TARGET = "table9_macro_bpb"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class NoiseFloorSummary:
    n_rows: int
    macro_mean: float
    macro_std: float
    macro_sem: float
    component_std_median: float
    component_std_q90: float
    component_std_max: float


@dataclass(frozen=True)
class BudgetSummary:
    pctrl_fit_target_budget: int
    delphi_simulated_epoch_target_budget: int
    budgets_match: bool


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", default="effective_exposure,split_saturation_penalty")
    parser.add_argument("--linear-reg-values", default="0.0001,0.01")
    parser.add_argument("--cv-seeds", default="0,1,2,3,4")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--gamma-profile-linear-reg-values", default="0.0001,0.01")
    parser.add_argument("--gamma-profile-points", type=int, default=36)
    return parser.parse_args()


def load_table9_context() -> tuple[pd.DataFrame, dict[str, object], list[str], list[str], np.ndarray, dsp.PacketData]:
    signal, columns, domains, _natural = base.load_raw_signal_panel()
    del signal
    token_counts = base.load_domain_token_counts(domains)
    panel, metadata = paper_olmix.build_fit_panel(columns)
    packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, MACRO_TARGET)
    return panel, metadata, columns, domains, token_counts, packet


def proportional_noise_floor(components: list[str]) -> tuple[pd.DataFrame, NoiseFloorSummary]:
    olmo = paper_olmix.load_olmo_wide_with_table9_components()
    proportional = olmo[
        olmo["run_name"].eq("baseline_proportional") | olmo["panel"].eq("proportional_noise")
    ].copy()
    if len(proportional) != 11:
        raise ValueError(f"Expected 11 proportional rows, found {len(proportional)}")
    proportional[MACRO_TARGET] = proportional[components].mean(axis=1)
    component_std = proportional[components].std(axis=0, ddof=1)
    summary = NoiseFloorSummary(
        n_rows=int(len(proportional)),
        macro_mean=float(proportional[MACRO_TARGET].mean()),
        macro_std=float(proportional[MACRO_TARGET].std(ddof=1)),
        macro_sem=float(proportional[MACRO_TARGET].std(ddof=1) / np.sqrt(len(proportional))),
        component_std_median=float(component_std.median()),
        component_std_q90=float(component_std.quantile(0.9)),
        component_std_max=float(component_std.max()),
    )
    return proportional[["run_name", "panel", MACRO_TARGET, *components]].copy(), summary


def budget_summary() -> BudgetSummary:
    pctrl_budget = base.load_target_budget()
    delphi_budget = int(TARGET_BUDGET_DOLMA3_COMMON_CRAWL)
    return BudgetSummary(
        pctrl_fit_target_budget=int(pctrl_budget),
        delphi_simulated_epoch_target_budget=delphi_budget,
        budgets_match=bool(pctrl_budget == delphi_budget),
    )


def multi_seed_nested_cv(
    *,
    panel: pd.DataFrame,
    components: list[str],
    packet: dsp.PacketData,
    variants: list[str],
    linear_regs: list[float],
    cv_seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for linear_reg in linear_regs:
        for variant_key in variants:
            print(f"Fitting full model for {variant_key} L2={linear_reg:g}", flush=True)
            full_model, _tuning = phase_split.fit_variant_with_l2(
                packet,
                variant_key,
                linear_reg,
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
                basin_hopping_iters=basin_hopping_iters,
            )
            train_pred = dsp.predict(full_model, packet.w)
            for seed in cv_seeds:
                print(f"  seed {seed}", flush=True)
                folds = component_dsp.panel_stratified_folds(panel, n_splits=n_splits, seed=seed)
                fixed_oof = phase_split.fixed_param_oof(packet, full_model, folds)
                nested_oof, nested_models = phase_split.nested_oof(
                    packet,
                    variant_key,
                    linear_reg,
                    folds,
                    maxiter=maxiter,
                    coarse_top_k=coarse_top_k,
                    basin_hopping_iters=basin_hopping_iters,
                )
                summary = phase_split.summarize_variant(
                    panel=panel,
                    components=components,
                    packet=packet,
                    variant_key=variant_key,
                    linear_reg=linear_reg,
                    full_model=full_model,
                    train_pred=train_pred,
                    fixed_oof_pred=fixed_oof,
                    nested_oof_pred=nested_oof,
                    nested_models=nested_models,
                    folds=folds,
                )
                row = asdict(summary)
                row["cv_seed"] = int(seed)
                rows.append(row)
                pred = panel[["run_name", "panel_source", MACRO_TARGET]].copy()
                pred = pred.rename(columns={MACRO_TARGET: "actual"})
                pred["variant_key"] = variant_key
                pred["linear_reg"] = float(linear_reg)
                pred["cv_seed"] = int(seed)
                pred["nested_oof_prediction"] = nested_oof
                pred["nested_oof_residual"] = nested_oof - packet.y
                predictions.append(pred)
    per_seed = pd.DataFrame.from_records(rows)
    grouped = per_seed.groupby(["variant_key", "linear_reg"], dropna=False)
    aggregate = grouped.agg(
        nested_oof_rmse_mean=("nested_oof_rmse", "mean"),
        nested_oof_rmse_std=("nested_oof_rmse", "std"),
        nested_oof_spearman_mean=("nested_oof_spearman", "mean"),
        nested_oof_spearman_std=("nested_oof_spearman", "std"),
        nested_fold_mean_regret_at_1_mean=("nested_fold_mean_regret_at_1", "mean"),
        nested_fold_mean_regret_at_1_std=("nested_fold_mean_regret_at_1", "std"),
        nested_fold_mean_regret_at_3_mean=("nested_fold_mean_regret_at_3", "mean"),
        nested_fold_mean_regret_at_3_std=("nested_fold_mean_regret_at_3", "std"),
        nested_lower_tail_optimism_mean=("nested_lower_tail_optimism", "mean"),
        nested_lower_tail_optimism_std=("nested_lower_tail_optimism", "std"),
        selected_actual_bpb_mean=("nested_selected_actual_bpb", "mean"),
        selected_actual_bpb_std=("nested_selected_actual_bpb", "std"),
        selected_component_harm_count_mean=("nested_selected_component_harm_count", "mean"),
        selected_component_harm_count_std=("nested_selected_component_harm_count", "std"),
    ).reset_index()
    return per_seed, pd.concat(predictions, ignore_index=True), aggregate


def gamma_saturation_profile(
    *,
    panel: pd.DataFrame,
    packet: dsp.PacketData,
    linear_regs: list[float],
    gamma_points: int,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> pd.DataFrame:
    folds = component_dsp.panel_stratified_folds(panel, n_splits=5, seed=0)
    gamma_values = np.geomspace(1.0, 100.0, gamma_points)
    rows: list[dict[str, object]] = []
    for linear_reg in linear_regs:
        print(f"Gamma profile split_saturation_penalty L2={linear_reg:g}", flush=True)
        full_model, _tuning = phase_split.fit_variant_with_l2(
            packet,
            "split_saturation_penalty",
            linear_reg,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        full_gamma = float(full_model.params["gamma_saturation"])
        for gamma in gamma_values:
            params = dict(full_model.params)
            params["gamma_saturation"] = float(gamma)
            profiled = dsp.fit_linear_head(
                packet.w,
                packet.y,
                packet,
                full_model.variant,
                params,
            )
            train_pred = dsp.predict(profiled, packet.w)
            fixed_oof = phase_split.fixed_param_oof(packet, profiled, folds)
            train_rmse, train_spearman = phase_split.regression_metrics(packet.y, train_pred)
            oof_rmse, oof_spearman = phase_split.regression_metrics(packet.y, fixed_oof)
            oof_optimism, oof_low_tail_rmse = phase_split.lower_tail_optimism(packet.y, fixed_oof)
            rows.append(
                {
                    "linear_reg": float(linear_reg),
                    "gamma_saturation": float(gamma),
                    "full_fit_gamma_saturation": full_gamma,
                    "train_rmse": train_rmse,
                    "train_spearman": train_spearman,
                    "fixed_oof_rmse": oof_rmse,
                    "fixed_oof_spearman": oof_spearman,
                    "fixed_global_regret_at_1": phase_split.global_regret_at_k(packet.y, fixed_oof, 1),
                    "fixed_lower_tail_optimism": oof_optimism,
                    "fixed_low_tail_rmse": oof_low_tail_rmse,
                }
            )
    return pd.DataFrame.from_records(rows)


def write_multiseed_plot(aggregate: pd.DataFrame, output_path: Path) -> None:
    fig = go.Figure()
    for metric, axis_name in [
        ("nested_oof_rmse", "Nested OOF RMSE"),
        ("nested_fold_mean_regret_at_1", "Nested fold Regret@1"),
        ("nested_lower_tail_optimism", "Lower-tail optimism"),
    ]:
        for variant_key, group in aggregate.groupby("variant_key", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=group["linear_reg"].map(lambda value: f"{float(value):g}"),
                    y=group[f"{metric}_mean"],
                    error_y={"type": "data", "array": group[f"{metric}_std"].fillna(0.0)},
                    mode="markers+lines",
                    name=f"{variant_key}: {axis_name}",
                    visible=metric == "nested_oof_rmse",
                )
            )
    buttons = []
    trace_count_per_metric = int(aggregate["variant_key"].nunique())
    metrics = ["nested_oof_rmse", "nested_fold_mean_regret_at_1", "nested_lower_tail_optimism"]
    for metric_idx, label in enumerate(["RMSE", "Regret@1", "Optimism"]):
        visible = [False] * (trace_count_per_metric * len(metrics))
        start = metric_idx * trace_count_per_metric
        for idx in range(start, start + trace_count_per_metric):
            visible[idx] = True
        buttons.append({"label": label, "method": "update", "args": [{"visible": visible}, {"yaxis.title.text": label}]})
    fig.update_layout(
        title="Table-9 phase-DSP robustness over CV seeds",
        xaxis_title="Linear L2",
        yaxis_title="Nested OOF RMSE",
        template="plotly_white",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.15}],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_gamma_plot(profile: pd.DataFrame, output_path: Path) -> None:
    fig = go.Figure()
    for linear_reg, group in profile.groupby("linear_reg", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["gamma_saturation"],
                y=group["fixed_oof_rmse"],
                mode="lines+markers",
                name=f"L2={float(linear_reg):g}",
                hovertemplate="gamma_sat=%{x:.3g}<br>fixed OOF RMSE=%{y:.6f}<extra></extra>",
            )
        )
        full_gamma = float(group["full_fit_gamma_saturation"].iloc[0])
        fig.add_vline(x=full_gamma, line_dash="dot", annotation_text=f"fit gamma L2={float(linear_reg):g}")
    fig.update_xaxes(type="log", title="gamma_saturation")
    fig.update_layout(
        title="Split-phase DSP gamma_saturation profile",
        yaxis_title="Fixed-nonlinear OOF RMSE after refitting linear head",
        template="plotly_white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_str_list(args.variants)
    invalid = sorted(set(variants).difference(dsp.VARIANTS))
    if invalid:
        raise ValueError(f"Unknown DSP variants: {invalid}")
    linear_regs = parse_float_list(args.linear_reg_values)
    cv_seeds = parse_int_list(args.cv_seeds)
    gamma_linear_regs = parse_float_list(args.gamma_profile_linear_reg_values)

    panel, metadata, _columns, _domains, _token_counts, packet = load_table9_context()
    components = list(metadata["components"])

    prop_rows, prop_summary = proportional_noise_floor(components)
    prop_rows.to_csv(args.output_dir / "proportional_repeat_table9_rows.csv", index=False)
    pd.DataFrame([asdict(prop_summary)]).to_csv(args.output_dir / "proportional_repeat_noise_floor.csv", index=False)

    budget = budget_summary()
    pd.DataFrame([asdict(budget)]).to_csv(args.output_dir / "budget_consistency.csv", index=False)

    per_seed, predictions, aggregate = multi_seed_nested_cv(
        panel=panel,
        components=components,
        packet=packet,
        variants=variants,
        linear_regs=linear_regs,
        cv_seeds=cv_seeds,
        n_splits=int(args.n_splits),
        maxiter=int(args.maxiter),
        coarse_top_k=int(args.coarse_top_k),
        basin_hopping_iters=int(args.basin_hopping_iters),
    )
    per_seed.to_csv(args.output_dir / "phase_variant_multiseed_nested_cv_per_seed.csv", index=False)
    predictions.to_csv(args.output_dir / "phase_variant_multiseed_nested_cv_predictions.csv", index=False)
    aggregate.to_csv(args.output_dir / "phase_variant_multiseed_nested_cv_aggregate.csv", index=False)
    write_multiseed_plot(aggregate, args.output_dir / "phase_variant_multiseed_nested_cv.html")

    profile = gamma_saturation_profile(
        panel=panel,
        packet=packet,
        linear_regs=gamma_linear_regs,
        gamma_points=int(args.gamma_profile_points),
        maxiter=int(args.maxiter),
        coarse_top_k=int(args.coarse_top_k),
        basin_hopping_iters=int(args.basin_hopping_iters),
    )
    profile.to_csv(args.output_dir / "split_gamma_saturation_profile.csv", index=False)
    write_gamma_plot(profile, args.output_dir / "split_gamma_saturation_profile.html")

    report = {
        "output_dir": str(args.output_dir),
        "variants": variants,
        "linear_regs": linear_regs,
        "cv_seeds": cv_seeds,
        "noise_floor": asdict(prop_summary),
        "budget": asdict(budget),
        "best_multiseed_rmse": aggregate.sort_values("nested_oof_rmse_mean").iloc[0].to_dict(),
        "best_multiseed_regret_at_1": aggregate.sort_values("nested_fold_mean_regret_at_1_mean").iloc[0].to_dict(),
        "best_multiseed_optimism": aggregate.sort_values("nested_lower_tail_optimism_mean").iloc[0].to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame([asdict(prop_summary)]).to_string(index=False), flush=True)
    print(pd.DataFrame([asdict(budget)]).to_string(index=False), flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
