# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "joblib", "tabulate"]
# ///
"""Independently check concrete CC consultation claims against saved evidence."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)

ROOT = Path(__file__).parent / "reference_outputs" / "mariner_local_validation_20260917"
OUTPUT = ROOT / "convex_round2_consult" / "independent_audit"
KEYS = ["panel", "target", "component", "repeat", "fold"]


def basis_audit() -> pd.DataFrame:
    diagnostic = pd.read_csv(ROOT / "convex/report_certify/basis_and_convexity_diagnostics.csv")
    records = []
    for row in diagnostic.itertuples():
        task = bench.FitTask(
            row.model_id, row.panel, row.target, row.component_index, row.component, row.repeat, row.fold
        )
        with np.load(bench.shard_path(ROOT / "convex", task), allow_pickle=False) as saved:
            rms = saved["training_column_rms"]
            names = saved["design_names"].astype(str)
        assert len(rms) == 78 and len(names) == 78
        records.append(
            {
                **{key: getattr(row, key) for key in KEYS},
                "model": row.model_id,
                "zero_benefit": int((rms[:39] == 0).sum()),
                "zero_harm": int((rms[39:] == 0).sum()),
            }
        )
    frame = pd.DataFrame(records)
    baseline_path = ROOT.parent / "mariner_harm_screen_20260917/fixed_basis/report_certify/component_fold_metrics.csv"
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline.model.eq("hs17_mariner")][[*KEYS, "rmse", "spearman"]]
    current = pd.read_csv(ROOT / "convex/report_certify/component_fold_metrics.csv")
    joined = frame.merge(current[[*KEYS, "model", "rmse", "spearman"]], on=[*KEYS, "model"], validate="one_to_one")
    joined = joined.merge(baseline, on=KEYS, suffixes=("", "_mariner"), validate="many_to_one")
    assert len(joined) == 1740
    joined["delta_rmse"] = joined.rmse - joined.rmse_mariner
    joined["delta_spearman"] = joined.spearman - joined.spearman_mariner
    joined["harm_zero_bin"] = pd.cut(joined.zero_harm, [-1, 2, 29, 38, 39], labels=["0-2", "3-29", "30-38", "39"])
    joined.to_csv(OUTPUT / "basis_matched_metrics.csv", index=False)
    grouped = joined.groupby(["model", "panel", "target", "harm_zero_bin"], observed=True)
    grouped.agg(
        n=("delta_rmse", "size"), delta_rmse=("delta_rmse", "mean"), delta_spearman=("delta_spearman", "mean")
    ).reset_index().to_csv(OUTPUT / "basis_strata.csv", index=False)
    associations = []
    for (model, panel, target), group in joined.groupby(["model", "panel", "target"]):
        associations.append(
            {
                "model": model,
                "panel": panel,
                "target": target,
                "n": len(group),
                "zero_vs_delta_rmse_rho": stats.spearmanr(group.zero_harm, group.delta_rmse).statistic,
                "zero_vs_delta_rho_rho": stats.spearmanr(group.zero_harm, group.delta_spearman).statistic,
            }
        )
    pd.DataFrame(associations).to_csv(OUTPUT / "basis_associations.csv", index=False)
    return joined


def floor_and_support_audit() -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(ROOT / "banks/fits.csv"), pd.read_csv(ROOT / "banks/convex/fits.csv")])
    frame = frame[frame.model.isin(["hs17_mariner", "cv17_smooth_d1", "cv17_huber_d8"])]
    records = []
    for row in frame.itertuples():
        shape, diagnostic = json.loads(row.shape), json.loads(row.diagnostics)
        delta = diagnostic["inner_cv_rmse"] - diagnostic["selection_inner_cv_rmse"]
        record = {
            "model": row.model,
            "panel": row.panel,
            "target": row.target,
            "component": row.component,
            "threshold": shape["threshold"],
            "kappa_is_1_5": np.isclose(diagnostic["kappa"], 1.5, atol=1e-12, rtol=0),
            "inner_cv_delta": delta,
            "inner_cv_relative_delta": delta / diagnostic["selection_inner_cv_rmse"],
        }
        if row.model == "hs17_mariner":
            fitted = joblib.load(row.fit_path)
            features = bench.load_panel(row.panel).features
            active = fitted.head.coefficients[features.buckets :] > 0
            inflection = (np.e - 1) * np.exp(shape["threshold"]) - 1
            observed_max = float(features.exposures[:, active].max()) if active.any() else 0.0
            reachable_max = float(features.inventory[active].max()) if active.any() else 0.0
            record.update(
                inflection=inflection,
                active_harm_columns=int(active.sum()),
                observed_max_active=observed_max,
                reachable_max_active=reachable_max,
                harm_convex_on_observed_box=observed_max <= inflection,
                harm_convex_on_simplex=reachable_max <= inflection,
            )
        records.append(record)
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT / "floor_and_support.csv", index=False)
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    basis = basis_audit()
    floors = floor_and_support_audit()
    counts = basis.groupby("model").agg(
        fits=("zero_harm", "size"),
        benefit_zeros=("zero_benefit", "sum"),
        harm_ge_30=("zero_harm", lambda x: (x >= 30).sum()),
        harm_all_zero=("zero_harm", lambda x: (x == 39).sum()),
    )
    floor_summary = floors.groupby("model").agg(
        fits=("threshold", "size"),
        threshold_1=("threshold", lambda x: (x == 1).sum()),
        threshold_6=("threshold", lambda x: (x == 6).sum()),
        kappa_is_1_5=("kappa_is_1_5", "sum"),
        mean_cv_delta=("inner_cv_delta", "mean"),
        mean_relative_cv_delta=("inner_cv_relative_delta", "mean"),
    )
    support = (
        floors[floors.model.eq("hs17_mariner")]
        .groupby("panel")
        .agg(
            fits=("threshold", "size"),
            observed_box=("harm_convex_on_observed_box", lambda values: int(values.eq(True).sum())),
            full_simplex=("harm_convex_on_simplex", lambda values: int(values.eq(True).sum())),
        )
    )
    text = "# Independent consultation audit\n\n" + counts.to_markdown() + "\n\n" + floor_summary.to_markdown()
    text += "\n\nMARINER harm-convex sufficient conditions for each fitted head:\n\n" + support.to_markdown()
    text += (
        "\n\nAll zero columns in these 1,740 fits are in the harm block. The saved count claim is correct, "
        "but zero columns are downstream of selected shape. Stratified associations do not identify causation. "
        "A coefficient of exactly zero is inactive; no amplitude threshold is imposed. Convex harm over observed "
        "coordinate bounds is only a sufficient condition for component convexity there, not a necessity test "
        "or a whole-simplex guarantee. A failed harm test does not prove the complete raw-loss objective is "
        "nonconvex: benefit curvature and the exponential link can compensate. The kappa=1.5 count matches "
        "the flat-profile fallback value but is not an independently logged branch indicator.\n\n"
        "The simple dead-column explanation predicts increasing RMSE deficit and decreasing correlation "
        "as zeros increase within each panel/objective. See basis_associations.csv and basis_strata.csv; "
        "do not pool different panels or treat folds/components as independent replications.\n"
    )
    (OUTPUT / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
