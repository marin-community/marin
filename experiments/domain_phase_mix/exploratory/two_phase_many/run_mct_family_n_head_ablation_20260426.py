#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test family-dependent N and D heads in the barrier-free MCT-LRQ law.

The current compact structural MCT-LRQ law uses an LRQ mixture anchor, a global
N head, a family-dependent D head, and a global N-D cross head:

    E_LRQ(w) + A * u_N + B_fam(w) * u_D + C * u_ND.

This script tests the natural symmetric variant:

    E_LRQ(w) + A_fam(w) * u_N + B_fam(w) * u_D + C * u_ND.

All head features are nonnegative and all coefficients are fit with NNLS, so
fixed-mixture predictions remain monotone decreasing in N and D.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.run_mct_mix_term_ablation_20260424 import (
    FALLBACK_MCT_CODE_DIR,
    LOCAL_MCT_CODE_DIR,
    MCT_BALANCED_EXPONENTS,
    MCT_PAIR_WEIGHT,
    evaluate_models,
    import_from_path,
    jsonify,
    md_table,
    parameter_counts,
    plot_drop_ratios,
    plot_metric_bars,
    plot_pred_actual,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MCT_DROP_EXPONENTS = (0.154791, 0.146425, 0.014295, 1.063376)


def make_mct_law(
    cbs,
    s2_mod,
    data,
    ff,
    model_id: str,
    train_mask: np.ndarray,
    exponents: tuple[float, float, float, float],
    head_a: str,
    head_b: str = "family",
):
    return cbs.CompatibilityBarrierPowerLaw(
        s2_mod,
        model_id=model_id,
        data=data,
        ff=ff,
        strength=0.0,
        anchor_kind="lrq_scarcity",
        exponents=exponents,
        pair_weight=MCT_PAIR_WEIGHT,
        ridge_anchor=1e-4,
        ridge_scale=1e-5,
        head_A=head_a,
        head_B=head_b,
        head_C="constant",
    ).fit(train_mask)


def _fit_variants(cbs, s2_mod, data, ff, train_mask: np.ndarray) -> dict[str, object]:
    specs = [
        ("mct_lrq69_drop_Dfam", MCT_DROP_EXPONENTS, "constant"),
        ("mct_lrq75_drop_Nfam_Dfam", MCT_DROP_EXPONENTS, "family"),
        ("mct_lrq69_balanced_Dfam", MCT_BALANCED_EXPONENTS, "constant"),
        ("mct_lrq75_balanced_Nfam_Dfam", MCT_BALANCED_EXPONENTS, "family"),
    ]
    return {
        model_id: make_mct_law(
            cbs,
            s2_mod,
            data,
            ff,
            model_id,
            train_mask,
            exponents,
            head_a=head_a,
        )
        for model_id, exponents, head_a in specs
    }


def _core_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[
        metrics["split"].isin(["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"])
    ].copy()


def write_report(
    outdir: Path,
    metrics: pd.DataFrame,
    drop_summary: pd.DataFrame,
    opt: pd.DataFrame,
    params: pd.DataFrame,
    monotone: pd.DataFrame,
) -> None:
    metric_cols = [
        "model",
        "fit_protocol",
        "split",
        "n",
        "rmse",
        "spearman",
        "bias_pred_minus_actual",
        "slope_pred_on_actual",
        "std_ratio",
        "low_tail_rmse",
    ]
    opt_cols = [
        "model",
        "target_scale",
        "opt_kind",
        "predicted_bpb",
        "hard_corner_flag",
        "phase1_tech_collapse_flag",
        "any_family_collapse_flag",
        "nearest_observed_phase_mean_tv",
        "p0_broad_text_share",
        "p0_tech_code_share",
        "p0_reasoning_share",
        "p1_broad_text_share",
        "p1_tech_code_share",
        "p1_reasoning_share",
    ]
    core = _core_metrics(metrics).sort_values(["model", "fit_protocol", "split"])
    fixed340 = metrics[(metrics["fit_protocol"] == "seed7") & (metrics["split"] == "fixed340_holdout")][
        ["model", "rmse", "slope_pred_on_actual", "std_ratio"]
    ].rename(
        columns={
            "rmse": "fixed340_rmse",
            "slope_pred_on_actual": "fixed340_slope",
            "std_ratio": "fixed340_std_ratio",
        }
    )
    seed7 = metrics[(metrics["fit_protocol"] == "seed7") & (metrics["split"] == "seed7_holdout")][
        ["model", "rmse"]
    ].rename(columns={"rmse": "seed7_holdout_rmse"})
    all900 = metrics[(metrics["fit_protocol"] == "leave900out") & (metrics["split"] == "all900_leave_scale_out")][
        ["model", "rmse"]
    ].rename(columns={"rmse": "all900_leaveout_rmse"})
    compact = seed7.merge(fixed340, on="model", how="outer").merge(all900, on="model", how="outer")
    compact = compact.merge(params[["model", "total_constant_count", "scale_param_count"]], on="model", how="left")

    lines = [
        "# MCT-LRQ Family-N Head Ablation",
        "",
        "Date: 2026-04-26",
        "",
        "## Question",
        "",
        (
            "Does the compact structural MCT-LRQ law improve if both the N and D scale heads are "
            "family-dependent? The previous canonical barrier-free law used a global N head and a "
            "family-dependent D head."
        ),
        "",
        "## Variants",
        "",
        "- `mct_lrq69_drop_Dfam`: drop-tuned exponents, global `A`, family `B`, global `C`.",
        "- `mct_lrq75_drop_Nfam_Dfam`: drop-tuned exponents, family `A`, family `B`, global `C`.",
        "- `mct_lrq69_balanced_Dfam`: balanced exponents, global `A`, family `B`, global `C`.",
        "- `mct_lrq75_balanced_Nfam_Dfam`: balanced exponents, family `A`, family `B`, global `C`.",
        "",
        "The `Nfam` variants add six parameters: `A(w)` changes from one nonnegative scalar to a "
        "nonnegative linear head over the intercept plus the six phase-family shares.",
        "",
        "## Compact Summary",
        "",
        md_table(compact.sort_values("fixed340_rmse"), floatfmt=".6f"),
        "",
        "## Predictive Metrics",
        "",
        md_table(core, cols=metric_cols),
        "",
        "## Fixed-340M Same-Mixture Drops",
        "",
        md_table(
            drop_summary.sort_values(["model", "drop_pair"]),
            cols=[
                "model",
                "drop_pair",
                "n",
                "actual_drop_mean",
                "pred_drop_mean",
                "drop_error_mean",
                "drop_ratio_mean",
                "drop_ratio_median",
                "drop_rmse",
            ],
        ),
        "",
        "## Raw And Constrained Optima",
        "",
        md_table(
            opt[opt["target_scale"].isin(["340M/10.4B", "900M/24B"])].sort_values(["model", "target_scale", "opt_kind"]),
            cols=opt_cols,
        ),
        "",
        "## Monotonicity Grid",
        "",
        md_table(monotone),
        "",
        "## Artifact Map",
        "",
        "- `csv/metric_summary.csv`: split metrics.",
        "- `csv/row_predictions.csv`: row-level predictions.",
        "- `csv/fixed340_drop_summary.csv`: same-mixture target-budget drop metrics.",
        "- `csv/optimum_diagnostics.csv`: raw/hull/trustblend optimum probes.",
        "- `plots/rmse_family_n_head_ablation.png`: RMSE comparison.",
        "- `plots/pred_actual_family_n_head_ablation.png`: fixed-340M and 900M predicted-vs-actual panels.",
        "- `plots/drop_ratios_family_n_head_ablation.png`: fixed-340M drop-ratio comparison.",
    ]
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-root",
        type=Path,
        default=SCRIPT_DIR / "chatgpt_pro_hybrid_data_mixing_packet_v31",
    )
    parser.add_argument(
        "--mct-code-dir",
        type=Path,
        default=LOCAL_MCT_CODE_DIR if (LOCAL_MCT_CODE_DIR / "cbs_lrq_base.py").exists() else FALLBACK_MCT_CODE_DIR,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "reference_outputs" / "mct_family_n_head_ablation_20260426",
    )
    args = parser.parse_args()

    code_dir = args.mct_code_dir.resolve()
    if not (code_dir / "cbs_lrq_base.py").exists():
        raise FileNotFoundError(f"Missing MCT helper code at {code_dir}")

    outdir = args.outdir.resolve()
    for subdir in ["csv", "plots", "models", "code"]:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)

    cbs = import_from_path("mct_family_n_head_cbs_lrq_base", code_dir / "cbs_lrq_base.py")
    packet_root = cbs.packet_root_from_arg(args.packet_root)
    s2_mod = cbs.import_s2(packet_root)
    data = s2_mod.load_packet(packet_root)
    base_ff = s2_mod.FeatureFactory(data)
    ff = cbs.LRQFeatureFactory(base_ff)

    seed_models = _fit_variants(cbs, s2_mod, data, ff, data.seed7_train)
    leave_models = _fit_variants(cbs, s2_mod, data, ff, data.all900_train)

    metrics, predictions = evaluate_models(cbs, data, seed_models, leave_models)
    drop_detail, drop_summary, beta_detail = cbs.fixed340_drop_tables(data, seed_models)
    opt = cbs.optimum_diagnostics(seed_models, data, ff)
    monotone = s2_mod.monotonicity_grid(seed_models, data, ff)
    params = parameter_counts(seed_models)

    metrics.to_csv(outdir / "csv" / "metric_summary.csv", index=False)
    predictions.to_csv(outdir / "csv" / "row_predictions.csv", index=False)
    drop_detail.to_csv(outdir / "csv" / "fixed340_drop_pairs.csv", index=False)
    drop_summary.to_csv(outdir / "csv" / "fixed340_drop_summary.csv", index=False)
    beta_detail.to_csv(outdir / "csv" / "fixed340_beta_triples.csv", index=False)
    opt.to_csv(outdir / "csv" / "optimum_diagnostics.csv", index=False)
    monotone.to_csv(outdir / "csv" / "monotonicity_grid.csv", index=False)
    params.to_csv(outdir / "csv" / "parameter_counts.csv", index=False)

    for name, model in seed_models.items():
        artifact = model.artifact() if hasattr(model, "artifact") else {"model_id": name}
        (outdir / "models" / f"{name}_seed7.json").write_text(json.dumps(jsonify(artifact), indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__), outdir / "code" / Path(__file__).name)
    for helper in ["cbs_lrq_base.py", "run_mct_lrq_law.py"]:
        helper_path = code_dir / helper
        if helper_path.exists():
            shutil.copy2(helper_path, outdir / "code" / helper)

    plot_metric_bars(metrics, outdir / "plots" / "rmse_family_n_head_ablation.png")
    plot_pred_actual(predictions, outdir / "plots" / "pred_actual_family_n_head_ablation.png")
    plot_drop_ratios(drop_summary, outdir / "plots" / "drop_ratios_family_n_head_ablation.png")
    write_report(outdir, metrics, drop_summary, opt, params, monotone)

    manifest = {
        "packet_root": str(packet_root),
        "mct_code_dir": str(code_dir),
        "outdir": str(outdir),
        "variants": list(seed_models),
    }
    (outdir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
