#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test tied or shrunk N-family heads for MCT-LRQ.

The free family-dependent N head overfit badly. This script tests two safer
ways to let mixture affect the model-size return:

1. Tied N head:

       A(w) = a0 + rho * B(w)

   where B(w) is the same nonnegative family head used for the D return.

2. Shrunk N head:

       A(w) = a0 + phase_family_shares(w) @ a_family

   with an explicit ridge penalty on the N-family deviation coefficients.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from experiments.domain_phase_mix.exploratory.two_phase_many.run_mct_mix_term_ablation_20260424 import (
    FALLBACK_MCT_CODE_DIR,
    LOCAL_MCT_CODE_DIR,
    MCT_BALANCED_EXPONENTS,
    MCT_PAIR_WEIGHT,
    evaluate_models,
    import_from_path,
    jsonify,
    make_pairs,
    md_table,
    parameter_counts,
    plot_drop_ratios,
    plot_metric_bars,
    plot_pred_actual,
    scale_terms,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MCT_DROP_EXPONENTS = (0.154791, 0.146425, 0.014295, 1.063376)
DONOR_CONSTANT_COUNT = 9


class AnchoredScaleLawBase:
    """Shared LRQ-anchor setup for constrained MCT scale-head variants."""

    def __init__(
        self,
        cbs,
        s2_mod,
        model_id: str,
        data,
        ff,
        exponents: tuple[float, float, float, float],
        pair_weight: float = MCT_PAIR_WEIGHT,
        ridge_anchor: float = 1e-4,
        ridge_scale: float = 1e-5,
    ):
        self.cbs = cbs
        self.s2_mod = s2_mod
        self.model_id = model_id
        self.data = data
        self.ff = ff
        self.exponents = exponents
        self.alpha, self.beta, self.gamma, self.delta = map(float, exponents)
        self.pair_weight = float(pair_weight)
        self.ridge_anchor = float(ridge_anchor)
        self.ridge_scale = float(ridge_scale)
        self.donor_constant_count = DONOR_CONSTANT_COUNT
        self.barrier_constant_count = 0
        self.anchor_kind = "lrq_scarcity"

    def offset(self, W: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(W).shape[0], dtype=float)

    def _fit_anchor(self, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = self.data
        X_anchor_all, anchor_names = self.ff.anchor_features(data.W, self.anchor_kind)
        anchor_rows = train_mask & (data.scale_names_by_row == "300m_6b") & np.isclose(data.mu, 1.0)
        coef_anchor, stats_anchor = self.s2_mod.ridge_fit(
            X_anchor_all[anchor_rows], data.y[anchor_rows], ridge=self.ridge_anchor
        )
        anchor_pred_all = self.s2_mod.ridge_predict_from_stats(X_anchor_all, coef_anchor, stats_anchor)
        self.anchor_coef = coef_anchor
        self.anchor_stats = stats_anchor
        self.anchor_names = anchor_names
        self.anchor_feature_count = X_anchor_all.shape[1] + 1
        return X_anchor_all, anchor_pred_all

    def _solve_scale(
        self,
        train_mask: np.ndarray,
        design_all: np.ndarray,
        y_resid: np.ndarray,
        extra_ridge: list[tuple[int, float]] | None = None,
    ) -> None:
        data = self.data
        rows = np.where(train_mask)[0]
        matrix = design_all[rows]
        target = y_resid[rows]
        pair_rows: list[np.ndarray] = []
        pair_y: list[float] = []
        for i, j in make_pairs(data, train_mask):
            weight = math.sqrt(self.pair_weight)
            pair_rows.append(weight * (design_all[i] - design_all[j]))
            pair_y.append(weight * (y_resid[i] - y_resid[j]))
        if pair_rows:
            matrix = np.vstack([matrix, np.asarray(pair_rows)])
            target = np.concatenate([target, np.asarray(pair_y)])
        ridge_rows: list[np.ndarray] = []
        ridge_target: list[float] = []
        if self.ridge_scale > 0:
            ridge_rows.extend(np.sqrt(self.ridge_scale) * np.eye(design_all.shape[1]))
            ridge_target.extend([0.0] * design_all.shape[1])
        if extra_ridge:
            for column, strength in extra_ridge:
                row = np.zeros(design_all.shape[1], dtype=float)
                row[int(column)] = math.sqrt(float(strength))
                ridge_rows.append(row)
                ridge_target.append(0.0)
        if ridge_rows:
            matrix = np.vstack([matrix, np.asarray(ridge_rows)])
            target = np.concatenate([target, np.asarray(ridge_target)])
        result = lsq_linear(matrix, target, bounds=(0.0, np.inf), max_iter=2000, lsmr_tol="auto", verbose=0)
        self.scale_coef = np.clip(np.asarray(result.x, dtype=float), 0.0, np.inf)
        self.scale_fit_success = bool(result.success)
        self.scale_fit_cost = float(result.cost)
        self.pair_count = len(pair_rows)

    def _anchor_prediction(self, W: np.ndarray) -> np.ndarray:
        X, _ = self.ff.anchor_features(W, self.anchor_kind)
        return self.s2_mod.ridge_predict_from_stats(X, self.anchor_coef, self.anchor_stats)

    def predict_all(self) -> np.ndarray:
        return self.predict_custom(self.data.W, self.data.N, self.data.D)


class TiedNToDFamilyHeadLaw(AnchoredScaleLawBase):
    """MCT-LRQ with A(w) = a0 + rho * B(w)."""

    def __init__(self, *args, rho: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = float(rho)
        self.tied_constant_count = 1

    def fit(self, train_mask: np.ndarray):
        data = self.data
        self.train_mask = np.asarray(train_mask, dtype=bool).copy()
        _anchor_features, anchor_pred_all = self._fit_anchor(self.train_mask)
        family_features = self.ff.scale_head_features(data.W, "family")
        uN, uD, uND = scale_terms(data, self.exponents, data.N, data.D)
        design_all = np.column_stack(
            [
                uN[:, None],
                family_features * (uD + self.rho * uN)[:, None],
                uND[:, None],
            ]
        )
        self.family_feature_count = family_features.shape[1]
        self.scale_param_count = design_all.shape[1]
        self._solve_scale(self.train_mask, design_all, data.y - anchor_pred_all)
        self.fitted_param_count = self.anchor_feature_count + self.scale_param_count + 4 + self.tied_constant_count
        self.total_constant_count = self.fitted_param_count + self.donor_constant_count
        return self

    def predict_custom(self, W: np.ndarray, N: np.ndarray | float, D: np.ndarray | float) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        count = W.shape[0]
        N_arr = np.full(count, float(N)) if np.ndim(N) == 0 else np.asarray(N, dtype=float)
        D_arr = np.full(count, float(D)) if np.ndim(D) == 0 else np.asarray(D, dtype=float)
        family_features = self.ff.scale_head_features(W, "family")
        uN, uD, uND = scale_terms(self.data, self.exponents, N_arr, D_arr)
        design = np.column_stack(
            [
                uN[:, None],
                family_features * (uD + self.rho * uN)[:, None],
                uND[:, None],
            ]
        )
        return self._anchor_prediction(W) + design @ self.scale_coef

    def head_values(self, W: np.ndarray) -> pd.DataFrame:
        family_features = self.ff.scale_head_features(W, "family")
        a0 = float(self.scale_coef[0])
        b = self.scale_coef[1 : 1 + self.family_feature_count]
        c = float(self.scale_coef[-1])
        b_values = family_features @ b
        return pd.DataFrame(
            {
                "A_N": a0 + self.rho * b_values,
                "B_D": b_values,
                "C_ND": np.full(W.shape[0], c),
            }
        )

    def artifact(self) -> dict:
        return {
            "model_id": self.model_id,
            "formula": "E_LRQ(w) + (a0 + rho*B(w))*uN + B(w)*uD + C*uND",
            "rho": self.rho,
            "N_head": "tied_to_D_family_head",
            "exponents": {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "delta": self.delta},
            "scale_coef": self.scale_coef,
            "scale_param_count": self.scale_param_count,
            "fitted_param_count_counting_exponents": self.fitted_param_count,
            "total_constant_count": self.total_constant_count,
            "scale_fit_success": self.scale_fit_success,
            "scale_fit_cost": self.scale_fit_cost,
        }


class ShrunkNFamilyDeviationLaw(AnchoredScaleLawBase):
    """MCT-LRQ with a penalized family-dependent N deviation head."""

    def __init__(self, *args, shrink_strength: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.shrink_strength = float(shrink_strength)
        self.shrink_constant_count = 1

    def fit(self, train_mask: np.ndarray):
        data = self.data
        self.train_mask = np.asarray(train_mask, dtype=bool).copy()
        _anchor_features, anchor_pred_all = self._fit_anchor(self.train_mask)
        family_features = self.ff.scale_head_features(data.W, "family")
        family_shares = family_features[:, 1:]
        uN, uD, uND = scale_terms(data, self.exponents, data.N, data.D)
        design_all = np.column_stack(
            [
                uN[:, None],
                family_shares * uN[:, None],
                family_features * uD[:, None],
                uND[:, None],
            ]
        )
        self.n_family_count = family_shares.shape[1]
        self.d_family_count = family_features.shape[1]
        self.scale_param_count = design_all.shape[1]
        extra_ridge = [(1 + idx, self.shrink_strength) for idx in range(self.n_family_count)]
        self._solve_scale(self.train_mask, design_all, data.y - anchor_pred_all, extra_ridge=extra_ridge)
        self.fitted_param_count = self.anchor_feature_count + self.scale_param_count + 4 + self.shrink_constant_count
        self.total_constant_count = self.fitted_param_count + self.donor_constant_count
        return self

    def predict_custom(self, W: np.ndarray, N: np.ndarray | float, D: np.ndarray | float) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        count = W.shape[0]
        N_arr = np.full(count, float(N)) if np.ndim(N) == 0 else np.asarray(N, dtype=float)
        D_arr = np.full(count, float(D)) if np.ndim(D) == 0 else np.asarray(D, dtype=float)
        family_features = self.ff.scale_head_features(W, "family")
        family_shares = family_features[:, 1:]
        uN, uD, uND = scale_terms(self.data, self.exponents, N_arr, D_arr)
        design = np.column_stack(
            [
                uN[:, None],
                family_shares * uN[:, None],
                family_features * uD[:, None],
                uND[:, None],
            ]
        )
        return self._anchor_prediction(W) + design @ self.scale_coef

    def head_values(self, W: np.ndarray) -> pd.DataFrame:
        family_features = self.ff.scale_head_features(W, "family")
        family_shares = family_features[:, 1:]
        a0 = float(self.scale_coef[0])
        a_dev = self.scale_coef[1 : 1 + self.n_family_count]
        b_start = 1 + self.n_family_count
        b = self.scale_coef[b_start : b_start + self.d_family_count]
        c = float(self.scale_coef[-1])
        return pd.DataFrame(
            {
                "A_N": a0 + family_shares @ a_dev,
                "B_D": family_features @ b,
                "C_ND": np.full(W.shape[0], c),
            }
        )

    def artifact(self) -> dict:
        return {
            "model_id": self.model_id,
            "formula": "E_LRQ(w) + (a0 + family_shares@a_dev)*uN + B(w)*uD + C*uND",
            "shrink_strength": self.shrink_strength,
            "N_head": "ridge_shrunk_family_deviation",
            "exponents": {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "delta": self.delta},
            "scale_coef": self.scale_coef,
            "scale_param_count": self.scale_param_count,
            "fitted_param_count_counting_exponents": self.fitted_param_count,
            "total_constant_count": self.total_constant_count,
            "scale_fit_success": self.scale_fit_success,
            "scale_fit_cost": self.scale_fit_cost,
        }


def make_mct_baseline(cbs, s2_mod, data, ff, model_id: str, train_mask: np.ndarray, exponents):
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
        head_A="constant",
        head_B="family",
        head_C="constant",
    ).fit(train_mask)


def _fit_variants(cbs, s2_mod, data, ff, train_mask: np.ndarray) -> dict[str, object]:
    variants: dict[str, object] = {
        "mct_lrq69_drop_Dfam": make_mct_baseline(
            cbs, s2_mod, data, ff, "mct_lrq69_drop_Dfam", train_mask, MCT_DROP_EXPONENTS
        ),
        "mct_lrq69_balanced_Dfam": make_mct_baseline(
            cbs, s2_mod, data, ff, "mct_lrq69_balanced_Dfam", train_mask, MCT_BALANCED_EXPONENTS
        ),
    }
    for rho in [0.025, 0.05, 0.10, 0.25, 0.50]:
        suffix = f"rho{rho:g}".replace(".", "p")
        variants[f"mct_lrq70_drop_Ntied_{suffix}"] = TiedNToDFamilyHeadLaw(
            cbs,
            s2_mod,
            f"mct_lrq70_drop_Ntied_{suffix}",
            data,
            ff,
            MCT_DROP_EXPONENTS,
            rho=rho,
        ).fit(train_mask)
    for shrink in [0.1, 1.0, 10.0, 100.0]:
        suffix = f"lam{shrink:g}".replace(".", "p")
        variants[f"mct_lrq76_drop_Nshrink_{suffix}"] = ShrunkNFamilyDeviationLaw(
            cbs,
            s2_mod,
            f"mct_lrq76_drop_Nshrink_{suffix}",
            data,
            ff,
            MCT_DROP_EXPONENTS,
            shrink_strength=shrink,
        ).fit(train_mask)
    return variants


def _compact_summary(metrics: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    seed7 = metrics[(metrics["fit_protocol"] == "seed7") & (metrics["split"] == "seed7_holdout")][
        ["model", "rmse"]
    ].rename(columns={"rmse": "seed7_holdout_rmse"})
    fixed = metrics[(metrics["fit_protocol"] == "seed7") & (metrics["split"] == "fixed340_holdout")][
        ["model", "rmse", "slope_pred_on_actual", "std_ratio"]
    ].rename(
        columns={
            "rmse": "fixed340_rmse",
            "slope_pred_on_actual": "fixed340_slope",
            "std_ratio": "fixed340_std_ratio",
        }
    )
    all900 = metrics[(metrics["fit_protocol"] == "leave900out") & (metrics["split"] == "all900_leave_scale_out")][
        ["model", "rmse"]
    ].rename(columns={"rmse": "all900_leaveout_rmse"})
    return (
        seed7.merge(fixed, on="model", how="outer")
        .merge(all900, on="model", how="outer")
        .merge(params[["model", "total_constant_count", "scale_param_count"]], on="model", how="left")
        .sort_values(["fixed340_rmse", "all900_leaveout_rmse"])
    )


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
    drop_cols = [
        "model",
        "drop_pair",
        "n",
        "actual_drop_mean",
        "pred_drop_mean",
        "drop_error_mean",
        "drop_ratio_mean",
        "drop_ratio_median",
        "drop_rmse",
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
    core = metrics[
        metrics["split"].isin(["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"])
    ].copy()
    compact = _compact_summary(metrics, params)
    lines = [
        "# MCT-LRQ Tied/Shrunk N-Head Ablation",
        "",
        "Date: 2026-04-26",
        "",
        "## Question",
        "",
        (
            "Can we safely add mixture-dependence to the model-size return term if the N-family head is tied "
            "to the D-family head or heavily shrunk, instead of being a free family head?"
        ),
        "",
        "## Result",
        "",
        (
            "No tested tied or shrunk N-family variant improves over the canonical global-N/family-D head. "
            "Small tied coefficients are mostly neutral; larger tied coefficients and shrunk free deviations "
            "degrade fixed-340M shape and 900M leaveout. The current evidence still favors keeping the N head "
            "global and the D head family-dependent."
        ),
        "",
        "## Compact Summary",
        "",
        md_table(compact, floatfmt=".6f"),
        "",
        "## Predictive Metrics",
        "",
        md_table(core.sort_values(["model", "fit_protocol", "split"]), cols=metric_cols),
        "",
        "## Fixed-340M Same-Mixture Drops",
        "",
        md_table(drop_summary.sort_values(["model", "drop_pair"]), cols=drop_cols),
        "",
        "## Raw 340M/900M Optima",
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
        "- `csv/fixed340_drop_summary.csv`: same-mixture target-budget drop metrics.",
        "- `csv/optimum_diagnostics.csv`: raw/hull/trustblend optimum probes.",
        "- `plots/rmse_tied_n_head_ablation.png`: RMSE comparison.",
        "- `plots/pred_actual_tied_n_head_ablation.png`: fixed-340M and 900M predicted-vs-actual panels.",
        "- `plots/drop_ratios_tied_n_head_ablation.png`: fixed-340M drop-ratio comparison.",
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
        default=SCRIPT_DIR / "reference_outputs" / "mct_tied_n_head_ablation_20260426",
    )
    args = parser.parse_args()

    code_dir = args.mct_code_dir.resolve()
    if not (code_dir / "cbs_lrq_base.py").exists():
        raise FileNotFoundError(f"Missing MCT helper code at {code_dir}")

    outdir = args.outdir.resolve()
    for subdir in ["csv", "plots", "models"]:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)

    cbs = import_from_path("mct_tied_n_head_cbs_lrq_base", code_dir / "cbs_lrq_base.py")
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

    shutil.copy2(Path(__file__), outdir / "models" / Path(__file__).name)
    plot_metric_bars(metrics, outdir / "plots" / "rmse_tied_n_head_ablation.png")
    plot_pred_actual(predictions, outdir / "plots" / "pred_actual_tied_n_head_ablation.png")
    plot_drop_ratios(drop_summary, outdir / "plots" / "drop_ratios_tied_n_head_ablation.png")
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
