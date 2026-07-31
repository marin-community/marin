# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Strictly evaluate whether MDE-style features improve DSP predictions.

This script addresses two leakage caveats in the lightweight DSP+MDE probe:

1. DSP nonlinear parameters are refit inside each outer fold, rather than loaded
   from the full-data canonical fit.
2. Candidate-queryable variants do not use held-out checkpoint features at test
   time.  They first predict compressed MDE features from mixture/DSP features
   using training rows only, then use those predicted features for residual
   correction.

Observed-MDE rows answer an explanatory question: if a checkpoint already
exists and we evaluate MDE-style features on it, do those features explain DSP
residuals?  Queryable-MDE rows answer the mixture-search question: can MDE-style
latent features improve predictions for unseen mixtures without evaluating the
checkpoint first?
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_grug_v4_aggregate_canonical_dsp import (
    DEFAULT_METADATA_CSV,
    DEFAULT_NOISE_CSV,
    DEFAULT_RAW_CSV,
    DEFAULT_VARIANT,
    DatasetBundle,
    aggregate_targets,
    load_data,
    weights_to_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_mde_checkpoint_feature_regression_300m import (
    RIDGE_ALPHAS,
    TF_PCA_COMPONENTS,
    aligned_features,
    score_predictions,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_dsp_strict_stacking_300m_20260529"
N_SPLITS = dsp.N_SPLITS
CV_SEED = dsp.CV_SEED
INNER_SPLITS = 4
DSP_MAXITER = dsp.FIT_MAXITER
DSP_COARSE_TOP_K = dsp.START_TOP_K
DSP_BASIN_HOPPING_ITERS = 3


@dataclass(frozen=True)
class ResidualDesign:
    """One residual-correction design."""

    name: str
    use_phase: bool
    use_teacher_forced: bool
    use_mcq: bool
    queryable: bool


RESIDUAL_DESIGNS = (
    ResidualDesign("strict_dsp_oof", False, False, False, False),
    ResidualDesign("strict_observed_phase_residual", True, False, False, False),
    ResidualDesign("strict_observed_teacher_forced_pca_residual", False, True, False, False),
    ResidualDesign("strict_observed_mcq_aggregate_residual", False, False, True, False),
    ResidualDesign("strict_observed_teacher_forced_mcq_residual", False, True, True, False),
    ResidualDesign("strict_observed_phase_teacher_forced_mcq_residual", True, True, True, False),
    ResidualDesign("strict_queryable_teacher_forced_mcq_residual", False, True, True, True),
    ResidualDesign("strict_queryable_phase_teacher_forced_mcq_residual", True, True, True, True),
)


def load_aggregate_packet() -> tuple[pd.DataFrame, np.ndarray, dsp.PacketData, list[str]]:
    """Load collaborator aggregate target and DSP packet."""
    bundle = DatasetBundle(
        name="sent_raw_metric_matrix_300m_zip",
        raw_path=DEFAULT_RAW_CSV,
        noise_path=DEFAULT_NOISE_CSV,
        metadata_path=DEFAULT_METADATA_CSV,
        drop_incomplete_task_cols=False,
    )
    data = load_data(bundle)
    raw = data["raw"]
    domains = data["domains"]
    w0 = data["w0"]
    w1 = data["w1"]
    c0 = data["c0"]
    c1 = data["c1"]
    z = data["z"]
    noise_share = data["noise_share"]
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(domains, list)
    assert isinstance(w0, np.ndarray)
    assert isinstance(w1, np.ndarray)
    assert isinstance(c0, np.ndarray)
    assert isinstance(c1, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert isinstance(noise_share, np.ndarray)
    aggregate = aggregate_targets(z, noise_share)
    y_factor = np.asarray(aggregate["y_factor"], dtype=np.float64)
    packet = weights_to_packet(raw, -y_factor, domains, w0, w1, c0, c1)
    return raw, y_factor, packet, domains


def subset_packet(packet: dsp.PacketData, index: np.ndarray) -> dsp.PacketData:
    """Return a row-subset DSP packet with shared domain constants."""
    return dsp.PacketData(
        frame=packet.frame.iloc[index].reset_index(drop=True),
        name_col=packet.name_col,
        y=packet.y[index],
        w=packet.w[index],
        m=packet.m,
        c0=packet.c0,
        c1=packet.c1,
        domain_names=packet.domain_names,
    )


def scaled_part(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute and standardize one feature part."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train)
    test_imputed = imputer.transform(test)
    return scaler.fit_transform(train_imputed), scaler.transform(test_imputed)


def compressed_residual_features(
    *,
    design: ResidualDesign,
    train_index: np.ndarray,
    test_index: np.ndarray,
    phase: np.ndarray,
    teacher_forced: np.ndarray,
    mcq: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return direct phase features and compressed MDE features for one fold."""
    direct_train: list[np.ndarray] = []
    direct_test: list[np.ndarray] = []
    mde_train: list[np.ndarray] = []
    mde_test: list[np.ndarray] = []

    if design.use_phase:
        train, test = scaled_part(phase[train_index], phase[test_index])
        direct_train.append(train)
        direct_test.append(test)

    if design.use_teacher_forced:
        train_scaled, test_scaled = scaled_part(teacher_forced[train_index], teacher_forced[test_index])
        n_components = min(TF_PCA_COMPONENTS, train_scaled.shape[0] - 1, train_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=CV_SEED)
        mde_train.append(pca.fit_transform(train_scaled))
        mde_test.append(pca.transform(test_scaled))

    if design.use_mcq:
        train, test = scaled_part(mcq[train_index], mcq[test_index])
        mde_train.append(train)
        mde_test.append(test)

    direct_train_array = np.hstack(direct_train) if direct_train else None
    direct_test_array = np.hstack(direct_test) if direct_test else None
    mde_train_array = np.hstack(mde_train) if mde_train else None
    mde_test_array = np.hstack(mde_test) if mde_test else None
    return direct_train_array, direct_test_array, mde_train_array, mde_test_array


def mixture_feature_matrix(packet: dsp.PacketData, model: dsp.FittedDSPModel, index: np.ndarray) -> np.ndarray:
    """Return candidate-queryable mixture/DSP features for rows."""
    weights = packet.w[index]
    e0 = weights[:, 0, :] * packet.c0[None, :]
    e1 = weights[:, 1, :] * packet.c1[None, :]
    signal, penalty = dsp.features(weights, model.c0, model.c1, model.variant, model.params)
    return np.hstack(
        [
            weights.reshape(weights.shape[0], -1),
            np.log1p(e0),
            np.log1p(e1),
            signal,
            penalty,
        ]
    )


def predict_mde_components(
    *,
    train_inputs: np.ndarray,
    test_inputs: np.ndarray,
    train_components: np.ndarray,
    outer_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fit train components and predict test components from mixture inputs."""
    inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=outer_seed)
    train_pred = np.full_like(train_components, np.nan, dtype=np.float64)
    row_ids = np.arange(train_components.shape[0])
    for inner_train, inner_valid in inner.split(row_ids):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        model.fit(train_inputs[inner_train], train_components[inner_train])
        train_pred[inner_valid] = model.predict(train_inputs[inner_valid])
    if not np.isfinite(train_pred).all():
        raise ValueError("inner cross-fitted MDE component predictions contain non-finite values")
    final_model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    final_model.fit(train_inputs, train_components)
    return train_pred, final_model.predict(test_inputs)


def residual_inputs_for_design(
    *,
    design: ResidualDesign,
    train_index: np.ndarray,
    test_index: np.ndarray,
    phase: np.ndarray,
    teacher_forced: np.ndarray,
    mcq: np.ndarray,
    packet: dsp.PacketData,
    model: dsp.FittedDSPModel,
    fold_id: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build residual-model train/test inputs for one design."""
    direct_train, direct_test, mde_train, mde_test = compressed_residual_features(
        design=design,
        train_index=train_index,
        test_index=test_index,
        phase=phase,
        teacher_forced=teacher_forced,
        mcq=mcq,
    )
    parts_train: list[np.ndarray] = []
    parts_test: list[np.ndarray] = []
    if direct_train is not None and direct_test is not None:
        parts_train.append(direct_train)
        parts_test.append(direct_test)
    if mde_train is not None and mde_test is not None:
        if design.queryable:
            train_inputs = mixture_feature_matrix(packet, model, train_index)
            test_inputs = mixture_feature_matrix(packet, model, test_index)
            mde_train, mde_test = predict_mde_components(
                train_inputs=train_inputs,
                test_inputs=test_inputs,
                train_components=mde_train,
                outer_seed=CV_SEED + fold_id + 1,
            )
        parts_train.append(mde_train)
        parts_test.append(mde_test)
    if not parts_train:
        return None, None
    return np.hstack(parts_train), np.hstack(parts_test)


def fit_strict_oof(
    *,
    y_factor: np.ndarray,
    packet: dsp.PacketData,
    phase: np.ndarray,
    teacher_forced: np.ndarray,
    mcq: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return strict OOF predictions, metrics, and per-fold DSP diagnostics."""
    variant = dsp.VARIANTS[DEFAULT_VARIANT]
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    predictions = pd.DataFrame(
        {
            "run_name": packet.frame[packet.name_col].astype(str),
            "actual_y_factor": y_factor,
        }
    )
    for design in RESIDUAL_DESIGNS:
        predictions[design.name] = np.nan

    fold_rows: list[dict[str, object]] = []
    for fold_id, (train_index, test_index) in enumerate(cv.split(np.arange(len(y_factor)))):
        start = time.perf_counter()
        train_packet = subset_packet(packet, train_index)
        print(
            f"fold {fold_id + 1}/{N_SPLITS}: fitting DSP on {len(train_index)} rows, "
            f"testing {len(test_index)} rows",
            flush=True,
        )
        model, tuning = dsp.fit_variant(
            train_packet,
            variant,
            maxiter=DSP_MAXITER,
            coarse_top_k=DSP_COARSE_TOP_K,
            basin_hopping_iters=DSP_BASIN_HOPPING_ITERS,
        )
        base_train = -dsp.predict(model, packet.w[train_index])
        base_test = -dsp.predict(model, packet.w[test_index])
        predictions.loc[test_index, "strict_dsp_oof"] = base_test
        residual_target = y_factor[train_index] - base_train

        fold_rows.append(
            {
                "fold": fold_id,
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "elapsed_sec": float(time.perf_counter() - start),
                "dsp_train_rmse": float(np.sqrt(np.mean((y_factor[train_index] - base_train) ** 2))),
                "dsp_test_rmse": float(np.sqrt(np.mean((y_factor[test_index] - base_test) ** 2))),
                "best_profile_objective": float(tuning["objective"].min()),
                "active_benefit_coef_count": int(np.sum(model.benefit_coef > 1e-12)),
                "active_penalty_coef_count": int(np.sum(model.penalty_coef > 1e-12)),
            }
        )

        for design in RESIDUAL_DESIGNS:
            if design.name == "strict_dsp_oof":
                continue
            x_train, x_test = residual_inputs_for_design(
                design=design,
                train_index=train_index,
                test_index=test_index,
                phase=phase,
                teacher_forced=teacher_forced,
                mcq=mcq,
                packet=packet,
                model=model,
                fold_id=fold_id,
            )
            if x_train is None or x_test is None:
                raise ValueError(f"residual design has no inputs: {design.name}")
            residual_model = RidgeCV(alphas=RIDGE_ALPHAS)
            residual_model.fit(x_train, residual_target)
            predictions.loc[test_index, design.name] = base_test + residual_model.predict(x_test)

    metric_rows = []
    baseline_scores: dict[str, float] | None = None
    for design in RESIDUAL_DESIGNS:
        pred = predictions[design.name].to_numpy(dtype=np.float64)
        if not np.isfinite(pred).all():
            raise ValueError(f"non-finite predictions for {design.name}")
        scores = score_predictions(y_factor, pred)
        if design.name == "strict_dsp_oof":
            baseline_scores = scores
        if baseline_scores is None:
            raise ValueError("strict DSP baseline was not evaluated first")
        metric_rows.append(
            {
                "model": design.name,
                "queryable": bool(design.queryable),
                "uses_observed_checkpoint_features_at_test": bool(
                    (design.use_teacher_forced or design.use_mcq) and not design.queryable
                ),
                "n": int(len(y_factor)),
                **scores,
                "rmse_delta_vs_dsp": scores["rmse"] - baseline_scores["rmse"],
                "r2_delta_vs_dsp": scores["r2"] - baseline_scores["r2"],
                "pearson_r_delta_vs_dsp": scores["pearson_r"] - baseline_scores["pearson_r"],
                "spearman_r_delta_vs_dsp": scores["spearman_r"] - baseline_scores["spearman_r"],
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("spearman_r", ascending=False).reset_index(drop=True)
    return predictions, metrics, pd.DataFrame(fold_rows)


def write_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    """Write strict-stacking HTML diagnostics."""
    plot_metrics = metrics.melt(
        id_vars=["model", "queryable", "uses_observed_checkpoint_features_at_test"],
        value_vars=["spearman_r", "pearson_r", "r2"],
        var_name="score",
        value_name="value",
    )
    fig = px.bar(
        plot_metrics,
        x="model",
        y="value",
        color="score",
        pattern_shape="queryable",
        barmode="group",
        title="Strict DSP+MDE stacking: OOF scores",
    )
    fig.update_layout(width=1450, height=680, xaxis_tickangle=-25)
    fig.write_html(
        output_dir / "strict_stacking_metric_bars.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )

    comparison_models = ["strict_dsp_oof"]
    best_observed = metrics[metrics["uses_observed_checkpoint_features_at_test"]].head(1)
    best_queryable = metrics[metrics["queryable"]].head(1)
    if not best_observed.empty:
        comparison_models.append(str(best_observed.iloc[0]["model"]))
    if not best_queryable.empty:
        comparison_models.append(str(best_queryable.iloc[0]["model"]))
    comparison_models = list(dict.fromkeys(comparison_models))
    scatter = predictions[["run_name", "actual_y_factor", *comparison_models]].melt(
        id_vars=["run_name", "actual_y_factor"],
        value_vars=comparison_models,
        var_name="model",
        value_name="pred_y_factor",
    )
    fig = px.scatter(
        scatter,
        x="actual_y_factor",
        y="pred_y_factor",
        color="model",
        hover_name="run_name",
        title="Strict OOF predictions: DSP baseline, best observed-MDE, best queryable-MDE",
    )
    lo = float(min(scatter["actual_y_factor"].min(), scatter["pred_y_factor"].min()))
    hi = float(max(scatter["actual_y_factor"].max(), scatter["pred_y_factor"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="identity", line={"color": "black"}))
    fig.update_layout(width=980, height=760)
    fig.write_html(
        output_dir / "strict_stacking_actual_vs_pred.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )


def main() -> None:
    """Run the strict DSP+MDE stacking probe."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, y_factor, packet, domains = load_aggregate_packet()
    phase, teacher_forced, mcq, feature_domains = aligned_features(raw)
    if domains != feature_domains:
        raise ValueError("canonical DSP domains and feature domains differ")
    print(
        f"loaded rows={len(raw)} domains={len(domains)} "
        f"teacher_forced_features={teacher_forced.shape[1]} mcq_features={mcq.shape[1]}",
        flush=True,
    )
    predictions, metrics, folds = fit_strict_oof(
        y_factor=y_factor,
        packet=packet,
        phase=phase,
        teacher_forced=teacher_forced,
        mcq=mcq,
    )

    predictions.to_csv(OUTPUT_DIR / "strict_stacking_oof_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "strict_stacking_metrics.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "strict_stacking_fold_diagnostics.csv", index=False)
    write_plots(metrics, predictions, OUTPUT_DIR)

    dsp_row = metrics[metrics["model"].eq("strict_dsp_oof")].iloc[0]
    best_observed = metrics[metrics["uses_observed_checkpoint_features_at_test"]].head(1).iloc[0]
    best_queryable = metrics[metrics["queryable"]].head(1).iloc[0]
    summary = {
        "rows": int(len(raw)),
        "domains": int(len(domains)),
        "target": "aggregate/y_factor",
        "cv": {"kind": "KFold", "splits": N_SPLITS, "seed": CV_SEED},
        "inner_cv": {"kind": "KFold", "splits": INNER_SPLITS},
        "dsp_variant": DEFAULT_VARIANT,
        "dsp_maxiter": DSP_MAXITER,
        "dsp_coarse_top_k": DSP_COARSE_TOP_K,
        "dsp_basin_hopping_iters": DSP_BASIN_HOPPING_ITERS,
        "strict_dsp_oof_spearman": float(dsp_row["spearman_r"]),
        "strict_dsp_oof_r2": float(dsp_row["r2"]),
        "best_observed_model": str(best_observed["model"]),
        "best_observed_spearman": float(best_observed["spearman_r"]),
        "best_observed_spearman_delta_vs_dsp": float(best_observed["spearman_r_delta_vs_dsp"]),
        "best_observed_r2_delta_vs_dsp": float(best_observed["r2_delta_vs_dsp"]),
        "best_queryable_model": str(best_queryable["model"]),
        "best_queryable_spearman": float(best_queryable["spearman_r"]),
        "best_queryable_spearman_delta_vs_dsp": float(best_queryable["spearman_r_delta_vs_dsp"]),
        "best_queryable_r2_delta_vs_dsp": float(best_queryable["r2_delta_vs_dsp"]),
        "teacher_forced_features": int(teacher_forced.shape[1]),
        "mcq_features": int(mcq.shape[1]),
        "artifacts": {
            "metrics_csv": str(OUTPUT_DIR / "strict_stacking_metrics.csv"),
            "predictions_csv": str(OUTPUT_DIR / "strict_stacking_oof_predictions.csv"),
            "fold_diagnostics_csv": str(OUTPUT_DIR / "strict_stacking_fold_diagnostics.csv"),
            "metric_bars_html": str(OUTPUT_DIR / "strict_stacking_metric_bars.html"),
            "actual_vs_pred_html": str(OUTPUT_DIR / "strict_stacking_actual_vs_pred.html"),
        },
        "interpretation": (
            "Observed-MDE rows are explanatory because test-time checkpoint features are observed. "
            "Queryable-MDE rows are the relevant conservative proxy for mixture search because "
            "test-time MDE components are predicted from mixture/DSP features only."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# Strict DSP + MDE-Style Stacking Probe",
        "",
        "Target: collaborator aggregate `y_factor` on the 242-row 300M signal matrix.",
        "",
        "Every outer fold refits DSP nonlinear parameters using only training rows.",
        "",
        f"- Strict DSP OOF Spearman: `{summary['strict_dsp_oof_spearman']:.4f}`.",
        f"- Best observed-MDE stacker: `{summary['best_observed_model']}`.",
        f"- Best observed-MDE Spearman: `{summary['best_observed_spearman']:.4f}` "
        f"({summary['best_observed_spearman_delta_vs_dsp']:+.4f} vs strict DSP).",
        f"- Best queryable-MDE stacker: `{summary['best_queryable_model']}`.",
        f"- Best queryable-MDE Spearman: `{summary['best_queryable_spearman']:.4f}` "
        f"({summary['best_queryable_spearman_delta_vs_dsp']:+.4f} vs strict DSP).",
        "",
        "## Model Scores",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
