# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Test whether MDE-style checkpoint features add residual signal over DSP.

This is an observed-checkpoint probe over the completed 300M signal swarm.  It
uses the saved canonical DSP nonlinear parameters and evaluates fold-local DSP
linear heads.  MDE-style features are then fit only to the DSP residuals inside
the same held-out folds.

The result tests whether checkpoint-behavior features explain observed DSP
residuals; it does not yet make those features queryable for arbitrary unseen
mixtures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_grug_v4_aggregate_canonical_dsp import (
    DEFAULT_METADATA_CSV,
    DEFAULT_NOISE_CSV,
    DEFAULT_OUTPUT_DIR as CANONICAL_DSP_OUTPUT_DIR,
    DEFAULT_RAW_CSV,
    DatasetBundle,
    aggregate_targets,
    load_data,
    weights_to_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_mde_checkpoint_feature_regression_300m import (
    RIDGE_ALPHAS,
    FeatureBlock,
    aligned_features,
    score_predictions,
    transform_block,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_dsp_stacked_regression_300m_20260529"
STACK_N_SPLITS = dsp.N_SPLITS
STACK_CV_SEED = dsp.CV_SEED


@dataclass(frozen=True)
class ResidualBlock:
    """One residual model to fit on top of fold-local DSP predictions."""

    name: str
    feature_block: FeatureBlock | None


RESIDUAL_BLOCKS = (
    ResidualBlock("dsp_oof", None),
    ResidualBlock(
        "dsp_plus_phase_residual",
        FeatureBlock("phase_residual", use_phase=True, use_teacher_forced=False, use_mcq=False),
    ),
    ResidualBlock(
        "dsp_plus_teacher_forced_pca_residual",
        FeatureBlock("teacher_forced_residual", use_phase=False, use_teacher_forced=True, use_mcq=False),
    ),
    ResidualBlock(
        "dsp_plus_mcq_aggregate_residual",
        FeatureBlock("mcq_residual", use_phase=False, use_teacher_forced=False, use_mcq=True),
    ),
    ResidualBlock(
        "dsp_plus_teacher_forced_mcq_residual",
        FeatureBlock("teacher_forced_mcq_residual", use_phase=False, use_teacher_forced=True, use_mcq=True),
    ),
    ResidualBlock(
        "dsp_plus_phase_teacher_forced_mcq_residual",
        FeatureBlock("phase_teacher_forced_mcq_residual", use_phase=True, use_teacher_forced=True, use_mcq=True),
    ),
)


def load_canonical_model() -> dsp.FittedDSPModel:
    """Load the saved canonical DSP nonlinear model."""
    model_path = CANONICAL_DSP_OUTPUT_DIR / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"canonical DSP model missing: {model_path}")
    return dsp.model_from_json(json.loads(model_path.read_text()))


def load_aggregate_packet() -> tuple[pd.DataFrame, np.ndarray, dsp.PacketData, list[str]]:
    """Load the collaborator aggregate target and matching DSP packet."""
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


def residual_stacked_oof(
    *,
    block: ResidualBlock,
    canonical_model: dsp.FittedDSPModel,
    packet: dsp.PacketData,
    y_factor: np.ndarray,
    phase: np.ndarray,
    teacher_forced: np.ndarray,
    mcq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fold-local DSP baseline and optional residual-stacked predictions."""
    cv = KFold(n_splits=STACK_N_SPLITS, shuffle=True, random_state=STACK_CV_SEED)
    base_pred = np.full(len(y_factor), np.nan, dtype=np.float64)
    stacked_pred = np.full(len(y_factor), np.nan, dtype=np.float64)

    for train_index, test_index in cv.split(np.arange(len(y_factor))):
        fold_model = dsp.fit_linear_head(
            packet.w[train_index],
            packet.y[train_index],
            packet,
            canonical_model.variant,
            canonical_model.params,
        )
        base_train = -dsp.predict(fold_model, packet.w[train_index])
        base_test = -dsp.predict(fold_model, packet.w[test_index])
        base_pred[test_index] = base_test

        if block.feature_block is None:
            stacked_pred[test_index] = base_test
            continue

        x_train, x_test = transform_block(
            block=block.feature_block,
            train_index=train_index,
            test_index=test_index,
            phase=phase,
            teacher_forced=teacher_forced,
            mcq=mcq,
        )
        residual_target = y_factor[train_index] - base_train
        residual_model = RidgeCV(alphas=RIDGE_ALPHAS)
        residual_model.fit(x_train, residual_target)
        stacked_pred[test_index] = base_test + residual_model.predict(x_test)

    return base_pred, stacked_pred


def write_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    """Write compact HTML diagnostics for the stacked probe."""
    plot_metrics = metrics.melt(
        id_vars=["model"],
        value_vars=["spearman_r", "pearson_r", "r2"],
        var_name="score",
        value_name="value",
    )
    fig = px.bar(
        plot_metrics,
        x="model",
        y="value",
        color="score",
        barmode="group",
        title="DSP baseline versus DSP+MDE residual stackers",
    )
    fig.update_layout(width=1300, height=650, xaxis_tickangle=-25)
    fig.write_html(
        output_dir / "dsp_mde_stacked_metric_bars.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )

    best_model = str(metrics.sort_values("spearman_r", ascending=False).iloc[0]["model"])
    scatter = predictions[["run_name", "actual_y_factor", "dsp_oof", best_model]].copy()
    scatter = scatter.melt(
        id_vars=["run_name", "actual_y_factor"],
        value_vars=["dsp_oof", best_model],
        var_name="model",
        value_name="pred_y_factor",
    )
    fig = px.scatter(
        scatter,
        x="actual_y_factor",
        y="pred_y_factor",
        color="model",
        hover_name="run_name",
        title=f"OOF predictions: DSP baseline versus {best_model}",
    )
    lo = float(min(scatter["actual_y_factor"].min(), scatter["pred_y_factor"].min()))
    hi = float(max(scatter["actual_y_factor"].max(), scatter["pred_y_factor"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="identity", line={"color": "black"}))
    fig.update_layout(width=950, height=750)
    fig.write_html(
        output_dir / "dsp_mde_stacked_actual_vs_pred.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )

    if best_model != "dsp_oof":
        residual_frame = predictions[["run_name", "actual_y_factor", "dsp_oof", best_model]].copy()
        residual_frame["dsp_residual"] = residual_frame["actual_y_factor"] - residual_frame["dsp_oof"]
        residual_frame["mde_correction"] = residual_frame[best_model] - residual_frame["dsp_oof"]
        fig = px.scatter(
            residual_frame,
            x="dsp_residual",
            y="mde_correction",
            hover_name="run_name",
            title=f"Residual correction alignment for {best_model}",
        )
        fig.add_hline(y=0.0, line_color="black")
        fig.add_vline(x=0.0, line_color="black")
        fig.update_layout(width=950, height=750)
        fig.write_html(
            output_dir / "dsp_mde_residual_correction.html",
            include_plotlyjs="cdn",
            config={"toImageButtonOptions": {"scale": 4}},
        )


def main() -> None:
    """Run the DSP+MDE residual-stacking probe."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, y_factor, packet, domains = load_aggregate_packet()
    canonical_model = load_canonical_model()
    phase, teacher_forced, mcq, feature_domains = aligned_features(raw)
    if domains != feature_domains:
        raise ValueError("canonical DSP domains and feature domains differ")

    print(
        f"loaded rows={len(raw)} domains={len(domains)} "
        f"teacher_forced_features={teacher_forced.shape[1]} mcq_features={mcq.shape[1]}",
        flush=True,
    )

    metrics_rows: list[dict[str, object]] = []
    predictions = pd.DataFrame(
        {
            "run_name": raw["run_name"].astype(str),
            "run_id": raw["run_id"],
            "actual_y_factor": y_factor,
        }
    )
    base_reference: np.ndarray | None = None

    for block in RESIDUAL_BLOCKS:
        print(f"evaluating {block.name}", flush=True)
        base_pred, pred = residual_stacked_oof(
            block=block,
            canonical_model=canonical_model,
            packet=packet,
            y_factor=y_factor,
            phase=phase,
            teacher_forced=teacher_forced,
            mcq=mcq,
        )
        if base_reference is None:
            base_reference = base_pred
            predictions["dsp_oof"] = base_pred
        elif not np.allclose(base_reference, base_pred, equal_nan=True):
            raise ValueError(f"DSP baseline changed while evaluating {block.name}")
        predictions[block.name] = pred
        scores = score_predictions(y_factor, pred)
        metrics_rows.append({"model": block.name, "n": int(len(y_factor)), **scores})

    metrics = pd.DataFrame(metrics_rows)
    if metrics.empty:
        raise ValueError("no stacked models were evaluated")
    baseline = metrics.loc[metrics["model"].eq("dsp_oof")].iloc[0]
    for column in ("rmse", "r2", "pearson_r", "spearman_r"):
        metrics[f"{column}_delta_vs_dsp"] = metrics[column] - float(baseline[column])

    metrics = metrics.sort_values("spearman_r", ascending=False).reset_index(drop=True)
    best = metrics.iloc[0].to_dict()
    metrics.to_csv(OUTPUT_DIR / "mde_dsp_stacked_regression_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "mde_dsp_stacked_regression_oof_predictions.csv", index=False)
    write_plots(metrics, predictions, OUTPUT_DIR)

    canonical_summary_path = CANONICAL_DSP_OUTPUT_DIR / "summary.json"
    canonical_summary = json.loads(canonical_summary_path.read_text()) if canonical_summary_path.exists() else {}
    summary = {
        "rows": int(len(raw)),
        "domains": int(len(domains)),
        "target": "aggregate/y_factor",
        "canonical_dsp_model_json": str(CANONICAL_DSP_OUTPUT_DIR / "model.json"),
        "canonical_dsp_summary_oof_spearman": canonical_summary.get("oof_spearman"),
        "fold_local_dsp_oof_spearman": float(baseline["spearman_r"]),
        "best_model": str(best["model"]),
        "best_spearman": float(best["spearman_r"]),
        "best_spearman_delta_vs_dsp": float(best["spearman_r_delta_vs_dsp"]),
        "best_r2": float(best["r2"]),
        "best_r2_delta_vs_dsp": float(best["r2_delta_vs_dsp"]),
        "best_rmse": float(best["rmse"]),
        "best_rmse_delta_vs_dsp": float(best["rmse_delta_vs_dsp"]),
        "teacher_forced_features": int(teacher_forced.shape[1]),
        "mcq_features": int(mcq.shape[1]),
        "cv": {"kind": "KFold", "splits": STACK_N_SPLITS, "seed": STACK_CV_SEED},
        "artifacts": {
            "metrics_csv": str(OUTPUT_DIR / "mde_dsp_stacked_regression_metrics.csv"),
            "predictions_csv": str(OUTPUT_DIR / "mde_dsp_stacked_regression_oof_predictions.csv"),
            "metric_bars_html": str(OUTPUT_DIR / "dsp_mde_stacked_metric_bars.html"),
            "actual_vs_pred_html": str(OUTPUT_DIR / "dsp_mde_stacked_actual_vs_pred.html"),
            "residual_correction_html": str(OUTPUT_DIR / "dsp_mde_residual_correction.html"),
        },
        "caveat": (
            "This probe uses fixed DSP nonlinear parameters from the canonical full-data fit, "
            "then refits the DSP linear head and MDE residual head inside each fold. It tests "
            "observed-checkpoint residual explainability, not arbitrary-mixture optimization."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# DSP + MDE-Style Residual Stacking Probe",
        "",
        "Target: collaborator aggregate `y_factor` on the 242-row 300M signal matrix.",
        "",
        f"- Fold-local canonical DSP OOF Spearman: `{summary['fold_local_dsp_oof_spearman']:.4f}`.",
        f"- Best residual stacker: `{summary['best_model']}`.",
        f"- Best OOF Spearman: `{summary['best_spearman']:.4f}` "
        f"({summary['best_spearman_delta_vs_dsp']:+.4f} vs DSP).",
        f"- Best OOF R2: `{summary['best_r2']:.4f}` ({summary['best_r2_delta_vs_dsp']:+.4f} vs DSP).",
        f"- Best OOF RMSE: `{summary['best_rmse']:.4f}` ({summary['best_rmse_delta_vs_dsp']:+.4f} vs DSP).",
        "",
        "## Model Scores",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Caveat",
        "",
        summary["caveat"],
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
