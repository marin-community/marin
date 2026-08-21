# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Compare MDE-style checkpoint features against phase-weight regressions.

This is an observed-checkpoint regression probe. The MDE-style features here are
measured from already trained 300M checkpoints, so they are not yet queryable
features for arbitrary candidate mixtures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_MATRIX_CSV = (
    SCRIPT_DIR
    / "reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/sent_zip_input/"
    "raw_metric_matrix_300m/raw_metric_matrix_300m.csv"
)
SELECTED_TASKS_CSV = (
    SCRIPT_DIR
    / "reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/sent_raw_metric_matrix_300m_zip/"
    "selected_tasks.csv"
)
MDE_DIR = SCRIPT_DIR / "reference_outputs/mde_checkpoint_features_full_swarm_20260529"
TEACHER_FORCED_FEATURES = MDE_DIR / "teacher_forced_surrogate_probe/teacher_forced_request_nll_wide.parquet"
MCQ_FEATURES = MDE_DIR / "mcq_aggregate_surrogate_probe/mcq_aggregate_features_wide.parquet"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_checkpoint_feature_regression_300m_20260529"

N_SPLITS = 5
CV_SEED = 5416
RIDGE_ALPHAS = np.logspace(-4, 4, 33)
TF_PCA_COMPONENTS = 32


@dataclass(frozen=True)
class FeatureBlock:
    """One model feature-block configuration."""

    name: str
    use_phase: bool
    use_teacher_forced: bool
    use_mcq: bool


FEATURE_BLOCKS = (
    FeatureBlock("phase_ridge", use_phase=True, use_teacher_forced=False, use_mcq=False),
    FeatureBlock("teacher_forced_pca_ridge", use_phase=False, use_teacher_forced=True, use_mcq=False),
    FeatureBlock("mcq_aggregate_ridge", use_phase=False, use_teacher_forced=False, use_mcq=True),
    FeatureBlock("phase_teacher_forced_pca_ridge", use_phase=True, use_teacher_forced=True, use_mcq=False),
    FeatureBlock("phase_mcq_aggregate_ridge", use_phase=True, use_teacher_forced=False, use_mcq=True),
    FeatureBlock("phase_teacher_forced_mcq_pca_ridge", use_phase=True, use_teacher_forced=True, use_mcq=True),
)


def phase_domains(frame: pd.DataFrame) -> list[str]:
    """Return domains with complete two-phase weight columns."""
    phase0 = {column.removeprefix("phase_0_") for column in frame.columns if column.startswith("phase_0_")}
    phase1 = {column.removeprefix("phase_1_") for column in frame.columns if column.startswith("phase_1_")}
    domains = sorted(phase0.intersection(phase1))
    if not domains:
        raise ValueError("no complete phase-weight columns found")
    return domains


def phase_feature_matrix(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    """Return two-phase weights in stable phase/domain order."""
    columns = [f"{phase}_{domain}" for phase in ("phase_0", "phase_1") for domain in domains]
    return frame.loc[:, columns].to_numpy(dtype=np.float64)


def aligned_features(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and align phase, teacher-forced, and MCQ feature matrices."""
    domains = phase_domains(raw)
    phase = phase_feature_matrix(raw, domains)
    tf = pd.read_parquet(TEACHER_FORCED_FEATURES)
    mcq = pd.read_parquet(MCQ_FEATURES)
    run_names = raw["run_name"].astype(str).tolist()
    missing_tf = sorted(set(run_names) - set(map(str, tf.index)))
    missing_mcq = sorted(set(run_names) - set(map(str, mcq.index)))
    if missing_tf or missing_mcq:
        raise ValueError(f"feature tables missing runs: tf={missing_tf[:5]} mcq={missing_mcq[:5]}")
    tf = tf.loc[run_names]
    mcq = mcq.loc[run_names]
    return phase, tf.to_numpy(dtype=np.float64), mcq.to_numpy(dtype=np.float64), domains


def standardize_target(values: np.ndarray) -> np.ndarray:
    """Return finite target values standardized to mean zero and unit variance."""
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 0:
        raise ValueError("target has zero variance")
    return (values - mean) / std


def transform_block(
    *,
    block: FeatureBlock,
    train_index: np.ndarray,
    test_index: np.ndarray,
    phase: np.ndarray,
    teacher_forced: np.ndarray,
    mcq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-fold transforms and return transformed train/test features."""

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    def add_scaled(values: np.ndarray) -> None:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train = imputer.fit_transform(values[train_index])
        test = imputer.transform(values[test_index])
        train_parts.append(scaler.fit_transform(train))
        test_parts.append(scaler.transform(test))

    if block.use_phase:
        add_scaled(phase)

    if block.use_teacher_forced:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train = scaler.fit_transform(imputer.fit_transform(teacher_forced[train_index]))
        test = scaler.transform(imputer.transform(teacher_forced[test_index]))
        n_components = min(TF_PCA_COMPONENTS, train.shape[0] - 1, train.shape[1])
        pca = PCA(n_components=n_components, random_state=CV_SEED)
        train_parts.append(pca.fit_transform(train))
        test_parts.append(pca.transform(test))

    if block.use_mcq:
        add_scaled(mcq)

    if not train_parts:
        raise ValueError(f"feature block has no enabled features: {block.name}")
    return np.hstack(train_parts), np.hstack(test_parts)


def score_predictions(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Return standard held-out prediction metrics."""
    finite = np.isfinite(y) & np.isfinite(pred)
    y = y[finite]
    pred = pred[finite]
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson = float(pearsonr(y, pred).statistic) if len(y) >= 3 else float("nan")
    spearman = float(spearmanr(y, pred).statistic) if len(y) >= 3 else float("nan")
    return {"rmse": rmse, "r2": r2, "pearson_r": pearson, "spearman_r": spearman}


def oof_predictions(
    *,
    block: FeatureBlock,
    y: np.ndarray,
    phase: np.ndarray,
    teacher_forced: np.ndarray,
    mcq: np.ndarray,
) -> np.ndarray:
    """Return fold-local-transform OOF predictions for one feature block."""
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    pred = np.full(len(y), np.nan, dtype=np.float64)
    for train_index, test_index in cv.split(np.arange(len(y))):
        x_train, x_test = transform_block(
            block=block,
            train_index=train_index,
            test_index=test_index,
            phase=phase,
            teacher_forced=teacher_forced,
            mcq=mcq,
        )
        model = RidgeCV(alphas=RIDGE_ALPHAS)
        model.fit(x_train, y[train_index])
        pred[test_index] = model.predict(x_test)
    return pred


def metric_family(metric: str) -> str:
    """Return a compact metric family label."""
    parts = metric.split("/")
    if metric.startswith("eval/paloma/"):
        return "eval/paloma"
    if metric.startswith("eval/uncheatable_eval/"):
        return "eval/uncheatable_eval"
    if metric.startswith("lm_eval/"):
        return "lm_eval"
    if metric.startswith("mcq_smooth/"):
        return "mcq_smooth"
    if metric.startswith("teacher_forced/"):
        return "teacher_forced"
    return "/".join(parts[:2])


def aggregate_target_values(raw: pd.DataFrame, selected: pd.DataFrame) -> dict[str, np.ndarray]:
    """Construct collaborator-style aggregate targets from selected task metrics."""
    task_cols = selected["task"].astype(str).tolist()
    signs = selected["sign"].to_numpy(dtype=np.float64)
    missing = [column for column in task_cols if column not in raw.columns]
    if missing:
        raise ValueError(f"selected task columns missing from raw matrix: {missing[:5]}")
    x = raw.loc[:, task_cols].to_numpy(dtype=np.float64) * signs[None, :]
    if not np.isfinite(x).all():
        raise ValueError("aggregate target construction requires complete selected task matrix")
    z = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    y_mean = z.mean(axis=1)
    u, s, _ = np.linalg.svd(z - z.mean(axis=0, keepdims=True), full_matrices=False)
    y_pc1 = u[:, 0] * s[0]
    if np.corrcoef(y_pc1, y_mean)[0, 1] < 0:
        y_pc1 = -y_pc1
    factor = FactorAnalysis(n_components=5, rotation="varimax", random_state=0)
    factor_scores = factor.fit_transform(z)
    for index in range(factor_scores.shape[1]):
        if np.corrcoef(factor_scores[:, index], y_mean)[0, 1] < 0:
            factor_scores[:, index] = -factor_scores[:, index]
    return {
        "aggregate/y_mean": y_mean,
        "aggregate/y_pc1": y_pc1,
        "aggregate/y_factor": factor_scores.mean(axis=1),
    }


def write_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Write compact HTML diagnostics."""
    pivot = metrics.pivot(index="metric", columns="model", values="spearman_r").reset_index()
    fig = px.imshow(
        pivot.set_index("metric"),
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="OOF Spearman by metric and MDE-style feature block",
    )
    fig.update_layout(width=1200, height=max(700, 18 * len(pivot)))
    fig.write_html(output_dir / "spearman_heatmap.html", include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})

    if "spearman_lift_vs_phase" in metrics.columns:
        merged = metrics.copy()
    else:
        phase = metrics.loc[metrics["model"].eq("phase_ridge"), ["metric", "spearman_r"]].rename(
            columns={"spearman_r": "phase_spearman"}
        )
        merged = metrics.merge(phase, on="metric", how="left")
        merged["spearman_lift_vs_phase"] = merged["spearman_r"] - merged["phase_spearman"]
    best = merged.loc[merged.groupby("metric")["spearman_r"].idxmax()].copy()
    best = best.sort_values("spearman_lift_vs_phase")
    fig = px.bar(
        best,
        x="spearman_lift_vs_phase",
        y="metric",
        color="model",
        orientation="h",
        title="Best model Spearman lift over phase-weight ridge",
    )
    fig.update_layout(width=1200, height=max(700, 18 * len(best)))
    fig.write_html(
        output_dir / "best_spearman_lift_vs_phase.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )

    family = (
        merged.groupby(["family", "model"], as_index=False)
        .agg(mean_spearman=("spearman_r", "mean"), mean_lift_vs_phase=("spearman_lift_vs_phase", "mean"))
        .sort_values(["family", "mean_spearman"], ascending=[True, False])
    )
    fig = px.bar(
        family,
        x="family",
        y="mean_lift_vs_phase",
        color="model",
        barmode="group",
        title="Mean Spearman lift over phase-weight ridge by metric family",
    )
    fig.update_layout(width=1200, height=650)
    fig.write_html(output_dir / "family_mean_lift.html", include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def main() -> None:
    """Run the MDE-style checkpoint-feature regression probe."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_MATRIX_CSV, low_memory=False)
    raw = raw.loc[raw["status"].eq("completed") & raw["row_kind"].eq("signal")].reset_index(drop=True)
    selected = pd.read_csv(SELECTED_TASKS_CSV)
    phase, teacher_forced, mcq, domains = aligned_features(raw)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    target_values: dict[str, np.ndarray] = {}
    for _, target_row in selected.iterrows():
        metric = str(target_row["task"])
        if metric in raw.columns:
            sign = float(target_row["sign"])
            target_values[metric] = pd.to_numeric(raw[metric], errors="coerce").to_numpy(dtype=np.float64) * sign
    target_values.update(aggregate_target_values(raw, selected))

    for metric, values in target_values.items():
        valid = np.isfinite(values)
        if valid.sum() < N_SPLITS * 4:
            continue
        y = standardize_target(values[valid])
        run_names = raw.loc[valid, "run_name"].astype(str).to_numpy()
        metric_predictions = pd.DataFrame({"run_name": run_names, "metric": metric, "y_z": y})
        for block in FEATURE_BLOCKS:
            pred = oof_predictions(
                block=block,
                y=y,
                phase=phase[valid],
                teacher_forced=teacher_forced[valid],
                mcq=mcq[valid],
            )
            scores = score_predictions(y, pred)
            metrics_rows.append(
                {
                    "metric": metric,
                    "family": metric_family(metric),
                    "model": block.name,
                    "n": int(valid.sum()),
                    **scores,
                }
            )
            metric_predictions[block.name] = pred
        prediction_frames.append(metric_predictions)

    metrics = pd.DataFrame(metrics_rows)
    if metrics.empty:
        raise ValueError("no metrics were evaluated")
    phase = metrics.loc[metrics["model"].eq("phase_ridge"), ["metric", "spearman_r", "r2"]].rename(
        columns={"spearman_r": "phase_spearman", "r2": "phase_r2"}
    )
    metrics = metrics.merge(phase, on="metric", how="left")
    metrics["spearman_lift_vs_phase"] = metrics["spearman_r"] - metrics["phase_spearman"]
    metrics["r2_lift_vs_phase"] = metrics["r2"] - metrics["phase_r2"]
    best = metrics.loc[metrics.groupby("metric")["spearman_r"].idxmax()].sort_values(
        "spearman_lift_vs_phase", ascending=False
    )
    metrics.to_csv(OUTPUT_DIR / "mde_checkpoint_feature_regression_metrics.csv", index=False)
    best.to_csv(OUTPUT_DIR / "mde_checkpoint_feature_regression_best_by_spearman.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "mde_checkpoint_feature_regression_oof_predictions.csv", index=False
    )
    write_plots(metrics, OUTPUT_DIR)

    phase_best = metrics.loc[metrics["model"].eq("phase_ridge")].set_index("metric")
    hybrid = metrics.loc[metrics["model"].eq("phase_teacher_forced_mcq_pca_ridge")].set_index("metric")
    summary = {
        "rows": int(len(raw)),
        "targets_evaluated": int(metrics["metric"].nunique()),
        "domains": int(len(domains)),
        "teacher_forced_features": int(teacher_forced.shape[1]),
        "mcq_features": int(mcq.shape[1]),
        "cv": {"kind": "KFold", "splits": N_SPLITS, "seed": CV_SEED},
        "teacher_forced_pca_components": TF_PCA_COMPONENTS,
        "best_model_counts": best["model"].value_counts().to_dict(),
        "phase_mean_spearman": float(phase_best["spearman_r"].mean()),
        "hybrid_mean_spearman": float(hybrid["spearman_r"].mean()),
        "hybrid_mean_spearman_lift_vs_phase": float((hybrid["spearman_r"] - phase_best["spearman_r"]).mean()),
        "hybrid_median_spearman_lift_vs_phase": float((hybrid["spearman_r"] - phase_best["spearman_r"]).median()),
        "best_mean_spearman_lift_vs_phase": float(best["spearman_lift_vs_phase"].mean()),
        "best_median_spearman_lift_vs_phase": float(best["spearman_lift_vs_phase"].median()),
        "best_improves_over_phase_count": int((best["spearman_lift_vs_phase"] > 0).sum()),
        "best_regresses_vs_phase_count": int((best["spearman_lift_vs_phase"] < 0).sum()),
        "artifacts": {
            "metrics_csv": str(OUTPUT_DIR / "mde_checkpoint_feature_regression_metrics.csv"),
            "best_csv": str(OUTPUT_DIR / "mde_checkpoint_feature_regression_best_by_spearman.csv"),
            "predictions_csv": str(OUTPUT_DIR / "mde_checkpoint_feature_regression_oof_predictions.csv"),
            "spearman_heatmap": str(OUTPUT_DIR / "spearman_heatmap.html"),
            "best_lift_plot": str(OUTPUT_DIR / "best_spearman_lift_vs_phase.html"),
            "family_lift_plot": str(OUTPUT_DIR / "family_mean_lift.html"),
        },
        "caveat": (
            "These are observed-checkpoint behavioral features. They do not yet provide a "
            "lambda-queryable mixture-search surrogate for unseen candidate mixtures."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# MDE-Style Checkpoint Feature Regression Probe",
        "",
        "This probe compares phase-weight ridge against teacher-forced and MCQ MDE-style checkpoint features on observed 300M checkpoints.",
        "",
        f"- Rows: `{summary['rows']}`.",
        f"- Targets evaluated: `{summary['targets_evaluated']}`.",
        f"- Phase-weight mean OOF Spearman: `{summary['phase_mean_spearman']:.3f}`.",
        f"- Full hybrid mean OOF Spearman: `{summary['hybrid_mean_spearman']:.3f}`.",
        f"- Full hybrid mean Spearman lift vs phase: `{summary['hybrid_mean_spearman_lift_vs_phase']:+.3f}`.",
        f"- Best-per-target mean Spearman lift vs phase: `{summary['best_mean_spearman_lift_vs_phase']:+.3f}`.",
        f"- Best model improves over phase on `{summary['best_improves_over_phase_count']}` targets and regresses on `{summary['best_regresses_vs_phase_count']}` targets.",
        "",
        "## Best Model Counts",
        "",
        *[f"- `{model}`: `{count}`." for model, count in summary["best_model_counts"].items()],
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
