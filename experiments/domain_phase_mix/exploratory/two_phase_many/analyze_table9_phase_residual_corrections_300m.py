# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "fsspec", "gcsfs", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Evaluate small two-phase DSP residual-correction variants on Table-9.

This is a local diagnostic, not a launcher. It starts from the expanded 300M
Table-9 diagnostic panel and asks whether simple phase-aware corrections reduce
frontier optimism or improve post-selection regret beyond the existing
effective-exposure DSP predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_INPUT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_extra_300m_heldout_eval_20260630"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_phase_residual_corrections_20260630"
DEFAULT_AGGREGATE_DSP_MODEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "dsp_effective_exposure"
    / "table9_macro_bpb"
    / "linear_reg_0.0001"
    / "model.json"
)
BASELINE_PRED_COL = "aggregate_effective_exposure_dsp_l2_0p0001"
TARGET_COL = "table9_macro_bpb"
LOWER_TAIL_FRAC = 0.15
N_SPLITS = 5
CV_SEED = 0
RIDGE_ALPHAS = np.asarray([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0])
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class VariantResult:
    variant: str
    n_rows: int
    rmse: float
    mae: float
    pearson: float
    spearman: float
    fold_mean_regret_at_1: float
    fold_mean_regret_at_3: float
    fold_mean_regret_at_5: float
    global_regret_at_1: float
    global_regret_at_3: float
    global_regret_at_5: float
    lower_tail_optimism: float
    low_tail_rmse: float
    selected_at_1_run_name: str
    selected_at_1_actual: float
    selected_at_1_predicted: float
    selected_at_1_optimism: float
    best_observed_run_name: str
    best_observed_bpb: float
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aggregate-dsp-model", type=Path, default=DEFAULT_AGGREGATE_DSP_MODEL)
    return parser.parse_args()


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    residual = pred - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = float(pearsonr(y, pred).statistic) if np.std(y) > 0.0 and np.std(pred) > 0.0 else float("nan")
    spearman = float(spearmanr(y, pred).statistic) if np.std(y) > 0.0 and np.std(pred) > 0.0 else float("nan")
    return rmse, mae, pearson, spearman


def global_regret_at_k(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    selected = np.argsort(pred)[: min(k, len(y))]
    return float(np.min(y[selected]) - np.min(y))


def fold_mean_regret_at_k(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], k: int) -> float:
    regrets: list[float] = []
    for _train_idx, test_idx in folds:
        selected = test_idx[np.argsort(pred[test_idx])[: min(k, len(test_idx))]]
        regrets.append(float(np.min(y[selected]) - np.min(y[test_idx])))
    return float(np.mean(regrets))


def lower_tail_optimism(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(pred)[:tail_count]
    residual = pred[tail_idx] - y[tail_idx]
    optimism = float(np.mean(np.maximum(y[tail_idx] - pred[tail_idx], 0.0)))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return optimism, rmse


def stratified_folds(groups: pd.Series, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fold_id = np.full(len(groups), -1, dtype=int)
    for _group, indices in groups.groupby(groups, sort=True).indices.items():
        shuffled = np.asarray(list(indices), dtype=int)
        rng.shuffle(shuffled)
        for position, row_idx in enumerate(shuffled):
            fold_id[row_idx] = position % n_splits
    if np.any(fold_id < 0):
        raise ValueError("Failed to assign all rows to CV folds")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(groups))
    for fold in range(n_splits):
        test_idx = np.flatnonzero(fold_id == fold)
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f"Empty fold {fold}")
        folds.append((train_idx, test_idx))
    return folds


def summarize_variant(
    *,
    variant: str,
    frame: pd.DataFrame,
    pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    notes: str,
) -> VariantResult:
    y = frame[TARGET_COL].to_numpy(dtype=float)
    run_names = frame["run_name"].astype(str).to_numpy()
    rmse, mae, pearson, spearman = regression_metrics(y, pred)
    optimism, low_tail_rmse = lower_tail_optimism(y, pred)
    selected_idx = int(np.argmin(pred))
    best_idx = int(np.argmin(y))
    return VariantResult(
        variant=variant,
        n_rows=int(len(frame)),
        rmse=rmse,
        mae=mae,
        pearson=pearson,
        spearman=spearman,
        fold_mean_regret_at_1=fold_mean_regret_at_k(y, pred, folds, 1),
        fold_mean_regret_at_3=fold_mean_regret_at_k(y, pred, folds, 3),
        fold_mean_regret_at_5=fold_mean_regret_at_k(y, pred, folds, 5),
        global_regret_at_1=global_regret_at_k(y, pred, 1),
        global_regret_at_3=global_regret_at_k(y, pred, 3),
        global_regret_at_5=global_regret_at_k(y, pred, 5),
        lower_tail_optimism=optimism,
        low_tail_rmse=low_tail_rmse,
        selected_at_1_run_name=str(run_names[selected_idx]),
        selected_at_1_actual=float(y[selected_idx]),
        selected_at_1_predicted=float(pred[selected_idx]),
        selected_at_1_optimism=float(y[selected_idx] - pred[selected_idx]),
        best_observed_run_name=str(run_names[best_idx]),
        best_observed_bpb=float(y[best_idx]),
        notes=notes,
    )


def domain_family(domain: str) -> str:
    if "/cc/" in domain or "dolma3_cc/" in domain:
        return "cc_high" if domain.endswith("_high") else "cc_low"
    if "stack_edu" in domain:
        return "stack_edu"
    if "synth_code" in domain:
        return "synth_code"
    if "synth_math" in domain or "finemath" in domain:
        return "math"
    if "synth_qa" in domain or "synth_instruction" in domain or "synth_thinking" in domain:
        return "synthetic_reasoning"
    if "wikipedia" in domain:
        return "wikipedia"
    if "arxiv" in domain or "stem_heavy" in domain:
        return "stem_text"
    if "common_crawl_hq" in domain or "olmocr" in domain:
        return "hq_web_text"
    return "other"


def load_frames(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(input_dir / "expanded_300m_table9_diagnostic_panel.csv", low_memory=False)
    predictions = pd.read_csv(input_dir / "expanded_300m_table9_cv_predictions.csv", low_memory=False)
    merged = panel.merge(
        predictions[["run_name", BASELINE_PRED_COL]],
        on="run_name",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(panel):
        raise ValueError(f"Prediction merge lost rows: panel={len(panel)} merged={len(merged)}")
    return merged.reset_index(drop=True), predictions


def build_packet(panel: pd.DataFrame) -> tuple[dsp.PacketData, list[str], list[str], np.ndarray]:
    _signal, columns, domains, _natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, TARGET_COL)
    return packet, columns, domains, token_counts


def collapsed_dsp_prediction(model_path: Path, packet: dsp.PacketData) -> np.ndarray:
    model = dsp.model_from_json(json.loads(model_path.read_text()))
    aggregate = np.einsum("p,npd->nd", base.PHASE_FRACTIONS, packet.w)
    collapsed = np.stack([aggregate, aggregate], axis=1)
    return dsp.predict(model, collapsed)


def entropy(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def global_phase_features(packet: dsp.PacketData, natural: np.ndarray) -> pd.DataFrame:
    w = packet.w
    e0 = w[:, 0, :] * packet.c0[None, :]
    e1 = w[:, 1, :] * packet.c1[None, :]
    aggregate = np.einsum("p,npd->nd", base.PHASE_FRACTIONS, w)
    reference = np.stack([natural, natural], axis=0)
    out = pd.DataFrame(
        {
            "phase_tv": 0.5 * np.abs(w[:, 0, :] - w[:, 1, :]).sum(axis=1),
            "mean_tv_to_proportional": 0.5 * np.abs(w - reference[None, :, :]).sum(axis=2).mean(axis=1),
            "aggregate_tv_to_proportional": 0.5 * np.abs(aggregate - natural[None, :]).sum(axis=1),
            "log_exposure_phase_l2": np.sqrt(np.mean((np.log1p(e1) - np.log1p(e0)) ** 2, axis=1)),
            "phase_0_entropy": entropy(w[:, 0, :]),
            "phase_1_entropy": entropy(w[:, 1, :]),
            "phase_entropy_delta": entropy(w[:, 1, :]) - entropy(w[:, 0, :]),
            "max_phase_weight": np.max(w, axis=(1, 2)),
        }
    )
    return out


def family_phase_features(packet: dsp.PacketData, domains: list[str]) -> pd.DataFrame:
    e0 = packet.w[:, 0, :] * packet.c0[None, :]
    e1 = packet.w[:, 1, :] * packet.c1[None, :]
    log_delta = np.log1p(e1) - np.log1p(e0)
    rows: dict[str, np.ndarray] = {}
    families = sorted({domain_family(domain) for domain in domains})
    for family in families:
        mask = np.asarray([domain_family(domain) == family for domain in domains])
        rows[f"family_{family}_log_exposure_delta"] = log_delta[:, mask].mean(axis=1)
        rows[f"family_{family}_abs_log_exposure_delta"] = np.abs(log_delta[:, mask]).mean(axis=1)
        rows[f"family_{family}_phase1_minus_phase0_weight"] = packet.w[:, 1, mask].sum(axis=1) - packet.w[:, 0, mask].sum(axis=1)
    return pd.DataFrame(rows)


def domain_phase_features(packet: dsp.PacketData, domains: list[str]) -> pd.DataFrame:
    e0 = packet.w[:, 0, :] * packet.c0[None, :]
    e1 = packet.w[:, 1, :] * packet.c1[None, :]
    log_delta = np.log1p(e1) - np.log1p(e0)
    data: dict[str, np.ndarray] = {}
    for idx, domain in enumerate(domains):
        safe = domain.replace("/", "__").replace("-", "_")
        data[f"domain_{safe}_log_exposure_delta"] = log_delta[:, idx]
    return pd.DataFrame(data)


def residual_ridge_oof(
    *,
    y: np.ndarray,
    base_pred: np.ndarray,
    features: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    x = features.to_numpy(dtype=float)
    residual = y - base_pred
    out = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in folds:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        model.fit(x[train_idx], residual[train_idx])
        out[test_idx] = base_pred[test_idx] + model.predict(x[test_idx])
    return out


def blend_ridge_oof(
    *,
    y: np.ndarray,
    features: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    x = features.to_numpy(dtype=float)
    out = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in folds:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        model.fit(x[train_idx], y[train_idx])
        out[test_idx] = model.predict(x[test_idx])
    return out


def conservative_penalty_oof(
    *,
    y: np.ndarray,
    base_pred: np.ndarray,
    risk: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    risk_std = (risk - np.mean(risk)) / (np.std(risk) + 1e-12)
    lambdas = np.asarray([0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.016, 0.02, 0.03, 0.05])
    out = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in folds:
        best_lambda = 0.0
        best_key: tuple[float, float] | None = None
        for lam in lambdas:
            train_pred = base_pred[train_idx] + lam * risk_std[train_idx]
            regret = global_regret_at_k(y[train_idx], train_pred, 1)
            optimism, _rmse = lower_tail_optimism(y[train_idx], train_pred)
            key = (regret, optimism)
            if best_key is None or key < best_key:
                best_key = key
                best_lambda = float(lam)
        out[test_idx] = base_pred[test_idx] + best_lambda * risk_std[test_idx]
    return out


def write_scatter(path: Path, predictions: pd.DataFrame) -> None:
    fig = go.Figure()
    for variant, group in predictions.groupby("variant", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group[TARGET_COL],
                y=group["prediction"],
                mode="markers",
                name=str(variant),
                text=group["run_name"],
                customdata=np.stack([group["diagnostic_group"], group["diagnostic_family"]], axis=1),
                hovertemplate=(
                    "run=%{text}<br>group=%{customdata[0]}<br>family=%{customdata[1]}"
                    "<br>actual=%{x:.5f}<br>pred=%{y:.5f}<extra></extra>"
                ),
            )
        )
    lo = min(float(predictions[TARGET_COL].min()), float(predictions["prediction"].min()))
    hi = max(float(predictions[TARGET_COL].max()), float(predictions["prediction"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line={"color": "#64748b", "dash": "dash"}))
    fig.update_layout(
        title="Table-9 phase-correction variants: OOF predictions",
        xaxis_title="Observed Table-9 macro BPB",
        yaxis_title="Predicted Table-9 macro BPB",
        template="plotly_white",
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_regret_plot(path: Path, summary: pd.DataFrame) -> None:
    metrics = ["fold_mean_regret_at_1", "lower_tail_optimism", "rmse", "global_regret_at_1"]
    fig = go.Figure()
    for metric in metrics:
        ordered = summary.sort_values(metric, ascending=True)
        fig.add_trace(
            go.Bar(
                x=ordered["variant"],
                y=ordered[metric],
                name=metric,
                visible=metric == metrics[0],
                hovertemplate=f"{metric}=%{{y:.6f}}<extra></extra>",
            )
        )
    buttons = []
    for idx, metric in enumerate(metrics):
        visible = [False] * len(metrics)
        visible[idx] = True
        buttons.append({"label": metric, "method": "update", "args": [{"visible": visible}, {"yaxis.title.text": metric}]})
    fig.update_layout(
        title="Table-9 phase-correction variant diagnostics",
        xaxis_title="Variant",
        yaxis_title=metrics[0],
        template="plotly_white",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.16}],
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, _existing_predictions = load_frames(args.input_dir)
    packet, _columns, domains, _token_counts = build_packet(panel)
    baseline_row = panel["run_name"].eq("baseline_proportional")
    if int(baseline_row.sum()) != 1:
        raise ValueError("Expected one baseline_proportional row")
    natural = packet.w[int(np.flatnonzero(baseline_row)[0]), 0].copy()
    y = panel[TARGET_COL].to_numpy(dtype=float)
    base_pred = panel[BASELINE_PRED_COL].to_numpy(dtype=float)
    collapsed_pred = collapsed_dsp_prediction(args.aggregate_dsp_model, packet)
    folds = stratified_folds(panel["diagnostic_group"], n_splits=N_SPLITS, seed=CV_SEED)

    global_features = global_phase_features(packet, natural)
    family_features = family_phase_features(packet, domains)
    domain_features = domain_phase_features(packet, domains)
    collapsed_features = pd.concat(
        [
            pd.DataFrame(
                {
                    "base_pred": base_pred,
                    "collapsed_pred": collapsed_pred,
                    "base_minus_collapsed_pred": base_pred - collapsed_pred,
                }
            ),
            global_features,
        ],
        axis=1,
    )

    variant_predictions: dict[str, tuple[np.ndarray, str]] = {
        "baseline_aggregate_dsp": (base_pred, "Existing expanded-panel aggregate effective-exposure DSP OOF prediction."),
        "residual_global_phase_ridge": (
            residual_ridge_oof(y=y, base_pred=base_pred, features=global_features, folds=folds),
            "OOF ridge correction to aggregate DSP residuals using global phase diagnostics.",
        ),
        "residual_family_phase_ridge": (
            residual_ridge_oof(y=y, base_pred=base_pred, features=pd.concat([global_features, family_features], axis=1), folds=folds),
            "OOF ridge correction using global diagnostics plus coarse domain-family phase contrasts.",
        ),
        "residual_domain_phase_ridge": (
            residual_ridge_oof(y=y, base_pred=base_pred, features=pd.concat([global_features, domain_features], axis=1), folds=folds),
            "OOF ridge correction using global diagnostics plus per-domain phase exposure contrasts.",
        ),
        "collapsed_prediction_blend_ridge": (
            blend_ridge_oof(y=y, features=collapsed_features, folds=folds),
            "OOF ridge blend of current DSP prediction, collapsed-mixture DSP prediction, and global phase diagnostics.",
        ),
        "collapsed_family_blend_ridge": (
            blend_ridge_oof(y=y, features=pd.concat([collapsed_features, family_features], axis=1), folds=folds),
            "OOF ridge blend with collapsed prediction and family phase-contrast features.",
        ),
        "conservative_phase_risk_penalty": (
            conservative_penalty_oof(
                y=y,
                base_pred=base_pred,
                risk=global_features["log_exposure_phase_l2"].to_numpy(dtype=float)
                + global_features["phase_tv"].to_numpy(dtype=float),
                folds=folds,
            ),
            "Fold-tuned positive penalty on phase-risk score; tests optimism control without changing DSP fit.",
        ),
    }

    results = [
        summarize_variant(variant=variant, frame=panel, pred=pred, folds=folds, notes=notes)
        for variant, (pred, notes) in variant_predictions.items()
    ]
    summary = pd.DataFrame.from_records([asdict(result) for result in results])
    summary = summary.sort_values(
        ["fold_mean_regret_at_1", "global_regret_at_1", "lower_tail_optimism", "rmse"],
        ascending=[True, True, True, True],
    )
    prediction_frames: list[pd.DataFrame] = []
    for variant, (pred, _notes) in variant_predictions.items():
        frame = panel[
            [
                "run_name",
                "diagnostic_family",
                "diagnostic_group",
                TARGET_COL,
            ]
        ].copy()
        frame["variant"] = variant
        frame["prediction"] = pred
        frame["residual"] = pred - y
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    feature_manifest = {
        "global_features": global_features.columns.tolist(),
        "family_features": family_features.columns.tolist(),
        "domain_features": domain_features.columns.tolist(),
        "ridge_alphas": RIDGE_ALPHAS.tolist(),
        "n_rows": int(len(panel)),
        "n_domains": int(len(domains)),
        "domains": domains,
    }
    summary.to_csv(args.output_dir / "phase_correction_variant_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "phase_correction_variant_predictions.csv", index=False)
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n")
    write_scatter(args.output_dir / "phase_correction_oof_scatter.html", predictions)
    write_regret_plot(args.output_dir / "phase_correction_regret_summary.html", summary)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
