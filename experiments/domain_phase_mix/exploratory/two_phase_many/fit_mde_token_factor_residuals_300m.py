# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "plotly", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Probe token-factor residual models over DSP using dense MDE token losses.

This script tests whether the MDE-style per-token loss matrix is useful as a
residual representation rather than as the kNN token-MDE predictor in
``fit_mde_token_dsp_uncheatable_300m.py``.

It compares:

* DSP-only OOF predictions;
* observed-token factor upper bounds, which use held-out checkpoint token
  residuals and are not deployable for unseen candidate mixtures;
* queryable factor models, which learn ``mixture -> token factors`` on the
  train fold before predicting held-out target residuals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_mde_token_dsp_uncheatable_300m import (
    DEFAULT_METADATA_CSV,
    DEFAULT_RAW_CSV,
    LOG2E,
    RIDGE_ALPHAS,
    load_packet,
    load_token_matrix,
    mixture_feature_matrix,
    path_exists,
    prepare_output_dir,
    score_predictions,
    write_text,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DENSE_FEATURE_DIR = SCRIPT_DIR / "reference_outputs/mde_uncheatable_token_dense_matrix_300m_20260531"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_token_factor_residuals_300m_20260531"
DEFAULT_TARGETS = [
    "eval/uncheatable_eval/bpb",
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
]
DEFAULT_COMPONENT_GRID = (4, 8, 16)
DEFAULT_QUERY_FEATURE_MODES = ("phase_log_exposure", "dsp_features")
DEFAULT_FACTOR_METHODS = ("pca", "pls")
TOKEN_RIDGE_ALPHA = 10.0


@dataclass(frozen=True)
class FactorSpec:
    """One token-factor residual probe configuration."""

    method: str
    feature_mode: str
    components: int
    queryable: bool

    @property
    def name(self) -> str:
        prefix = "queryable" if self.queryable else "observed_upper_bound"
        return f"{prefix}_{self.method}_{self.feature_mode}_c{self.components}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default=str(DEFAULT_DENSE_FEATURE_DIR))
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--dsp-variant", choices=sorted(dsp.VARIANTS), default="effective_exposure")
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--coarse-top-k", type=int, default=8)
    parser.add_argument("--basin-hopping-iters", type=int, default=6)
    parser.add_argument("--cv-seed", type=int, default=dsp.CV_SEED)
    parser.add_argument("--splits", type=int, default=dsp.N_SPLITS)
    parser.add_argument("--component", type=int, action="append", default=[])
    parser.add_argument("--method", choices=DEFAULT_FACTOR_METHODS, action="append", default=[])
    parser.add_argument("--feature-mode", choices=DEFAULT_QUERY_FEATURE_MODES, action="append", default=[])
    parser.add_argument(
        "--queryable-only",
        action="store_true",
        help="Skip observed-token upper bounds and fit only deployable queryable factor models.",
    )
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help="Skip queryable models and fit only observed-token upper bounds.",
    )
    return parser.parse_args()


def path_join(base: str, name: str) -> str:
    """Join a local path or URI with one child component."""
    if "://" in base:
        return f"{base.rstrip('/')}/{name}"
    return str(Path(base) / name)


def safe_label(value: str) -> str:
    """Make a target label safe for filenames."""
    return value.replace("/", "__").replace(":", "_")


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute ordinary R^2."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def token_bpb_matrix(token_nll: np.ndarray, token_bytes: np.ndarray) -> np.ndarray:
    """Normalize token NLLs to token-level BPB values."""
    return np.asarray(token_nll * (LOG2E / token_bytes[None, :]), dtype=np.float64)


def residualize_tokens(
    *,
    token_train: np.ndarray,
    token_test: np.ndarray,
    features_train: np.ndarray,
    features_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the mixture-predictable token component within one fold."""
    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(features_train)
    test_features = feature_scaler.transform(features_test)
    token_model = Ridge(alpha=TOKEN_RIDGE_ALPHA)
    token_model.fit(train_features, token_train)
    return token_train - token_model.predict(train_features), token_test - token_model.predict(test_features)


def observed_factor_scores(
    *,
    method: str,
    token_train: np.ndarray,
    token_test: np.ndarray,
    target_train: np.ndarray,
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test token-factor scores using observed token residuals."""
    token_scaler = StandardScaler()
    train_scaled = token_scaler.fit_transform(token_train)
    test_scaled = token_scaler.transform(token_test)
    n_components = min(components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    if method == "pca":
        factor_model = PCA(n_components=n_components, random_state=0)
        return factor_model.fit_transform(train_scaled), factor_model.transform(test_scaled)
    if method == "pls":
        factor_model = PLSRegression(n_components=n_components, scale=False)
        train_scores = factor_model.fit_transform(train_scaled, target_train)[0]
        test_scores = factor_model.transform(test_scaled)
        return np.asarray(train_scores, dtype=np.float64), np.asarray(test_scores, dtype=np.float64)
    raise ValueError(f"Unknown factor method: {method}")


def queryable_factor_scores(
    *,
    method: str,
    token_train: np.ndarray,
    features_train: np.ndarray,
    features_test: np.ndarray,
    target_train: np.ndarray,
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn token factors from train tokens, then predict factors from mixtures."""
    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(features_train)
    test_features = feature_scaler.transform(features_test)
    train_scores, _unused_observed_test = observed_factor_scores(
        method=method,
        token_train=token_train,
        token_test=token_train[:1],
        target_train=target_train,
        components=components,
    )
    factor_surrogate = RidgeCV(alphas=RIDGE_ALPHAS)
    factor_surrogate.fit(train_features, train_scores)
    return train_scores, factor_surrogate.predict(test_features)


def fit_factor_oof(
    *,
    packet: dsp.PacketData,
    token_bpb: np.ndarray,
    dsp_model: dsp.FittedDSPModel,
    spec: FactorSpec,
    splits: int,
    cv_seed: int,
) -> np.ndarray:
    """Fit OOF DSP plus token-factor residual predictions for one spec."""
    base_features = mixture_feature_matrix(packet, spec.feature_mode, dsp_model)
    cv = KFold(n_splits=splits, shuffle=True, random_state=cv_seed)
    oof = np.zeros_like(packet.y)
    for train_idx, test_idx in cv.split(packet.w):
        fold_dsp_model = dsp.fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            dsp_model.variant,
            dsp_model.params,
        )
        base_train = dsp.predict(fold_dsp_model, packet.w[train_idx])
        base_test = dsp.predict(fold_dsp_model, packet.w[test_idx])
        token_resid_train, token_resid_test = residualize_tokens(
            token_train=token_bpb[train_idx],
            token_test=token_bpb[test_idx],
            features_train=base_features[train_idx],
            features_test=base_features[test_idx],
        )
        target_resid_train = packet.y[train_idx] - base_train
        if spec.queryable:
            train_scores, test_scores = queryable_factor_scores(
                method=spec.method,
                token_train=token_resid_train,
                features_train=base_features[train_idx],
                features_test=base_features[test_idx],
                target_train=target_resid_train,
                components=spec.components,
            )
        else:
            train_scores, test_scores = observed_factor_scores(
                method=spec.method,
                token_train=token_resid_train,
                token_test=token_resid_test,
                target_train=target_resid_train,
                components=spec.components,
            )
        residual_model = RidgeCV(alphas=RIDGE_ALPHAS)
        residual_model.fit(train_scores, target_resid_train)
        oof[test_idx] = base_test + residual_model.predict(test_scores)
    return oof


def dsp_oof_predictions(packet: dsp.PacketData, model: dsp.FittedDSPModel, splits: int, cv_seed: int) -> np.ndarray:
    """Compute fixed-nonlinear-parameter DSP OOF predictions."""
    oof = np.zeros_like(packet.y)
    cv = KFold(n_splits=splits, shuffle=True, random_state=cv_seed)
    for train_idx, test_idx in cv.split(packet.w):
        fold_model = dsp.fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
        )
        oof[test_idx] = dsp.predict(fold_model, packet.w[test_idx])
    return oof


def target_fit_rows(
    *,
    target: str,
    raw_csv: Path,
    metadata_csv: Path,
    feature_dir: str,
    variant: dsp.VariantSpec,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
    splits: int,
    cv_seed: int,
    component_grid: tuple[int, ...],
    feature_modes: tuple[str, ...],
    methods: tuple[str, ...],
    queryable_options: tuple[bool, ...],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Fit all token-factor probes for one target."""
    raw, packet = load_packet(raw_csv, metadata_csv, target)
    token_nll, token_bytes, _token_datasets, _token_keys = load_token_matrix(feature_dir, raw["run_name"].astype(str))
    token_bpb = token_bpb_matrix(token_nll, token_bytes)
    dsp_model, _tuning = dsp.fit_variant(
        packet,
        variant,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
        basin_hopping_iters=basin_hopping_iters,
    )
    dsp_oof = dsp_oof_predictions(packet, dsp_model, splits, cv_seed)
    predictions = pd.DataFrame(
        {
            "target": target,
            "run_name": raw["run_name"].astype(str),
            "actual": packet.y,
            "dsp_oof": dsp_oof,
        }
    )
    rows: list[dict[str, object]] = [{"target": target, "model": "dsp_oof", **score_predictions(packet.y, dsp_oof)}]
    specs = [
        FactorSpec(method=method, feature_mode=feature_mode, components=components, queryable=queryable)
        for feature_mode in feature_modes
        for method in methods
        for components in component_grid
        for queryable in queryable_options
    ]
    for spec in specs:
        print(f"target={target} fitting {spec.name}", flush=True)
        oof = fit_factor_oof(
            packet=packet,
            token_bpb=token_bpb,
            dsp_model=dsp_model,
            spec=spec,
            splits=splits,
            cv_seed=cv_seed,
        )
        predictions[spec.name] = oof
        rows.append(
            {
                "target": target,
                "model": spec.name,
                "method": spec.method,
                "feature_mode": spec.feature_mode,
                "components": spec.components,
                "queryable": spec.queryable,
                **score_predictions(packet.y, oof),
            }
        )
    return rows, predictions


def write_outputs(output_dir: str, metrics: pd.DataFrame, predictions: pd.DataFrame, summary: dict[str, object]) -> None:
    """Write summary tables and plots."""
    prepare_output_dir(output_dir)
    metrics_path = Path(output_dir) / "token_factor_residual_metrics.csv"
    predictions_path = Path(output_dir) / "token_factor_residual_oof_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    write_text(output_dir, "summary.json", json.dumps(summary, indent=2, sort_keys=True))

    top = metrics.loc[metrics["model"].ne("dsp_oof")].copy()
    top["spearman_delta_vs_dsp"] = top["spearman_r"] - top["dsp_spearman"]
    fig = px.bar(
        top.sort_values(["target", "spearman_delta_vs_dsp"], ascending=[True, False]).groupby("target").head(8),
        x="model",
        y="spearman_delta_vs_dsp",
        color="queryable_label",
        facet_col="target_label",
        facet_col_wrap=2,
        title="Best token-factor residual lifts over DSP by target",
    )
    fig.update_layout(width=1700, height=1100, xaxis_tickangle=-25)
    write_text(
        output_dir,
        "token_factor_residual_lifts.html",
        fig.to_html(include_plotlyjs="cdn", config={"toImageButtonOptions": {"format": "png", "scale": 4}}),
    )

    best = metrics.sort_values(["target", "spearman_r"], ascending=[True, False]).groupby("target").head(1)
    fig = px.bar(
        best,
        x="target_label",
        y="spearman_r",
        color="queryable_label",
        hover_name="model",
        title="Best model by target",
    )
    fig.update_layout(width=1300, height=600, xaxis_tickangle=-25)
    write_text(
        output_dir,
        "token_factor_best_by_target.html",
        fig.to_html(include_plotlyjs="cdn", config={"toImageButtonOptions": {"format": "png", "scale": 4}}),
    )


def main() -> None:
    """Run the token-factor residual probe."""
    args = parse_args()
    if args.queryable_only and args.observed_only:
        raise ValueError("--queryable-only and --observed-only are mutually exclusive")
    if not path_exists(path_join(args.feature_dir, "summary.json")):
        raise FileNotFoundError(f"Dense feature summary not found under {args.feature_dir}")
    targets = args.target or DEFAULT_TARGETS
    variant = dsp.VARIANTS[args.dsp_variant]
    component_grid = tuple(args.component or DEFAULT_COMPONENT_GRID)
    feature_modes = tuple(args.feature_mode or DEFAULT_QUERY_FEATURE_MODES)
    methods = tuple(args.method or DEFAULT_FACTOR_METHODS)
    if args.queryable_only:
        queryable_options = (True,)
    elif args.observed_only:
        queryable_options = (False,)
    else:
        queryable_options = (False, True)
    all_rows: list[dict[str, object]] = []
    prediction_frames = []
    for target in targets:
        rows, predictions = target_fit_rows(
            target=target,
            raw_csv=args.raw_csv,
            metadata_csv=args.metadata_csv,
            feature_dir=args.feature_dir,
            variant=variant,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
            basin_hopping_iters=args.basin_hopping_iters,
            splits=args.splits,
            cv_seed=args.cv_seed,
            component_grid=component_grid,
            feature_modes=feature_modes,
            methods=methods,
            queryable_options=queryable_options,
        )
        all_rows.extend(rows)
        prediction_frames.append(predictions)

    metrics = pd.DataFrame.from_records(all_rows)
    metrics["target_label"] = metrics["target"].str.replace("eval/uncheatable_eval/", "", regex=False)
    metrics["target_label"] = metrics["target_label"].str.replace("/bpb", "", regex=False)
    metrics["queryable_label"] = np.where(
        metrics["model"].eq("dsp_oof"),
        "dsp",
        np.where(metrics["queryable"].fillna(False).astype(bool), "queryable", "observed upper bound"),
    )
    dsp_by_target = metrics.loc[metrics["model"].eq("dsp_oof"), ["target", "spearman_r", "r2", "rmse"]]
    dsp_by_target = dsp_by_target.rename(columns={"spearman_r": "dsp_spearman", "r2": "dsp_r2", "rmse": "dsp_rmse"})
    metrics = metrics.merge(dsp_by_target, on="target", how="left")
    metrics["spearman_delta_vs_dsp"] = metrics["spearman_r"] - metrics["dsp_spearman"]
    metrics = metrics.sort_values(["target", "spearman_r"], ascending=[True, False]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    best_queryable = (
        metrics.loc[metrics["queryable_label"].eq("queryable")]
        .sort_values(["target", "spearman_r"], ascending=[True, False])
        .groupby("target")
        .head(1)
    )
    best_observed = (
        metrics.loc[metrics["queryable_label"].eq("observed upper bound")]
        .sort_values(["target", "spearman_r"], ascending=[True, False])
        .groupby("target")
        .head(1)
    )
    summary = {
        "targets": targets,
        "component_grid": component_grid,
        "feature_modes": feature_modes,
        "methods": methods,
        "queryable_options": queryable_options,
        "rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "best_queryable_mean_spearman_delta_vs_dsp": float(best_queryable["spearman_delta_vs_dsp"].mean()),
        "best_queryable_max_spearman_delta_vs_dsp": float(best_queryable["spearman_delta_vs_dsp"].max()),
        "best_observed_mean_spearman_delta_vs_dsp": float(best_observed["spearman_delta_vs_dsp"].mean()),
        "best_observed_max_spearman_delta_vs_dsp": float(best_observed["spearman_delta_vs_dsp"].max()),
        "best_queryable": best_queryable[
            ["target", "model", "spearman_r", "spearman_delta_vs_dsp", "r2", "rmse"]
        ].to_dict(orient="records"),
        "best_observed_upper_bound": best_observed[
            ["target", "model", "spearman_r", "spearman_delta_vs_dsp", "r2", "rmse"]
        ].to_dict(orient="records"),
        "semantics": (
            "Observed upper bound rows use held-out checkpoint token factors and are not deployable for unseen "
            "candidate mixtures. Queryable rows predict token factors from mixture features within each fold."
        ),
    }
    write_outputs(args.output_dir, metrics, predictions, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
