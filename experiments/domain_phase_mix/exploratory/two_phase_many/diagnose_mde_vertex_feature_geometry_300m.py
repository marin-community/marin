# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "plotly", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Diagnose queryable MDE vertex-feature geometry on 300M swarm targets.

This script is intentionally read-only.  It rebuilds candidate-queryable MDE
vertex features for one target row set, then checks whether those features are
numerically collapsed, largely spanned by simpler mixture features, or prone to
ridge overfit under shuffled targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_mde_token_dsp_uncheatable_300m import (
    DEFAULT_METADATA_CSV,
    DEFAULT_RAW_CSV,
    RIDGE_ALPHAS,
    load_packet,
    mixture_feature_matrix,
    score_predictions,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_mde_vertex_dsp_residuals_300m import (
    DEFAULT_FEATURE_DIR,
    FEATURE_BLOCKS,
    build_vertex_features,
    load_expert_manifest,
    ridge_oof,
    target_transform,
    transform_packet_objective,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_vertex_feature_geometry_300m_20260610"
DEFAULT_TARGET = "mcq_smooth/truthfulqa_mc1_0shot/choice_logprob"
RIDGE_NULL_MODELS = ("phase_log_exposure", "vertex_mde", "phase_plus_vertex_mde", "dsp_features")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block", choices=[block.name for block in FEATURE_BLOCKS], action="append", default=[])
    parser.add_argument("--target-transform", choices=("auto", "none", "negate"), default="auto")
    parser.add_argument("--dsp-variant", choices=sorted(dsp.VARIANTS), default="effective_exposure")
    parser.add_argument("--maxiter", type=int, default=36)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    parser.add_argument("--basin-hopping-iters", type=int, default=1)
    parser.add_argument("--splits", type=int, default=dsp.N_SPLITS)
    parser.add_argument("--cv-seed", type=int, default=dsp.CV_SEED)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--permutation-seed", type=int, default=20260610)
    parser.add_argument("--screen-top-k", type=int, action="append", default=[])
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def standardized_matrix(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized nonconstant columns and retained column mask."""
    features = np.asarray(features, dtype=np.float64)
    finite = np.isfinite(features).all(axis=0)
    std = np.nanstd(features[:, finite], axis=0, ddof=1)
    keep_finite_positions = std > 1e-12
    keep = np.zeros(features.shape[1], dtype=bool)
    keep[np.flatnonzero(finite)[keep_finite_positions]] = True
    if not keep.any():
        return np.empty((features.shape[0], 0), dtype=np.float64), keep
    return StandardScaler().fit_transform(features[:, keep]), keep


def spectral_summary(name: str, features: np.ndarray) -> dict[str, float | int | str]:
    """Summarize standardized feature rank and singular spectrum."""
    z, keep = standardized_matrix(features)
    if z.shape[1] == 0:
        return {
            "feature_set": name,
            "rows": int(features.shape[0]),
            "columns": int(features.shape[1]),
            "nonconstant_columns": 0,
            "rank": 0,
            "effective_rank": 0.0,
            "condition_number": float("nan"),
            "top1_variance_share": float("nan"),
            "top5_variance_share": float("nan"),
            "top10_variance_share": float("nan"),
            "top20_variance_share": float("nan"),
        }
    singular_values = np.linalg.svd(z, full_matrices=False, compute_uv=False)
    eig = singular_values**2
    total = float(eig.sum())
    tol = float(singular_values[0] * max(z.shape) * np.finfo(np.float64).eps)
    rank = int(np.sum(singular_values > tol))
    probs = eig[eig > 0] / total
    effective_rank = float(np.exp(-np.sum(probs * np.log(probs)))) if total > 0.0 else 0.0
    condition = float(singular_values[0] / singular_values[rank - 1]) if rank > 0 else float("nan")

    def top_share(k: int) -> float:
        return float(eig[: min(k, len(eig))].sum() / total) if total > 0.0 else float("nan")

    return {
        "feature_set": name,
        "rows": int(features.shape[0]),
        "columns": int(features.shape[1]),
        "nonconstant_columns": int(keep.sum()),
        "rank": rank,
        "effective_rank": effective_rank,
        "condition_number": condition,
        "top1_variance_share": top_share(1),
        "top5_variance_share": top_share(5),
        "top10_variance_share": top_share(10),
        "top20_variance_share": top_share(20),
    }


def projection_r2_summary(source_name: str, source: np.ndarray, target_name: str, target: np.ndarray) -> dict[str, float | str]:
    """Summarize how well source features span target features in sample."""
    x, _x_keep = standardized_matrix(source)
    y, _y_keep = standardized_matrix(target)
    if x.shape[1] == 0 or y.shape[1] == 0:
        return {"source": source_name, "target": target_name, "mean_r2": float("nan")}
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    residual = y - pred
    ss_res = np.sum(residual * residual, axis=0)
    ss_tot = np.sum(y * y, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-300)
    quantiles = np.quantile(r2, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "source": source_name,
        "target": target_name,
        "mean_r2": float(np.mean(r2)),
        "q00_r2": float(quantiles[0]),
        "q10_r2": float(quantiles[1]),
        "q25_r2": float(quantiles[2]),
        "q50_r2": float(quantiles[3]),
        "q75_r2": float(quantiles[4]),
        "q90_r2": float(quantiles[5]),
        "q100_r2": float(quantiles[6]),
    }


def ridge_scores(features_by_name: dict[str, np.ndarray], y: np.ndarray, *, splits: int, cv_seed: int) -> pd.DataFrame:
    """Fit ridge OOF models for actual target."""
    rows = []
    for model_name, features in features_by_name.items():
        pred = ridge_oof(features, y, splits=splits, seed=cv_seed)
        rows.append({"model": model_name, "permutation": "actual", **score_predictions(y, pred)})
    return pd.DataFrame.from_records(rows)


def feature_screen(feature_names: list[str], features: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Rank individual features by target correlation."""
    rows = []
    for idx, name in enumerate(feature_names):
        values = features[:, idx]
        rho = spearmanr(values, y).statistic
        rows.append(
            {
                "feature_index": idx,
                "feature": name,
                "spearman_r": float(0.0 if np.isnan(rho) else rho),
                "std": float(np.std(values, ddof=1)),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values("spearman_r", key=lambda column: column.abs(), ascending=False)


def fold_screened_ridge_scores(
    features: np.ndarray,
    y: np.ndarray,
    *,
    top_ks: list[int],
    splits: int,
    cv_seed: int,
) -> pd.DataFrame:
    """Fit ridge after selecting top correlated MDE features within each training fold."""
    rows = []
    cv = KFold(n_splits=splits, shuffle=True, random_state=cv_seed)
    for top_k in top_ks:
        oof = np.empty_like(y, dtype=np.float64)
        top_k = min(top_k, features.shape[1])
        for train_idx, test_idx in cv.split(features):
            y_train = y[train_idx]
            correlations = np.empty(features.shape[1], dtype=np.float64)
            for feature_idx in range(features.shape[1]):
                rho = spearmanr(features[train_idx, feature_idx], y_train).statistic
                correlations[feature_idx] = 0.0 if np.isnan(rho) else abs(float(rho))
            selected = np.argpartition(correlations, -top_k)[-top_k:]
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_idx][:, selected])
            x_test = scaler.transform(features[test_idx][:, selected])
            model = RidgeCV(alphas=RIDGE_ALPHAS)
            model.fit(x_train, y_train)
            oof[test_idx] = model.predict(x_test)
        rows.append({"model": f"vertex_mde_fold_screen_top{top_k}", **score_predictions(y, oof)})
    return pd.DataFrame.from_records(rows)


def permutation_null(
    features_by_name: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    splits: int,
    cv_seed: int,
    permutations: int,
    permutation_seed: int,
) -> pd.DataFrame:
    """Run target-shuffle null for ridge feature models."""
    rng = np.random.default_rng(permutation_seed)
    rows = []
    for perm_index in range(permutations):
        shuffled = rng.permutation(y)
        for model_name, features in features_by_name.items():
            pred = ridge_oof(features, shuffled, splits=splits, seed=cv_seed)
            rows.append(
                {
                    "model": model_name,
                    "permutation": f"shuffle_{perm_index:03d}",
                    **score_predictions(shuffled, pred),
                }
            )
        if perm_index + 1 == permutations or (perm_index + 1) % 10 == 0:
            print(f"completed permutation {perm_index + 1}/{permutations}", flush=True)
    return pd.DataFrame.from_records(rows)


def write_plots(
    actual: pd.DataFrame,
    null: pd.DataFrame,
    projection: pd.DataFrame,
    screen: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write compact diagnostic plots."""
    combined = pd.concat([actual, null], ignore_index=True)
    combined["is_actual"] = combined["permutation"].eq("actual")
    fig = px.box(
        combined,
        x="model",
        y="spearman_r",
        color="is_actual",
        points="all",
        title="Ridge OOF Spearman: actual target versus target-shuffle null",
        color_discrete_map={True: "#1a9850", False: "#d73027"},
    )
    fig.update_layout(xaxis_tickangle=-30)
    fig.write_html(
        output_dir / "ridge_actual_vs_shuffle_null.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )

    fig = px.bar(
        projection,
        x="source",
        y="q50_r2",
        color="target",
        barmode="group",
        hover_data=["mean_r2", "q10_r2", "q90_r2", "q100_r2"],
        title="In-sample feature-span diagnostic: median target-feature R2",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.write_html(
        output_dir / "feature_projection_r2.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )

    top_screen = screen.head(50).copy()
    fig = px.bar(
        top_screen,
        x="spearman_r",
        y="feature",
        orientation="h",
        title="Top individual MDE feature correlations",
        color="spearman_r",
        color_continuous_scale="RdYlGn_r",
        hover_data=["feature_index", "std"],
    )
    fig.update_layout(height=1100, yaxis={"categoryorder": "total ascending"})
    fig.write_html(
        output_dir / "individual_mde_feature_screen.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def main() -> None:
    """Run MDE vertex feature geometry diagnostics."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enabled_blocks = set(args.block or [block.name for block in FEATURE_BLOCKS])
    _raw, packet = load_packet(args.raw_csv, args.metadata_csv, args.target)
    transform = target_transform(args.target, args.target_transform)
    packet = transform_packet_objective(packet, transform)
    _manifest, expert_rows = load_expert_manifest(args.feature_dir, packet.domain_names)
    vertex_features, vertex_names, vertex_summary = build_vertex_features(
        feature_dir=args.feature_dir,
        packet=packet,
        expert_rows=expert_rows,
        enabled_blocks=enabled_blocks,
    )
    dsp_model, trace = dsp.fit_variant(
        packet,
        dsp.VARIANTS[args.dsp_variant],
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )
    phase = packet.w.reshape(len(packet.y), -1)
    log_exposure = mixture_feature_matrix(packet, "log_exposure", dsp_model)
    phase_log_exposure = mixture_feature_matrix(packet, "phase_log_exposure", dsp_model)
    dsp_features = mixture_feature_matrix(packet, "dsp_features", dsp_model)
    feature_sets = {
        "phase": phase,
        "log_exposure": log_exposure,
        "phase_log_exposure": phase_log_exposure,
        "dsp_features": dsp_features,
        "vertex_mde": vertex_features,
        "phase_plus_vertex_mde": np.hstack([phase_log_exposure, vertex_features]),
        "dsp_plus_vertex_mde": np.hstack([dsp_features, vertex_features]),
    }
    geometry = pd.DataFrame.from_records([spectral_summary(name, matrix) for name, matrix in feature_sets.items()])
    projection = pd.DataFrame.from_records(
        [
            projection_r2_summary("phase_log_exposure", phase_log_exposure, "vertex_mde", vertex_features),
            projection_r2_summary("dsp_features", dsp_features, "vertex_mde", vertex_features),
            projection_r2_summary("vertex_mde", vertex_features, "phase_log_exposure", phase_log_exposure),
            projection_r2_summary("vertex_mde", vertex_features, "dsp_features", dsp_features),
        ]
    )
    ridge_feature_sets = {
        "phase_log_exposure": phase_log_exposure,
        "vertex_mde": vertex_features,
        "phase_plus_vertex_mde": np.hstack([phase_log_exposure, vertex_features]),
        "dsp_features": dsp_features,
    }
    actual = ridge_scores(ridge_feature_sets, packet.y, splits=args.splits, cv_seed=args.cv_seed)
    screen = feature_screen(vertex_names, vertex_features, packet.y)
    top_ks = args.screen_top_k or [10, 20, 50, 100]
    screened_actual = fold_screened_ridge_scores(
        vertex_features,
        packet.y,
        top_ks=top_ks,
        splits=args.splits,
        cv_seed=args.cv_seed,
    )
    null = permutation_null(
        ridge_feature_sets,
        packet.y,
        splits=args.splits,
        cv_seed=args.cv_seed,
        permutations=args.permutations,
        permutation_seed=args.permutation_seed,
    )
    geometry.to_csv(args.output_dir / "feature_geometry.csv", index=False)
    projection.to_csv(args.output_dir / "feature_projection_r2.csv", index=False)
    actual.to_csv(args.output_dir / "ridge_actual_scores.csv", index=False)
    screen.to_csv(args.output_dir / "individual_mde_feature_screen.csv", index=False)
    screened_actual.to_csv(args.output_dir / "fold_screened_ridge_scores.csv", index=False)
    null.to_csv(args.output_dir / "ridge_shuffle_null_scores.csv", index=False)
    write_plots(actual, null, projection, screen, args.output_dir)
    summary = {
        "target": args.target,
        "target_transform": transform,
        "rows": int(len(packet.y)),
        "enabled_blocks": sorted(enabled_blocks),
        "vertex_feature_count": int(vertex_features.shape[1]),
        "vertex_feature_names_sample": vertex_names[:20],
        "dsp_variant": dsp_model.variant.name,
        "dsp_trace_rows": int(len(trace)),
        "permutations": int(args.permutations),
        "geometry_csv": str(args.output_dir / "feature_geometry.csv"),
        "projection_csv": str(args.output_dir / "feature_projection_r2.csv"),
        "actual_scores_csv": str(args.output_dir / "ridge_actual_scores.csv"),
        "individual_feature_screen_csv": str(args.output_dir / "individual_mde_feature_screen.csv"),
        "fold_screened_scores_csv": str(args.output_dir / "fold_screened_ridge_scores.csv"),
        "null_scores_csv": str(args.output_dir / "ridge_shuffle_null_scores.csv"),
        "vertex_summary": vertex_summary,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
