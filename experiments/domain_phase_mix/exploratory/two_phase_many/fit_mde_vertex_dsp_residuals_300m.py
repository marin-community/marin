# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "plotly", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Probe whether queryable MDE vertex-expert features improve DSP fits.

The MDE vertex feature graph trains one cap-1 single-domain expert for each
300M qsplit240 domain, scores each expert on shared eval substrates, then
compacts the resulting losses into dense matrices.  This script turns those
expert losses into candidate-queryable features by applying a candidate's
phase weights to the vertex experts.

Two feature semantics are compared:

* ``weighted_loss``: expected expert loss, ``sum_i w_i ell_i``.
* ``logprob_mixture``: MDE-style conditional mixture,
  ``-log sum_i w_i exp(-ell_i)``.

Both are evaluated only as queryable features of the candidate mixture, never
as observed checkpoint losses for the held-out candidate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.special import logsumexp
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_mde_token_dsp_uncheatable_300m import (
    DEFAULT_METADATA_CSV,
    DEFAULT_RAW_CSV,
    LOG2E,
    RIDGE_ALPHAS,
    fit_dsp_oof,
    load_packet,
    mixture_feature_matrix,
    score_predictions,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = SCRIPT_DIR / "reference_outputs/mde_vertex_expert_dense_features_300m_20260531"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_vertex_dsp_residuals_300m_20260610"
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

LOWER_IS_BETTER_SUFFIXES = ("/bpb", "/loss", "/nll", "/perplexity", "/bits")
HIGHER_IS_BETTER_SUFFIXES = (
    "/acc",
    "/acc_norm",
    "/exact_match",
    "/pass_at_1",
    "/choice_prob",
    "/choice_prob_norm",
    "/choice_logprob",
    "/choice_logprob_norm",
    "/logprob",
)


@dataclass(frozen=True)
class VertexFeatureBlock:
    """One expert-loss matrix plus grouping metadata."""

    name: str
    matrix_file: str
    metadata_file: str
    group_column: str
    denominator_column: str | None
    scale_to_bpb: bool


FEATURE_BLOCKS = [
    VertexFeatureBlock(
        name="raw_token",
        matrix_file="raw_text_token_nll_matrix.npy",
        metadata_file="raw_text_token_metadata.parquet",
        group_column="dataset_name",
        denominator_column="token_bytes",
        scale_to_bpb=True,
    ),
    VertexFeatureBlock(
        name="raw_document",
        matrix_file="raw_text_document_nll_matrix.npy",
        metadata_file="raw_text_document_metadata.parquet",
        group_column="dataset_name",
        denominator_column="document_bytes",
        scale_to_bpb=True,
    ),
    VertexFeatureBlock(
        name="teacher_forced",
        matrix_file="teacher_forced_request_matrix.npy",
        metadata_file="teacher_forced_request_metadata.parquet",
        group_column="metric_prefix",
        denominator_column=None,
        scale_to_bpb=False,
    ),
    VertexFeatureBlock(
        name="mcq",
        matrix_file="mcq_choice_matrix.npy",
        metadata_file="mcq_choice_metadata.parquet",
        group_column="task_alias",
        denominator_column=None,
        scale_to_bpb=False,
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--block", choices=[block.name for block in FEATURE_BLOCKS], action="append", default=[])
    parser.add_argument(
        "--target-transform",
        choices=("auto", "none", "negate"),
        default="auto",
        help=(
            "Transform raw metric values into the lower-is-better objective expected by DSP. "
            "'auto' keeps BPB/loss/perplexity-like metrics and negates accuracy/probability/logprob-like metrics."
        ),
    )
    parser.add_argument("--dsp-variant", choices=sorted(dsp.VARIANTS), default="effective_exposure")
    parser.add_argument("--maxiter", type=int, default=72)
    parser.add_argument("--coarse-top-k", type=int, default=5)
    parser.add_argument("--basin-hopping-iters", type=int, default=3)
    parser.add_argument("--cv-seed", type=int, default=dsp.CV_SEED)
    parser.add_argument("--splits", type=int, default=dsp.N_SPLITS)
    return parser.parse_args()


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute ordinary R^2."""
    residual_ss = float(np.sum((actual - predicted) ** 2))
    total_ss = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - residual_ss / total_ss if total_ss > 0.0 else float("nan")


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def safe_label(value: str) -> str:
    """Make a target label safe for filenames."""
    return value.replace("/", "__").replace(":", "_")


def target_transform(metric: str, requested: str) -> str:
    """Return the concrete objective transform for one metric."""
    if requested in {"none", "negate"}:
        return requested
    if metric.endswith(LOWER_IS_BETTER_SUFFIXES):
        return "none"
    if metric.endswith(HIGHER_IS_BETTER_SUFFIXES):
        return "negate"
    return "none"


def transform_packet_objective(packet: dsp.PacketData, transform: str) -> dsp.PacketData:
    """Apply the objective transform expected by DSP."""
    if transform == "none":
        return packet
    if transform != "negate":
        raise ValueError(f"Unknown target transform: {transform}")
    frame = packet.frame.copy()
    frame["target"] = -pd.to_numeric(frame["target"], errors="raise")
    return replace(packet, frame=frame, y=-packet.y)


def load_expert_manifest(feature_dir: Path, domain_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Load cap-1 expert manifest and return row indices aligned to packet domains."""
    manifest = pd.read_csv(feature_dir / "expert_manifest.csv")
    cap1 = manifest.loc[~manifest["is_control"].astype(bool)].copy()
    domain_to_row = dict(zip(cap1["domain_name"].astype(str), cap1.index, strict=True))
    missing = [domain for domain in domain_names if domain not in domain_to_row]
    if missing:
        raise ValueError(f"MDE vertex manifest is missing packet domains: {missing[:10]}")
    return manifest, np.asarray([domain_to_row[domain] for domain in domain_names], dtype=np.int64)


def block_items_by_group(metadata: pd.DataFrame, group_column: str) -> list[tuple[str, np.ndarray]]:
    """Return stable item indices for each metadata group."""
    groups = []
    for group_name in sorted(metadata[group_column].astype(str).unique()):
        mask = metadata[group_column].astype(str).eq(group_name).to_numpy()
        groups.append((group_name, np.flatnonzero(mask)))
    return groups


def weighted_loss_group_features(
    expert_matrix: np.ndarray,
    candidate_weights: np.ndarray,
    groups: list[tuple[str, np.ndarray]],
    denominators: np.ndarray | None,
    *,
    scale_to_bpb: bool,
) -> np.ndarray:
    """Aggregate ``sum_i w_i ell_i`` features by metadata group."""
    expert_group_values = np.empty((expert_matrix.shape[0], len(groups)), dtype=np.float64)
    for group_idx, (_group_name, item_idx) in enumerate(groups):
        group_losses = expert_matrix[:, item_idx].astype(np.float64, copy=False)
        if denominators is None:
            expert_group_values[:, group_idx] = group_losses.mean(axis=1)
        else:
            denom = float(denominators[item_idx].sum())
            expert_group_values[:, group_idx] = group_losses.sum(axis=1) * (LOG2E if scale_to_bpb else 1.0) / denom
    return candidate_weights @ expert_group_values


def logprob_mixture_group_features(
    expert_matrix: np.ndarray,
    candidate_weights: np.ndarray,
    groups: list[tuple[str, np.ndarray]],
    denominators: np.ndarray | None,
    *,
    scale_to_bpb: bool,
) -> np.ndarray:
    """Aggregate ``-log sum_i w_i exp(-ell_i)`` features by metadata group."""
    out = np.empty((candidate_weights.shape[0], len(groups)), dtype=np.float64)
    clipped = np.clip(candidate_weights.astype(np.float64, copy=False), 1e-300, 1.0)
    log_weights = np.log(clipped)
    for group_idx, (_group_name, item_idx) in enumerate(groups):
        group_losses = expert_matrix[:, item_idx].astype(np.float64, copy=False)
        for row_idx, row_log_weights in enumerate(log_weights):
            mixed_nll = -logsumexp(row_log_weights[:, None] - group_losses, axis=0)
            if denominators is None:
                out[row_idx, group_idx] = float(mixed_nll.mean())
            else:
                denom = float(denominators[item_idx].sum())
                out[row_idx, group_idx] = float(mixed_nll.sum() * (LOG2E if scale_to_bpb else 1.0) / denom)
    return out


def build_vertex_features(
    *,
    feature_dir: Path,
    packet: dsp.PacketData,
    expert_rows: np.ndarray,
    enabled_blocks: set[str],
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    """Build candidate-queryable vertex-MDE features for all requested blocks."""
    feature_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    summary: dict[str, object] = {}
    phase_weights = {
        "phase0": packet.w[:, 0, :],
        "phase1": packet.w[:, 1, :],
        "phase_mean": 0.5 * (packet.w[:, 0, :] + packet.w[:, 1, :]),
    }
    for block in FEATURE_BLOCKS:
        if block.name not in enabled_blocks:
            continue
        print(f"Loading vertex feature block: {block.name}", flush=True)
        matrix = np.load(feature_dir / block.matrix_file, allow_pickle=False)
        matrix = matrix[expert_rows]
        metadata = pd.read_parquet(feature_dir / block.metadata_file)
        groups = block_items_by_group(metadata, block.group_column)
        denominators = None
        if block.denominator_column is not None:
            denominators = metadata[block.denominator_column].to_numpy(dtype=np.float64)
        summary[f"{block.name}_expert_shape"] = [int(matrix.shape[0]), int(matrix.shape[1])]
        summary[f"{block.name}_groups"] = [group_name for group_name, _idx in groups]
        for semantics in ("weighted_loss", "logprob_mixture"):
            for phase_name, weights in phase_weights.items():
                print(f"  aggregating {block.name}/{semantics}/{phase_name}", flush=True)
                if semantics == "weighted_loss":
                    values = weighted_loss_group_features(
                        matrix,
                        weights,
                        groups,
                        denominators,
                        scale_to_bpb=block.scale_to_bpb,
                    )
                else:
                    values = logprob_mixture_group_features(
                        matrix,
                        weights,
                        groups,
                        denominators,
                        scale_to_bpb=block.scale_to_bpb,
                    )
                feature_parts.append(values)
                feature_names.extend(f"{block.name}:{semantics}:{phase_name}:{group_name}" for group_name, _idx in groups)
    if not feature_parts:
        raise ValueError("No MDE vertex feature blocks selected")
    return np.hstack(feature_parts), feature_names, summary


def ridge_oof(features: np.ndarray, target: np.ndarray, *, splits: int, seed: int) -> np.ndarray:
    """Fit ridge OOF predictions with fold-local scaling."""
    oof = np.empty_like(target, dtype=np.float64)
    cv = KFold(n_splits=splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(features):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])
        model = RidgeCV(alphas=RIDGE_ALPHAS)
        model.fit(x_train, target[train_idx])
        oof[test_idx] = model.predict(x_test)
    return oof


def dsp_plus_feature_residual_oof(
    *,
    packet: dsp.PacketData,
    dsp_model: dsp.FittedDSPModel,
    features: np.ndarray,
    splits: int,
    seed: int,
) -> np.ndarray:
    """Fit DSP linear heads plus fold-local ridge residuals from vertex features."""
    oof = np.empty_like(packet.y, dtype=np.float64)
    cv = KFold(n_splits=splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(packet.w):
        fold_model = dsp.fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            dsp_model.variant,
            dsp_model.params,
        )
        base_train = dsp.predict(fold_model, packet.w[train_idx])
        base_test = dsp.predict(fold_model, packet.w[test_idx])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])
        residual_model = RidgeCV(alphas=RIDGE_ALPHAS)
        residual_model.fit(x_train, packet.y[train_idx] - base_train)
        oof[test_idx] = base_test + residual_model.predict(x_test)
    return oof


def plot_metric_bars(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Write metric comparison bars."""
    fig = px.bar(
        metrics,
        x="target",
        y="spearman_r",
        color="model",
        barmode="group",
        title="MDE vertex features versus DSP: OOF Spearman",
        hover_data=["r2", "rmse", "spearman_delta_vs_dsp"],
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(height=max(520, 60 * metrics["target"].nunique()), xaxis_tickangle=-30)
    fig.write_html(
        output_dir / "mde_vertex_dsp_spearman_bars.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def plot_actual_vs_pred(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Write actual-vs-predicted scatter."""
    long = predictions.melt(
        id_vars=["target", "run_name", "actual"],
        var_name="model",
        value_name="predicted",
        value_vars=[column for column in predictions.columns if column.startswith("pred__")],
    )
    long["model"] = long["model"].str.removeprefix("pred__")
    fig = px.scatter(
        long,
        x="actual",
        y="predicted",
        color="model",
        facet_col="target",
        facet_col_wrap=2,
        hover_data=["run_name"],
        title="MDE vertex features versus DSP: OOF actual vs predicted",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(height=max(620, 320 * int(np.ceil(long["target"].nunique() / 2))))
    fig.write_html(
        output_dir / "mde_vertex_dsp_actual_vs_pred.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def run_target(
    *,
    target: str,
    args: argparse.Namespace,
    enabled_blocks: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run all models for one target."""
    print(f"Fitting target: {target}", flush=True)
    raw, packet = load_packet(args.raw_csv, args.metadata_csv, target)
    transform = target_transform(target, args.target_transform)
    packet = transform_packet_objective(packet, transform)
    _manifest, expert_rows = load_expert_manifest(args.feature_dir, packet.domain_names)
    vertex_features, feature_names, feature_summary = build_vertex_features(
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
    dsp_oof = fit_dsp_oof(packet, dsp_model, args.splits, args.cv_seed)
    phase_features = mixture_feature_matrix(packet, "phase_log_exposure", dsp_model)
    dsp_features = mixture_feature_matrix(packet, "dsp_features", dsp_model)
    model_predictions = {
        "phase_log_exposure_ridge": ridge_oof(phase_features, packet.y, splits=args.splits, seed=args.cv_seed),
        "dsp_features_ridge": ridge_oof(dsp_features, packet.y, splits=args.splits, seed=args.cv_seed),
        "vertex_mde_ridge": ridge_oof(vertex_features, packet.y, splits=args.splits, seed=args.cv_seed),
        "phase_plus_vertex_mde_ridge": ridge_oof(
            np.hstack([phase_features, vertex_features]),
            packet.y,
            splits=args.splits,
            seed=args.cv_seed,
        ),
        "dsp": dsp_oof,
        "dsp_plus_vertex_mde_residual": dsp_plus_feature_residual_oof(
            packet=packet,
            dsp_model=dsp_model,
            features=vertex_features,
            splits=args.splits,
            seed=args.cv_seed,
        ),
        "dsp_plus_dspfeat_vertex_mde_residual": dsp_plus_feature_residual_oof(
            packet=packet,
            dsp_model=dsp_model,
            features=np.hstack([dsp_features, vertex_features]),
            splits=args.splits,
            seed=args.cv_seed,
        ),
    }
    rows = []
    dsp_score = score_predictions(packet.y, dsp_oof)
    for model_name, prediction in model_predictions.items():
        scores = score_predictions(packet.y, prediction)
        rows.append(
            {
                "target": target,
                "model": model_name,
                **scores,
                "spearman_delta_vs_dsp": float(scores["spearman_r"] - dsp_score["spearman_r"]),
                "r2_delta_vs_dsp": float(scores["r2"] - dsp_score["r2"]),
                "rmse_delta_vs_dsp": float(scores["rmse"] - dsp_score["rmse"]),
            }
        )
    pred_frame = pd.DataFrame(
        {
            "target": target,
            "run_name": raw["run_name"].astype(str).tolist(),
            "actual": packet.y,
            **{f"pred__{name}": values for name, values in model_predictions.items()},
        }
    )
    target_summary = {
        "target": target,
        "target_transform": transform,
        "rows": int(len(packet.y)),
        "vertex_feature_count": int(vertex_features.shape[1]),
        "vertex_feature_names_sample": feature_names[:20],
        "dsp_variant": dsp_model.variant.name,
        "dsp_trace_rows": int(len(trace)),
        **feature_summary,
    }
    return pd.DataFrame.from_records(rows), pred_frame, target_summary


def main() -> None:
    """Run vertex-MDE residual probes."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = args.target or DEFAULT_TARGETS
    enabled_blocks = set(args.block or [block.name for block in FEATURE_BLOCKS])
    all_metrics = []
    all_predictions = []
    summaries = []
    for target in targets:
        metrics, predictions, target_summary = run_target(target=target, args=args, enabled_blocks=enabled_blocks)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        summaries.append(target_summary)
    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    metrics_frame.to_csv(args.output_dir / "mde_vertex_dsp_metrics.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "mde_vertex_dsp_oof_predictions.csv", index=False)
    best = metrics_frame.sort_values(["target", "spearman_r"], ascending=[True, False]).groupby("target").head(1)
    best.to_csv(args.output_dir / "mde_vertex_dsp_best_by_target.csv", index=False)
    plot_metric_bars(metrics_frame, args.output_dir)
    plot_actual_vs_pred(prediction_frame, args.output_dir)
    summary = {
        "feature_dir": str(args.feature_dir),
        "output_dir": str(args.output_dir),
        "targets": targets,
        "enabled_blocks": sorted(enabled_blocks),
        "metrics_csv": str(args.output_dir / "mde_vertex_dsp_metrics.csv"),
        "predictions_csv": str(args.output_dir / "mde_vertex_dsp_oof_predictions.csv"),
        "best_csv": str(args.output_dir / "mde_vertex_dsp_best_by_target.csv"),
        "best_mean_spearman": float(best["spearman_r"].mean()),
        "best_mean_delta_vs_dsp": float(best["spearman_delta_vs_dsp"].mean()),
        "target_summaries": summaries,
        "semantics": (
            "All MDE vertex features are candidate-queryable functions of mixture weights and "
            "single-domain expert losses. Held-out checkpoint losses are not used."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
