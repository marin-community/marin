# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "plotly", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Fit queryable token-MDE residual models over DSP for uncheatable BPB.

This script consumes the compact token sketch produced by
``extract_mde_uncheatable_token_features_300m.py``.  It compares:

* DSP-only OOF predictions;
* queryable token-MDE kernel interpolation over train-fold checkpoint experts;
* DSP plus ridge residual correction from token-MDE features.

The key guardrail is that a held-out checkpoint's own token losses are never
used as predictors.  Token features for a held-out mixture are estimated from
training-fold checkpoint probabilities and the held-out mixture weights.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = str(SCRIPT_DIR / "reference_outputs/mde_uncheatable_token_features_300m_20260530")
DEFAULT_RAW_CSV = (
    SCRIPT_DIR / "collaborator_scaling_data_packet_20260430/data/raw_metric_matrix_300m/raw_metric_matrix_300m.csv"
)
DEFAULT_METADATA_CSV = SCRIPT_DIR / "two_phase_many_epoch_metadata.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_token_dsp_uncheatable_300m_20260530"
TARGET = "eval/uncheatable_eval/bpb"
LOG2E = float(np.log2(np.e))
RIDGE_ALPHAS = np.logspace(-4, 4, 25)
DENSE_MATRIX_FILE = "token_nll_matrix.npy"
DENSE_RUN_NAMES_FILE = "run_names.csv"
DENSE_TOKEN_METADATA_FILE = "token_metadata.parquet"


@dataclass(frozen=True)
class KernelSpec:
    """One token-MDE kernel interpolation configuration."""

    feature_mode: str
    k: int
    temperature: float

    @property
    def name(self) -> str:
        temp = str(self.temperature).replace(".", "p")
        return f"token_mde_{self.feature_mode}_k{self.k}_t{temp}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--dsp-variant", choices=sorted(dsp.VARIANTS), default="effective_exposure")
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=3)
    parser.add_argument("--cv-seed", type=int, default=dsp.CV_SEED)
    parser.add_argument("--splits", type=int, default=dsp.N_SPLITS)
    return parser.parse_args()


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute ordinary R^2."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return regression and ranking metrics."""
    residual = predicted - actual
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2_score(actual, predicted),
        "spearman_r": float(spearmanr(actual, predicted).statistic),
        "pearson_r": float(pearsonr(actual, predicted).statistic),
    }


def weights_to_packet(
    raw: pd.DataFrame,
    y_loss: np.ndarray,
    domains: list[str],
    w0: np.ndarray,
    w1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> dsp.PacketData:
    """Build a DSP packet from 300M mixture weights and one target vector."""
    frame = pd.DataFrame(
        {
            "run_name": raw["run_name"].astype(str),
            "run_id": raw["run_id"],
            "target": y_loss,
        }
    )
    weights = dsp.normalize_weights(np.stack([w0, w1], axis=1))
    return dsp.PacketData(
        frame=frame,
        name_col="run_name",
        y=np.asarray(y_loss, dtype=float),
        w=weights,
        m=len(domains),
        c0=np.asarray(c0, dtype=float),
        c1=np.asarray(c1, dtype=float),
        domain_names=list(domains),
    )


def load_packet(raw_csv: Path, metadata_csv: Path, target: str) -> tuple[pd.DataFrame, dsp.PacketData]:
    """Load the complete target matrix as a DSP packet."""
    raw = pd.read_csv(raw_csv, low_memory=False)
    if target not in raw.columns:
        raise ValueError(f"Target column {target!r} is missing from {raw_csv}")
    raw = raw.loc[raw["status"].eq("completed") & raw[target].notna()].reset_index(drop=True)
    domains = sorted(column.removeprefix("phase_0_") for column in raw.columns if column.startswith("phase_0_"))
    w0 = raw[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=np.float64)
    w1 = raw[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=np.float64)
    w0 = w0 / w0.sum(axis=1, keepdims=True)
    w1 = w1 / w1.sum(axis=1, keepdims=True)

    metadata = pd.read_csv(metadata_csv).set_index("domain_name")
    c0 = np.asarray([metadata.loc[domain, "phase_0_epoch_multiplier"] for domain in domains], dtype=np.float64)
    c1 = np.asarray([metadata.loc[domain, "phase_1_epoch_multiplier"] for domain in domains], dtype=np.float64)
    packet = weights_to_packet(raw, raw[target].to_numpy(dtype=np.float64), domains, w0, w1, c0, c1)
    return raw, packet


def join_path_or_uri(base: str, filename: str) -> str:
    """Join a local directory or URI with a filename."""
    if "://" in base:
        return f"{base.rstrip('/')}/{filename}"
    return str(Path(base) / filename)


def path_exists(path: str) -> bool:
    """Return whether a local path or fsspec URI exists."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if len(paths) != 1:
        raise ValueError(f"Expected one path, got {paths}")
    return bool(fs.exists(paths[0]))


def load_token_matrix(
    feature_dir: str, run_names: Iterable[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load token NLL matrix aligned to run order."""
    run_order = list(run_names)
    dense_matrix_path = join_path_or_uri(feature_dir, DENSE_MATRIX_FILE)
    dense_run_names_path = join_path_or_uri(feature_dir, DENSE_RUN_NAMES_FILE)
    dense_token_metadata_path = join_path_or_uri(feature_dir, DENSE_TOKEN_METADATA_FILE)
    selected_path = join_path_or_uri(feature_dir, "selected_tokens.parquet")
    token_path = join_path_or_uri(feature_dir, "uncheatable_token_nll_long.parquet")
    summary_path = join_path_or_uri(feature_dir, "summary.json")
    shard_manifest_path = join_path_or_uri(feature_dir, "shard_manifest.csv")
    if path_exists(summary_path):
        with fsspec.open(summary_path, "rt") as handle:
            summary = json.load(handle)
        dense_matrix_path = str(summary.get("matrix_path", dense_matrix_path))
        dense_run_names_path = str(summary.get("run_names_path", dense_run_names_path))
        dense_token_metadata_path = str(summary.get("token_metadata_path", dense_token_metadata_path))
        selected_path = str(summary.get("selected_tokens_path", selected_path))
        shard_manifest_path = str(summary.get("shard_manifest_path", shard_manifest_path))
    if path_exists(dense_matrix_path) and path_exists(dense_run_names_path) and path_exists(dense_token_metadata_path):
        token_metadata = pd.read_parquet(dense_token_metadata_path)
        token_keys = token_metadata["token_key"].astype(str).tolist()
        token_bytes = token_metadata["token_bytes"].to_numpy(dtype=np.float64)
        token_datasets = token_metadata["dataset_name"].astype(str).to_numpy()
        stored_run_names = pd.read_csv(dense_run_names_path)["run_name"].astype(str).tolist()
        run_index = {run_name: idx for idx, run_name in enumerate(stored_run_names)}
        missing_runs = [run_name for run_name in run_order if run_name not in run_index]
        if missing_runs:
            raise ValueError(f"Dense token matrix is missing run names: {missing_runs[:10]}")
        with fsspec.open(dense_matrix_path, "rb") as handle:
            dense_matrix = np.load(handle, allow_pickle=False)
        if dense_matrix.shape != (len(stored_run_names), len(token_keys)):
            raise ValueError(
                f"Dense token matrix has shape {dense_matrix.shape}; "
                f"expected {(len(stored_run_names), len(token_keys))}"
            )
        row_indices = np.asarray([run_index[run_name] for run_name in run_order], dtype=np.int64)
        matrix = dense_matrix[row_indices].astype(np.float64, copy=False)
        return matrix, token_bytes, token_datasets, token_keys

    has_monolithic = path_exists(selected_path) and path_exists(token_path)
    has_shards = path_exists(selected_path) and path_exists(shard_manifest_path)
    if not has_monolithic and not has_shards:
        raise FileNotFoundError(
            f"Missing compact token features in {feature_dir}. "
            "Run extract_mde_uncheatable_token_features_300m.py or the sharded launcher first."
        )
    selected = pd.read_parquet(selected_path)
    token_keys = selected["token_key"].astype(str).tolist()
    token_bytes = selected["token_bytes"].to_numpy(dtype=np.float64)
    token_datasets = selected["dataset_name"].astype(str).to_numpy()

    if has_monolithic:
        token_long = pd.read_parquet(token_path)
        matrix = token_long.pivot(index="run_name", columns="token_key", values="token_nll")
        matrix = matrix.reindex(index=run_order, columns=token_keys)
        if matrix.isna().any().any():
            missing = int(matrix.isna().sum().sum())
            raise ValueError(f"Token matrix has {missing} missing run/token cells")
        return matrix.to_numpy(dtype=np.float64), token_bytes, token_datasets, token_keys

    if has_shards:
        manifest = pd.read_csv(shard_manifest_path).set_index("run_name")
        matrix = np.empty((len(run_order), len(token_keys)), dtype=np.float64)
        for run_idx, run_name in enumerate(run_order, start=1):
            if run_name not in manifest.index:
                raise ValueError(f"Missing token shard for run_name={run_name}")
            path = str(manifest.loc[run_name, "token_path"])
            if run_idx == 1 or run_idx % 25 == 0 or run_idx == len(matrix):
                print(f"loading token shard {run_idx}/{len(matrix)}: {path}", flush=True)
            shard = pd.read_parquet(path, columns=["token_key", "token_nll"])
            values = shard.set_index("token_key").reindex(token_keys)["token_nll"].to_numpy(dtype=np.float64)
            if np.isnan(values).any():
                missing = int(np.isnan(values).sum())
                raise ValueError(f"Token shard for run_name={run_name} has {missing} missing selected tokens")
            matrix[run_idx - 1] = values
        return matrix, token_bytes, token_datasets, token_keys

    raise FileNotFoundError(
        f"Missing compact token features in {feature_dir}. "
        "Run extract_mde_uncheatable_token_features_300m.py or the sharded launcher first."
    )


def aggregate_token_bpb(token_nll: np.ndarray, token_bytes: np.ndarray) -> np.ndarray:
    """Aggregate token NLL rows into BPB."""
    return np.asarray(token_nll @ np.ones(token_nll.shape[1]) * LOG2E / token_bytes.sum(), dtype=np.float64)


def mixture_feature_matrix(
    packet: dsp.PacketData,
    mode: str,
    model: dsp.FittedDSPModel | None = None,
) -> np.ndarray:
    """Build queryable mixture features for kernel interpolation."""
    phase = packet.w.reshape(len(packet.y), -1)
    e0 = packet.w[:, 0, :] * packet.c0[None, :]
    e1 = packet.w[:, 1, :] * packet.c1[None, :]
    log_exposure = np.concatenate([np.log1p(e0), np.log1p(e1)], axis=1)
    if mode == "phase":
        return phase
    if mode == "log_exposure":
        return log_exposure
    if mode == "phase_log_exposure":
        return np.concatenate([phase, log_exposure], axis=1)
    if mode == "dsp_features":
        if model is None:
            raise ValueError("dsp_features mode requires a fitted DSP model")
        signal, penalty = dsp.features(packet.w, packet.c0, packet.c1, model.variant, model.params)
        return np.concatenate([signal, penalty], axis=1)
    raise ValueError(f"Unknown feature mode: {mode}")


def dataset_feature_names(token_datasets: np.ndarray) -> list[str]:
    """Return stable per-dataset MDE feature names."""
    return [f"mde_bpb::{name}" for name in sorted(set(token_datasets.tolist()))]


def token_mde_features(
    *,
    query_features: np.ndarray,
    anchor_features: np.ndarray,
    anchor_token_nll: np.ndarray,
    token_bytes: np.ndarray,
    token_datasets: np.ndarray,
    spec: KernelSpec,
    exclude_anchor_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Compute macro and per-dataset token-MDE losses for query rows."""
    dataset_names = sorted(set(token_datasets.tolist()))
    masks = [token_datasets == name for name in dataset_names]
    outputs = np.zeros((len(query_features), 1 + len(dataset_names)), dtype=np.float64)
    if len(anchor_features) < 2:
        raise ValueError("Need at least two anchors for token-MDE interpolation")

    for query_idx, query in enumerate(query_features):
        diff = anchor_features - query[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        if exclude_anchor_positions is not None:
            excluded = int(exclude_anchor_positions[query_idx])
            if excluded >= 0:
                dist2[excluded] = np.inf
        neighbor_count = min(spec.k, np.isfinite(dist2).sum())
        if neighbor_count <= 0:
            raise ValueError("No finite token-MDE neighbors available")
        neighbor_idx = np.argpartition(dist2, neighbor_count - 1)[:neighbor_count]
        neighbor_dist2 = dist2[neighbor_idx]
        finite_dist = np.sqrt(np.maximum(neighbor_dist2[np.isfinite(neighbor_dist2)], 0.0))
        bandwidth = max(float(np.max(finite_dist)), 1e-8) * spec.temperature
        logits = -neighbor_dist2 / (2.0 * bandwidth * bandwidth)
        logits = logits - logsumexp(logits)
        token_logprob = logsumexp(logits[:, None] - anchor_token_nll[neighbor_idx], axis=0)
        token_nll = -token_logprob
        outputs[query_idx, 0] = float(token_nll.sum() * LOG2E / token_bytes.sum())
        for col_idx, mask in enumerate(masks, start=1):
            outputs[query_idx, col_idx] = float(token_nll[mask].sum() * LOG2E / token_bytes[mask].sum())
    return outputs


def fit_dsp_oof(packet: dsp.PacketData, model: dsp.FittedDSPModel, splits: int, cv_seed: int) -> np.ndarray:
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


def run_kernel_spec(
    *,
    packet: dsp.PacketData,
    token_nll: np.ndarray,
    token_bytes: np.ndarray,
    token_datasets: np.ndarray,
    dsp_model: dsp.FittedDSPModel,
    spec: KernelSpec,
    splits: int,
    cv_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OOF standalone token-MDE and DSP+token-MDE residual predictions."""
    base_features = mixture_feature_matrix(packet, spec.feature_mode, dsp_model)
    cv = KFold(n_splits=splits, shuffle=True, random_state=cv_seed)
    mde_oof = np.zeros_like(packet.y)
    stacked_oof = np.zeros_like(packet.y)

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

        scaler = StandardScaler()
        train_features = scaler.fit_transform(base_features[train_idx])
        test_features = scaler.transform(base_features[test_idx])
        train_token_nll = token_nll[train_idx]

        train_query_features = train_features
        exclude_positions = np.arange(len(train_idx), dtype=int)
        train_mde = token_mde_features(
            query_features=train_query_features,
            anchor_features=train_features,
            anchor_token_nll=train_token_nll,
            token_bytes=token_bytes,
            token_datasets=token_datasets,
            spec=spec,
            exclude_anchor_positions=exclude_positions,
        )
        test_mde = token_mde_features(
            query_features=test_features,
            anchor_features=train_features,
            anchor_token_nll=train_token_nll,
            token_bytes=token_bytes,
            token_datasets=token_datasets,
            spec=spec,
            exclude_anchor_positions=None,
        )
        mde_oof[test_idx] = test_mde[:, 0]

        residual_model = RidgeCV(alphas=RIDGE_ALPHAS)
        residual_features_train = np.column_stack([train_mde, train_mde[:, 0] - base_train])
        residual_features_test = np.column_stack([test_mde, test_mde[:, 0] - base_test])
        residual_model.fit(residual_features_train, packet.y[train_idx] - base_train)
        stacked_oof[test_idx] = base_test + residual_model.predict(residual_features_test)

    return mde_oof, stacked_oof


def output_target(output_dir: str, filename: str) -> str:
    """Return a local or URI output target."""
    return join_path_or_uri(str(output_dir), filename)


def prepare_output_dir(output_dir: str) -> None:
    """Create local output directories; object-store prefixes are implicit."""
    if "://" in output_dir:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def write_text(output_dir: str, filename: str, text: str) -> None:
    """Write text to a local path or fsspec URI."""
    with fsspec.open(output_target(output_dir, filename), "wt") as handle:
        handle.write(text)


def write_outputs(
    *,
    output_dir: str,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    """Write tabular and HTML diagnostics."""
    prepare_output_dir(output_dir)
    metrics.to_csv(output_target(output_dir, "token_mde_dsp_metrics.csv"), index=False)
    predictions.to_csv(output_target(output_dir, "token_mde_dsp_oof_predictions.csv"), index=False)
    write_text(output_dir, "summary.json", json.dumps(summary, indent=2, sort_keys=True))

    metric_plot = metrics.melt(
        id_vars=["model"],
        value_vars=["spearman_r", "pearson_r", "r2"],
        var_name="score",
        value_name="value",
    )
    fig = px.bar(
        metric_plot,
        x="model",
        y="value",
        color="score",
        barmode="group",
        title="Queryable token-MDE variants versus DSP on uncheatable BPB",
    )
    fig.update_layout(width=1500, height=700, xaxis_tickangle=-25)
    write_text(
        output_dir,
        "token_mde_dsp_metric_bars.html",
        fig.to_html(
            include_plotlyjs="cdn",
            config={"toImageButtonOptions": {"scale": 4}},
        ),
    )

    best = str(metrics.sort_values("spearman_r", ascending=False).iloc[0]["model"])
    scatter = predictions[["run_name", "actual_bpb", "dsp_oof", best]].melt(
        id_vars=["run_name", "actual_bpb"],
        value_vars=["dsp_oof", best],
        var_name="model",
        value_name="predicted_bpb",
    )
    fig = px.scatter(
        scatter,
        x="actual_bpb",
        y="predicted_bpb",
        color="model",
        hover_name="run_name",
        title=f"OOF predictions: DSP versus {best}",
    )
    write_text(
        output_dir,
        "token_mde_dsp_actual_vs_pred.html",
        fig.to_html(
            include_plotlyjs="cdn",
            config={"toImageButtonOptions": {"scale": 4}},
        ),
    )


def main() -> None:
    """Run the model sweep."""
    args = parse_args()
    raw, packet = load_packet(args.raw_csv, args.metadata_csv, args.target)
    token_nll, token_bytes, token_datasets, token_keys = load_token_matrix(args.feature_dir, raw["run_name"].astype(str))

    variant = dsp.VARIANTS[args.dsp_variant]
    dsp_model, _tuning = dsp.fit_variant(
        packet,
        variant,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )
    dsp_oof = fit_dsp_oof(packet, dsp_model, args.splits, args.cv_seed)

    observed_token_bpb = np.asarray(token_nll.sum(axis=1) * LOG2E / token_bytes.sum(), dtype=np.float64)
    predictions = pd.DataFrame(
        {
            "run_name": raw["run_name"].astype(str),
            "actual_bpb": packet.y,
            "dsp_oof": dsp_oof,
            "observed_token_sketch_bpb_leaky": observed_token_bpb,
        }
    )
    rows = [
        {"model": "dsp_oof", **score_predictions(packet.y, dsp_oof)},
        {
            "model": "observed_token_sketch_bpb_leaky",
            **score_predictions(packet.y, observed_token_bpb),
        },
    ]

    specs = [
        KernelSpec(feature_mode=feature_mode, k=k, temperature=temp)
        for feature_mode in ("phase", "log_exposure", "phase_log_exposure", "dsp_features")
        for k in (5, 10, 20, 40)
        for temp in (0.5, 1.0, 2.0)
    ]
    for spec in specs:
        print(f"fitting {spec.name}", flush=True)
        mde_oof, stacked_oof = run_kernel_spec(
            packet=packet,
            token_nll=token_nll,
            token_bytes=token_bytes,
            token_datasets=token_datasets,
            dsp_model=dsp_model,
            spec=spec,
            splits=args.splits,
            cv_seed=args.cv_seed,
        )
        predictions[spec.name] = mde_oof
        stacked_name = f"dsp_plus_{spec.name}_residual"
        predictions[stacked_name] = stacked_oof
        rows.append({"model": spec.name, **score_predictions(packet.y, mde_oof)})
        rows.append({"model": stacked_name, **score_predictions(packet.y, stacked_oof)})

    metrics = pd.DataFrame.from_records(rows).sort_values("spearman_r", ascending=False).reset_index(drop=True)
    leaky_name = "observed_token_sketch_bpb_leaky"
    best = metrics.iloc[0].to_dict()
    queryable_metrics = metrics.loc[~metrics["model"].eq(leaky_name)]
    best_queryable = queryable_metrics.iloc[0].to_dict()
    standalone_mde = metrics.loc[metrics["model"].str.startswith("token_mde_")]
    best_standalone_mde = standalone_mde.iloc[0].to_dict()
    dsp_row = metrics.loc[metrics["model"].eq("dsp_oof")].iloc[0].to_dict()
    leaky_row = metrics.loc[metrics["model"].eq(leaky_name)].iloc[0].to_dict()
    summary = {
        "target": args.target,
        "rows": len(packet.y),
        "token_count": len(token_keys),
        "token_datasets": {name: int(np.sum(token_datasets == name)) for name in sorted(set(token_datasets.tolist()))},
        "dsp_variant": args.dsp_variant,
        "dsp_oof_spearman": float(dsp_row["spearman_r"]),
        "dsp_oof_rmse": float(dsp_row["rmse"]),
        "leaky_token_sketch_spearman": float(leaky_row["spearman_r"]),
        "leaky_token_sketch_rmse": float(leaky_row["rmse"]),
        "best_overall_model": str(best["model"]),
        "best_overall_spearman": float(best["spearman_r"]),
        "best_overall_rmse": float(best["rmse"]),
        "best_overall_spearman_delta_vs_dsp": float(best["spearman_r"] - dsp_row["spearman_r"]),
        "best_queryable_model": str(best_queryable["model"]),
        "best_queryable_spearman": float(best_queryable["spearman_r"]),
        "best_queryable_rmse": float(best_queryable["rmse"]),
        "best_queryable_spearman_delta_vs_dsp": float(best_queryable["spearman_r"] - dsp_row["spearman_r"]),
        "best_standalone_mde_model": str(best_standalone_mde["model"]),
        "best_standalone_mde_spearman": float(best_standalone_mde["spearman_r"]),
        "best_standalone_mde_spearman_delta_vs_dsp": float(
            best_standalone_mde["spearman_r"] - dsp_row["spearman_r"]
        ),
        "semantics": (
            "The observed_token_sketch_bpb_leaky row directly aggregates each checkpoint's own token losses "
            "and is included only as a token-sketch fidelity check. Queryable token-MDE rows use train-fold "
            "checkpoint probabilities and held-out mixture weights, not held-out token losses."
        ),
    }
    write_outputs(output_dir=args.output_dir, metrics=metrics, predictions=predictions, summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
