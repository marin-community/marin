# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "plotly", "scikit-learn", "scipy"]
# ///
"""Subsample the dense two-phase StarCoder landscape to debug DSP optima.

The dense StarCoder/Nemotron panel is only two-dimensional:
``phase_0_starcoder`` and ``phase_1_starcoder``.  This makes it a useful
controlled setting for separating three phenomena that are hard to disentangle
in the 39-domain 300M swarms:

1. in-support fit quality,
2. solved-optimum extrapolation away from sampled support, and
3. sample-size thresholds for recovering a known dense-surface optimum.

The script repeatedly fits DSP on subsamples of the 143-row dense panel, solves
the fitted optimum over the two phase simplices, and scores that optimum against
the full dense surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = (
    SCRIPT_DIR.parent / "paper_plots" / "data" / "two_phase_starcoder_combined_143_from_wandb.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_dsp_sample_complexity_20260702"
TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
DOMAIN_NAMES = ["nemotron_full", "starcoder"]
PHASES = ["phase_0", "phase_1"]
DEFAULT_VARIANTS = ["canonical", "effective_exposure"]
DEFAULT_SAMPLE_SIZES = [8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 143]
PLOTLY_CONFIG = {"toImageButtonOptions": {"scale": 4}}


@dataclass(frozen=True)
class DenseSurface:
    """Full dense StarCoder surface plus interpolation helper."""

    frame: pd.DataFrame
    packet: dsp.PacketData
    interpolator: LinearNDInterpolator
    best_index: int
    source_csv: Path

    @property
    def coords(self) -> np.ndarray:
        return self.frame[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)

    @property
    def values(self) -> np.ndarray:
        return self.frame[TARGET].to_numpy(dtype=float)

    @property
    def best_row(self) -> pd.Series:
        return self.frame.iloc[self.best_index]


def parse_sample_sizes(raw: str | None) -> list[int]:
    """Parse a comma-separated sample-size list."""
    if raw is None:
        return list(DEFAULT_SAMPLE_SIZES)
    sizes = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not sizes:
        raise ValueError("At least one sample size is required")
    return sizes


def phase_epoch_multipliers(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Infer per-phase epoch multipliers from nonzero rows."""
    multipliers: list[np.ndarray] = []
    for phase in PHASES:
        values = []
        for domain in DOMAIN_NAMES:
            weight_column = f"{phase}_{domain}"
            suffix = "nemotron" if domain == "nemotron_full" else "starcoder"
            epoch_column = f"{phase}_{suffix}_epochs"
            weights = frame[weight_column].to_numpy(dtype=float)
            epochs = frame[epoch_column].to_numpy(dtype=float)
            mask = weights > 1e-12
            if not np.any(mask):
                raise ValueError(f"Cannot infer epoch multiplier for {weight_column}")
            ratios = epochs[mask] / weights[mask]
            ratio = float(np.median(ratios))
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"Invalid epoch multiplier for {weight_column}: {ratio}")
            nonzero_ratio = ratios[np.isfinite(ratios) & (ratios > 0)]
            if len(nonzero_ratio) == 0:
                raise ValueError(f"No positive epoch ratios for {weight_column}")
            relative_mad = float(np.median(np.abs(nonzero_ratio - ratio)) / max(abs(ratio), 1e-12))
            if relative_mad > 1e-6:
                raise ValueError(
                    f"Epoch multiplier for {weight_column} is not stable: "
                    f"median={ratio}, relative_mad={relative_mad}"
                )
            values.append(ratio)
        multipliers.append(np.asarray(values, dtype=float))
    return multipliers[0], multipliers[1]


def weights_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Convert StarCoder frame columns into a ``(n, 2, 2)`` weight tensor."""
    weights = np.zeros((len(frame), 2, 2), dtype=float)
    for phase_idx, phase in enumerate(PHASES):
        for domain_idx, domain in enumerate(DOMAIN_NAMES):
            weights[:, phase_idx, domain_idx] = frame[f"{phase}_{domain}"].to_numpy(dtype=float)
    return dsp.normalize_weights(weights)


def packet_from_frame(frame: pd.DataFrame, c0: np.ndarray, c1: np.ndarray) -> dsp.PacketData:
    """Build a DSP packet for the two-domain StarCoder panel."""
    if "run_label" not in frame.columns:
        frame = frame.copy()
        frame["run_label"] = frame["run_id"].astype(str)
    return dsp.PacketData(
        frame=frame.reset_index(drop=True),
        name_col="run_label",
        y=frame[TARGET].to_numpy(dtype=float),
        w=weights_from_frame(frame),
        m=2,
        c0=np.asarray(c0, dtype=float),
        c1=np.asarray(c1, dtype=float),
        domain_names=list(DOMAIN_NAMES),
    )


def load_dense_surface(path: Path) -> DenseSurface:
    """Load the completed dense StarCoder surface."""
    frame = pd.read_csv(path)
    frame = frame.loc[frame["status"].eq("completed") & frame[TARGET].notna()].copy()
    frame = frame.sort_values(["phase_0_starcoder", "phase_1_starcoder", "run_id"]).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError(f"Expected dense StarCoder rows in {path}, found {len(frame)}")
    c0, c1 = phase_epoch_multipliers(frame)
    packet = packet_from_frame(frame, c0, c1)
    coords = frame[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    interpolator = LinearNDInterpolator(coords, frame[TARGET].to_numpy(dtype=float), fill_value=np.nan)
    best_index = int(frame[TARGET].to_numpy(dtype=float).argmin())
    return DenseSurface(frame=frame, packet=packet, interpolator=interpolator, best_index=best_index, source_csv=path)


def farthest_point_indices(coords: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Select a deterministic space-filling subset by farthest-point sampling."""
    rng = np.random.default_rng(seed)
    corners = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    distances_to_corners = cdist(corners, coords)
    selected = list(dict.fromkeys(int(idx) for idx in distances_to_corners.argmin(axis=1)))
    if len(selected) > sample_size:
        return np.asarray(selected[:sample_size], dtype=int)
    if not selected:
        selected.append(int(rng.integers(0, len(coords))))
    min_distance = cdist(coords, coords[selected]).min(axis=1)
    while len(selected) < sample_size:
        min_distance[selected] = -1.0
        next_idx = int(np.argmax(min_distance))
        selected.append(next_idx)
        min_distance = np.minimum(min_distance, np.linalg.norm(coords - coords[next_idx], axis=1))
    return np.asarray(selected, dtype=int)


def sample_indices(coords: np.ndarray, sample_size: int, method: str, seed: int) -> np.ndarray:
    """Return subsample indices for a sampling method."""
    if sample_size >= len(coords):
        return np.arange(len(coords), dtype=int)
    if method == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(len(coords), size=sample_size, replace=False))
    if method == "space_filling":
        return np.sort(farthest_point_indices(coords, sample_size, seed))
    raise ValueError(f"Unknown sampling method: {method}")


def train_hull_contains(train_coords: np.ndarray, point: np.ndarray) -> bool | None:
    """Return whether point is inside the training convex hull, if defined."""
    if len(train_coords) < 3:
        return None
    try:
        hull = Delaunay(train_coords)
    except QhullError:
        return None
    return bool(hull.find_simplex(point[None, :])[0] >= 0)


def dense_truth_at(surface: DenseSurface, coord: np.ndarray) -> tuple[float, bool]:
    """Interpolate dense-surface truth at coord; fall back to nearest row."""
    interpolated = float(surface.interpolator(coord[None, :])[0])
    if np.isfinite(interpolated):
        return interpolated, True
    distances = np.linalg.norm(surface.coords - coord[None, :], axis=1)
    return float(surface.values[int(np.argmin(distances))]), False


def evaluate_fit(
    surface: DenseSurface,
    train_indices: np.ndarray,
    variant_name: str,
    *,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
    optimum_starts: int,
    seed: int,
) -> tuple[dict[str, object], dsp.FittedDSPModel | None]:
    """Fit one DSP model and return sample-complexity diagnostics."""
    train_frame = surface.frame.iloc[train_indices].reset_index(drop=True)
    train_packet = packet_from_frame(train_frame, surface.packet.c0, surface.packet.c1)
    variant = dsp.VARIANTS[variant_name]
    try:
        model, tuning = dsp.fit_variant(
            train_packet,
            variant,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        raw_result, raw_weights = dsp.optimize_raw(
            model,
            num_starts=optimum_starts,
            seed=seed,
            observed_start_weights=train_packet.w,
            max_observed_starts=min(len(train_packet.w), 32),
        )
    except Exception as exc:
        return (
            {
                "variant": variant_name,
                "fit_status": "failed",
                "failure": repr(exc),
                "n": len(train_indices),
            },
            None,
        )

    dense_predictions = dsp.predict(model, surface.packet.w)
    train_predictions = dsp.predict(model, train_packet.w)
    optimum_coord = np.asarray([raw_weights[0, 1], raw_weights[1, 1]], dtype=float)
    dense_distances = np.linalg.norm(surface.coords - optimum_coord[None, :], axis=1)
    nearest_idx = int(np.argmin(dense_distances))
    nearest_value = float(surface.values[nearest_idx])
    dense_value, used_interpolation = dense_truth_at(surface, optimum_coord)
    best_value = float(surface.values[surface.best_index])
    best_coord = surface.coords[surface.best_index]
    train_coords = surface.coords[train_indices]
    train_distances = np.linalg.norm(train_coords - optimum_coord[None, :], axis=1)
    train_best_value = float(surface.values[train_indices].min())

    row: dict[str, object] = {
        "variant": variant_name,
        "fit_status": "ok",
        "n": len(train_indices),
        "n_params": int(model.total_param_count),
        "n_over_params": float(len(train_indices) / model.total_param_count),
        "phase_0_starcoder_opt": float(optimum_coord[0]),
        "phase_1_starcoder_opt": float(optimum_coord[1]),
        "dense_best_phase_0_starcoder": float(best_coord[0]),
        "dense_best_phase_1_starcoder": float(best_coord[1]),
        "dense_best_value": best_value,
        "predicted_optimum_value": float(raw_result.fun),
        "nearest_dense_value": nearest_value,
        "interpolated_dense_value": dense_value,
        "used_interpolation": bool(used_interpolation),
        "nearest_dense_run_id": str(surface.frame.iloc[nearest_idx]["run_id"]),
        "nearest_dense_distance": float(dense_distances[nearest_idx]),
        "optimum_distance_to_dense_best": float(np.linalg.norm(optimum_coord - best_coord)),
        "selected_regret_nearest": float(nearest_value - best_value),
        "selected_regret_interpolated": float(dense_value - best_value),
        "optimism_nearest": float(nearest_value - float(raw_result.fun)),
        "optimism_interpolated": float(dense_value - float(raw_result.fun)),
        "train_nearest_distance": float(train_distances.min()),
        "train_hull_contains_optimum": train_hull_contains(train_coords, optimum_coord),
        "train_best_value": train_best_value,
        "train_best_regret": float(train_best_value - best_value),
        "train_rmse": float(np.sqrt(np.mean((train_predictions - train_packet.y) ** 2))),
        "train_spearman": float(spearmanr(train_packet.y, train_predictions).statistic),
        "dense_rmse": float(np.sqrt(np.mean((dense_predictions - surface.packet.y) ** 2))),
        "dense_spearman": float(spearmanr(surface.packet.y, dense_predictions).statistic),
        "active_benefit_coef_count": int(np.sum(model.benefit_coef > 1e-10)),
        "active_penalty_coef_count": int(np.sum(model.penalty_coef > 1e-10)),
        "gamma": float(model.params.get("gamma", np.nan)),
        "optimum_success": bool(raw_result.success),
        "optimum_message": str(raw_result.message),
        "fit_objective_best": float(tuning["objective"].min()) if "objective" in tuning else np.nan,
    }
    return row, model


def run_diagnostic(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all subsampling diagnostics."""
    surface = load_dense_surface(args.source_csv)
    methods = ["random", "space_filling"]
    sample_sizes = [size for size in parse_sample_sizes(args.sample_sizes) if size <= len(surface.frame)]
    records: list[dict[str, object]] = []
    optimum_records: list[dict[str, object]] = []
    for sample_size in sample_sizes:
        for method in methods:
            repeat_count = 1 if sample_size == len(surface.frame) else args.repeats
            if method == "space_filling":
                repeat_count = min(repeat_count, args.space_filling_repeats)
            for repeat in range(repeat_count):
                seed = int(args.seed + 10_000 * sample_size + 101 * repeat + (0 if method == "random" else 1_000_000))
                indices = sample_indices(surface.coords, sample_size, method, seed)
                for variant_name in args.variants:
                    row, _model = evaluate_fit(
                        surface,
                        indices,
                        variant_name,
                        maxiter=args.maxiter,
                        coarse_top_k=args.coarse_top_k,
                        basin_hopping_iters=args.basin_hopping_iters,
                        optimum_starts=args.optimum_starts,
                        seed=seed,
                    )
                    row.update(
                        {
                            "sampling": method,
                            "repeat": repeat,
                            "seed": seed,
                            "train_indices_json": json.dumps(indices.tolist()),
                        }
                    )
                    records.append(row)
                    if row.get("fit_status") == "ok":
                        optimum_records.append(
                            {
                                "variant": variant_name,
                                "sampling": method,
                                "repeat": repeat,
                                "n": sample_size,
                                "phase_0_starcoder": row["phase_0_starcoder_opt"],
                                "phase_1_starcoder": row["phase_1_starcoder_opt"],
                                "selected_regret_interpolated": row["selected_regret_interpolated"],
                                "train_nearest_distance": row["train_nearest_distance"],
                                "n_over_params": row["n_over_params"],
                            }
                        )
                    print(
                        f"n={sample_size:3d} sampling={method:13s} rep={repeat:02d} "
                        f"variant={variant_name:18s} status={row.get('fit_status')} "
                        f"regret={row.get('selected_regret_interpolated', np.nan)}",
                        flush=True,
                    )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(optimum_records)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level subsampling diagnostics."""
    ok = results.loc[results["fit_status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    grouped = ok.groupby(["variant", "sampling", "n", "n_params", "n_over_params"], as_index=False)
    return grouped.agg(
        runs=("selected_regret_interpolated", "size"),
        success_regret_le_0p01=("selected_regret_interpolated", lambda s: float(np.mean(s <= 0.01))),
        success_regret_le_0p02=("selected_regret_interpolated", lambda s: float(np.mean(s <= 0.02))),
        success_regret_le_0p05=("selected_regret_interpolated", lambda s: float(np.mean(s <= 0.05))),
        median_regret=("selected_regret_interpolated", "median"),
        p90_regret=("selected_regret_interpolated", lambda s: float(np.quantile(s, 0.9))),
        median_optimum_distance=("optimum_distance_to_dense_best", "median"),
        median_optimism=("optimism_interpolated", "median"),
        median_train_nearest_distance=("train_nearest_distance", "median"),
        inside_train_hull_rate=("train_hull_contains_optimum", lambda s: float(np.mean(s.fillna(False)))),
        median_dense_rmse=("dense_rmse", "median"),
        median_dense_spearman=("dense_spearman", "median"),
        median_train_rmse=("train_rmse", "median"),
    )


def critical_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    """Return simple threshold estimates for reliable solved optima."""
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for (variant, sampling), group in summary.groupby(["variant", "sampling"]):
        ordered = group.sort_values("n")
        primary_ok = (ordered["success_regret_le_0p02"] >= 0.8) & (ordered["median_regret"] <= 0.01)
        tail_ok = ordered["p90_regret"] <= 0.02

        def first_crossing(mask: pd.Series) -> tuple[float, float]:
            crossing = ordered.loc[mask].head(1)
            if crossing.empty:
                return np.nan, np.nan
            row = crossing.iloc[0]
            return float(row["n"]), float(row["n_over_params"])

        def stable_crossing(mask: pd.Series) -> tuple[float, float]:
            values = mask.to_numpy(dtype=bool)
            for offset, ok in enumerate(values):
                if ok and bool(values[offset:].all()):
                    row = ordered.iloc[offset]
                    return float(row["n"]), float(row["n_over_params"])
            return np.nan, np.nan

        primary_n, primary_ratio = first_crossing(primary_ok)
        stable_n, stable_ratio = stable_crossing(primary_ok)
        p90_n, p90_ratio = stable_crossing(tail_ok)
        rows.append(
            {
                "variant": variant,
                "sampling": sampling,
                "first_crossing_found": np.isfinite(primary_n),
                "first_crossing_n": primary_n,
                "first_crossing_n_over_params": primary_ratio,
                "stable_threshold_found": np.isfinite(stable_n),
                "stable_threshold_n": stable_n,
                "stable_threshold_n_over_params": stable_ratio,
                "stable_p90_threshold_found": np.isfinite(p90_n),
                "stable_p90_threshold_n": p90_n,
                "stable_p90_threshold_n_over_params": p90_ratio,
                "first_crossing_criterion": "success_regret_le_0p02>=0.8 and median_regret<=0.01",
                "stable_p90_criterion": "p90_regret<=0.02 for this and all larger sampled n",
            }
        )
    return pd.DataFrame.from_records(rows)


def write_plots(
    output_dir: Path,
    surface: DenseSurface,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    optima: pd.DataFrame,
) -> None:
    """Write Plotly diagnostics."""
    ok = results.loc[results["fit_status"].eq("ok")].copy()
    if ok.empty:
        return

    fig = px.line(
        summary,
        x="n_over_params",
        y="median_regret",
        color="variant",
        line_dash="sampling",
        markers=True,
        hover_data=["n", "p90_regret", "success_regret_le_0p02", "median_train_nearest_distance"],
        title="DSP solved-optimum regret vs sample count on dense StarCoder surface",
        labels={
            "n_over_params": "subsample rows / fitted DSP parameter count",
            "median_regret": "median dense-surface regret of solved optimum (BPB)",
            "variant": "DSP variant",
            "sampling": "sampling",
        },
    )
    fig.update_layout(template="plotly_white")
    fig.write_html(output_dir / "sample_complexity_regret.html", include_plotlyjs="cdn", config=PLOTLY_CONFIG)

    support = px.scatter(
        ok,
        x="train_nearest_distance",
        y="selected_regret_interpolated",
        color="n",
        symbol="variant",
        facet_col="sampling",
        hover_data=[
            "variant",
            "repeat",
            "phase_0_starcoder_opt",
            "phase_1_starcoder_opt",
            "optimism_interpolated",
            "train_hull_contains_optimum",
        ],
        color_continuous_scale="RdYlGn_r",
        title="Solved-optimum regret increases when optimizer leaves sampled support",
        labels={
            "train_nearest_distance": "distance from solved optimum to nearest training row",
            "selected_regret_interpolated": "dense-surface regret of solved optimum (BPB)",
        },
    )
    support.update_layout(template="plotly_white")
    support.write_html(output_dir / "support_distance_vs_regret.html", include_plotlyjs="cdn", config=PLOTLY_CONFIG)

    if not optima.empty:
        selected_n = sorted({int(optima["n"].min()), 16, 32, 64, int(optima["n"].max())})
        selected_n = [n for n in selected_n if n in set(optima["n"])]
        plot_optima = optima.loc[optima["n"].isin(selected_n)].copy()
        landscape = go.Figure()
        landscape.add_trace(
            go.Scatter(
                x=surface.frame["phase_0_starcoder"],
                y=surface.frame["phase_1_starcoder"],
                mode="markers",
                marker={
                    "color": surface.frame[TARGET],
                    "colorscale": "RdYlGn_r",
                    "showscale": True,
                    "colorbar": {"title": "BPB"},
                    "size": 7,
                    "line": {"color": "white", "width": 0.4},
                },
                name="dense observed rows",
                text=surface.frame["run_id"].astype(str),
            )
        )
        landscape.add_trace(
            go.Scatter(
                x=[surface.best_row["phase_0_starcoder"]],
                y=[surface.best_row["phase_1_starcoder"]],
                mode="markers+text",
                marker={"color": "#FFD700", "size": 16, "symbol": "star", "line": {"color": "black", "width": 1}},
                text=["dense best"],
                textposition="top center",
                name="dense best",
            )
        )
        for (variant, sampling, n), group in plot_optima.groupby(["variant", "sampling", "n"]):
            landscape.add_trace(
                go.Scatter(
                    x=group["phase_0_starcoder"],
                    y=group["phase_1_starcoder"],
                    mode="markers",
                    marker={"size": 9, "opacity": 0.75},
                    name=f"{variant} {sampling} n={n}",
                    hovertext=[
                        f"regret={regret:.4f}<br>nearest train dist={dist:.3f}"
                        for regret, dist in zip(
                            group["selected_regret_interpolated"], group["train_nearest_distance"], strict=True
                        )
                    ],
                )
            )
        landscape.update_layout(
            template="plotly_white",
            title="Solved DSP optima over the dense StarCoder two-phase landscape",
            xaxis_title="Phase 0 StarCoder weight",
            yaxis_title="Phase 1 StarCoder weight",
            width=1050,
            height=780,
        )
        landscape.write_html(output_dir / "solved_optima_on_dense_surface.html", include_plotlyjs="cdn", config=PLOTLY_CONFIG)


def write_summary_json(output_dir: Path, surface: DenseSurface, summary: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    """Write machine-readable summary."""
    payload = {
        "source_csv": str(surface.source_csv),
        "target": TARGET,
        "dense_row_count": int(len(surface.frame)),
        "epoch_multipliers": {
            "phase_0": dict(zip(DOMAIN_NAMES, surface.packet.c0.tolist(), strict=True)),
            "phase_1": dict(zip(DOMAIN_NAMES, surface.packet.c1.tolist(), strict=True)),
        },
        "dense_best": {
            "run_id": str(surface.best_row["run_id"]),
            "phase_0_starcoder": float(surface.best_row["phase_0_starcoder"]),
            "phase_1_starcoder": float(surface.best_row["phase_1_starcoder"]),
            "bpb": float(surface.best_row[TARGET]),
        },
        "thresholds": thresholds.to_dict(orient="records"),
        "best_summary_rows": (
            summary.sort_values(["variant", "sampling", "median_regret"])
            .groupby(["variant", "sampling"], as_index=False)
            .head(1)
            .to_dict(orient="records")
            if not summary.empty
            else []
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", default=None)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=sorted(dsp.VARIANTS))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--space-filling-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--optimum-starts", type=int, default=48)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    surface = load_dense_surface(args.source_csv)
    results, optima = run_diagnostic(args)
    summary = aggregate_results(results)
    thresholds = critical_thresholds(summary)

    results.to_csv(args.output_dir / "starcoder_dsp_sample_complexity_rows.csv", index=False)
    optima.to_csv(args.output_dir / "starcoder_dsp_solved_optima.csv", index=False)
    summary.to_csv(args.output_dir / "starcoder_dsp_sample_complexity_summary.csv", index=False)
    thresholds.to_csv(args.output_dir / "starcoder_dsp_sample_complexity_thresholds.csv", index=False)
    write_plots(args.output_dir, surface, results, summary, optima)
    write_summary_json(args.output_dir, surface, summary, thresholds)
    print(f"Wrote StarCoder DSP sample-complexity diagnostics to {args.output_dir}")
    if not thresholds.empty:
        print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
