# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly"]
# ///
"""Analyze the 60M/1.2B single-phase exposure-average ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "reference_outputs" / "single_phase_exposure_average_60m_1p2b"
DEFAULT_MANIFEST = ARTIFACT_DIR / "single_phase_exposure_average_manifest.csv"
DEFAULT_SINGLE_FIT_DATASET = ARTIFACT_DIR / "fit_dataset.csv"
DEFAULT_SOURCE_FIT_SWARM = (
    BASE_DIR
    / "metric_registry"
    / "fit_datasets"
    / "eval_uncheatable_eval_bpb__60m_1p2b__signal__fit_swarm_60m_default.csv"
)
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "analysis"
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"


def _metric_column(df: pd.DataFrame) -> str:
    for column in ("objective_metric", OBJECTIVE_METRIC, "value"):
        if column in df.columns:
            return column
    raise ValueError(
        "Could not find a single-phase BPB column. Expected one of "
        "'objective_metric', 'eval/uncheatable_eval/bpb', or 'value'."
    )


def _domain_name(weight_column: str) -> str:
    return weight_column.split("_", 2)[2]


def _family_name(domain: str) -> str:
    name = domain.split("/", 1)[-1]
    for suffix in ("_high", "_low"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_high_quality(domain: str) -> bool:
    return domain.endswith("_high")


def _add_original_shift_features(merged: pd.DataFrame, source_fit_swarm: Path) -> pd.DataFrame:
    if not source_fit_swarm.exists():
        return merged

    source = pd.read_csv(source_fit_swarm)
    phase_0_columns = [column for column in source.columns if column.startswith("phase_0_")]
    phase_1_columns = [column for column in source.columns if column.startswith("phase_1_")]
    if not phase_0_columns or len(phase_0_columns) != len(phase_1_columns):
        return merged

    source_features = source[["run_name"]].copy()
    high_shift: list[float] = []
    family_tv: list[float] = []
    for _, row in source.iterrows():
        domains = [_domain_name(column) for column in phase_0_columns]
        phase_0 = np.array([row[f"phase_0_{domain}"] for domain in domains], dtype=float)
        phase_1 = np.array([row[f"phase_1_{domain}"] for domain in domains], dtype=float)
        high_mask = np.array([_is_high_quality(domain) for domain in domains], dtype=bool)
        high_shift.append(float(phase_1[high_mask].sum() - phase_0[high_mask].sum()))

        family_to_indices: dict[str, list[int]] = {}
        for index, domain in enumerate(domains):
            family_to_indices.setdefault(_family_name(domain), []).append(index)
        family_deltas = [
            abs(float(phase_1[indices].sum() - phase_0[indices].sum())) for indices in family_to_indices.values()
        ]
        family_tv.append(0.5 * float(np.sum(family_deltas)))

    source_features["original_high_quality_shift"] = high_shift
    source_features["original_family_tv"] = family_tv
    return merged.merge(
        source_features, left_on="source_run_name", right_on="run_name", how="left", suffixes=("", "_src")
    )


def _write_plot(fig, output_path: Path) -> None:
    fig.write_html(output_path.with_suffix(".html"))


def analyze(
    *,
    manifest_path: Path,
    single_fit_dataset_path: Path,
    source_fit_swarm_path: Path,
    output_dir: Path,
) -> None:
    """Generate paired single-vs-two phase summaries and HTML plots."""
    if not single_fit_dataset_path.exists():
        raise FileNotFoundError(f"Single-phase fit dataset not found yet: {single_fit_dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    single = pd.read_csv(single_fit_dataset_path)
    single_bpb_column = _metric_column(single)
    single = single[["run_name", single_bpb_column]].rename(columns={single_bpb_column: "single_phase_bpb"})

    merged = manifest.merge(single, on="run_name", how="left")
    if merged["single_phase_bpb"].isna().all():
        raise ValueError("No single-phase BPB labels were found after merging manifest and fit dataset.")
    merged["two_phase_bpb"] = merged["source_60m_bpb"]
    merged["delta_bpb"] = merged["single_phase_bpb"] - merged["two_phase_bpb"]
    merged = _add_original_shift_features(merged, source_fit_swarm_path)
    observed = merged[merged["single_phase_bpb"].notna()].copy()

    paired_csv = output_dir / "paired_single_vs_two_phase.csv"
    observed.to_csv(paired_csv, index=False)
    summary = {
        "n_manifest_rows": len(manifest),
        "n_observed_rows": len(observed),
        "mean_delta_bpb": float(observed["delta_bpb"].mean()),
        "median_delta_bpb": float(observed["delta_bpb"].median()),
        "positive_delta_fraction": float((observed["delta_bpb"] > 0).mean()),
        "paired_csv": str(paired_csv),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    scatter = px.scatter(
        observed,
        x="two_phase_bpb",
        y="single_phase_bpb",
        color="priority_tier",
        hover_data=["run_name", "source_run_name", "priority_rank", "phase_tv", "delta_bpb"],
        title="Single-phase exposure-average vs original two-phase BPB",
    )
    lower = min(observed["two_phase_bpb"].min(), observed["single_phase_bpb"].min())
    upper = max(observed["two_phase_bpb"].max(), observed["single_phase_bpb"].max())
    scatter.add_shape(type="line", x0=lower, y0=lower, x1=upper, y1=upper, line={"dash": "dash", "color": "black"})
    _write_plot(scatter, output_dir / "single_vs_two_phase_bpb.html")

    delta_hist = px.histogram(
        observed,
        x="delta_bpb",
        color="priority_tier",
        marginal="box",
        title="Delta BPB distribution: single-phase - two-phase",
    )
    _write_plot(delta_hist, output_dir / "delta_bpb_distribution.html")

    delta_phase_tv = px.scatter(
        observed,
        x="phase_tv",
        y="delta_bpb",
        color="priority_tier",
        hover_data=["run_name", "source_run_name", "priority_rank"],
        title="Delta BPB vs original phase TV distance",
    )
    _write_plot(delta_phase_tv, output_dir / "delta_bpb_vs_phase_tv.html")

    if "original_family_tv" in observed.columns:
        family_tv = px.scatter(
            observed,
            x="original_family_tv",
            y="delta_bpb",
            color="priority_tier",
            hover_data=["run_name", "source_run_name", "priority_rank"],
            title="Delta BPB vs original family-level phase TV",
        )
        _write_plot(family_tv, output_dir / "delta_bpb_vs_family_tv.html")

    if "original_high_quality_shift" in observed.columns:
        quality_shift = px.scatter(
            observed,
            x="original_high_quality_shift",
            y="delta_bpb",
            color="priority_tier",
            hover_data=["run_name", "source_run_name", "priority_rank"],
            title="Delta BPB vs phase-1 minus phase-0 high-quality share",
        )
        _write_plot(quality_shift, output_dir / "delta_bpb_vs_quality_shift.html")

    priority_curve = (
        observed.sort_values("priority_rank")
        .assign(completed=1)
        .loc[:, ["priority_rank", "completed"]]
        .assign(completed_rows=lambda frame: frame["completed"].cumsum())
    )
    curve = px.line(
        priority_curve,
        x="priority_rank",
        y="completed_rows",
        title="Priority-rank completion curve",
    )
    _write_plot(curve, output_dir / "priority_rank_completion_curve.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--single-fit-dataset", type=Path, default=DEFAULT_SINGLE_FIT_DATASET)
    parser.add_argument("--source-fit-swarm", type=Path, default=DEFAULT_SOURCE_FIT_SWARM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    analyze(
        manifest_path=args.manifest,
        single_fit_dataset_path=args.single_fit_dataset,
        source_fit_swarm_path=args.source_fit_swarm,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
