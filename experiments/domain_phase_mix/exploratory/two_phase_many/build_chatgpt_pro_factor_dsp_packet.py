#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Build a curated ChatGPT Pro packet for factor-DSP mixture analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import textwrap
from datetime import UTC, datetime
from pathlib import Path

TWO_PHASE_MANY_ROOT = Path(__file__).resolve().parent
MARIN_ROOT = TWO_PHASE_MANY_ROOT.parents[3]
PACKET_NAME = "chatgpt_pro_factor_dsp_packet_20260525"
PACKET_ROOT = TWO_PHASE_MANY_ROOT / "reference_outputs" / PACKET_NAME
ZIP_PATH = PACKET_ROOT.with_suffix(".zip")


README_TEXT = """\
# Partition and Swarm-Based Data Mixing Packet

This packet is a handoff for external analysis sessions on partition-based data mixing for LLM pretraining. It is intended to be self-contained enough for statistical critique and prototype analysis without importing Marin.

## Problem Framing

We have a fixed upstream partition of pretraining data into coarse domains. A training candidate is a two-phase mixture: one simplex vector over domains for phase 0 and one simplex vector for phase 1. The current 300M/6B swarm evaluates many such candidates and records a wide panel of metrics.

The research question is not just "which mixture is best?" We want a statistically defensible workflow for choosing mixtures that improve capabilities we care about while surfacing or constraining regressions on other tasks. Key difficulties are noisy metrics, metric-specific actuation gaps, surrogate extrapolation, and the fact that our feasible mixture family is constrained by the partition basis.

## Contents

- `data/raw_metric_matrix_300m/`: 300M/6B metric matrix snapshots, including signal rows and available noise panels.
- `data/aggregate_metric_clean_slate_20260518/`: aggregate/factor-analysis artifacts and task-level DSP controllability diagnostics.
- `data/dsp/`: canonical DSP and related model-comparison summaries.
- `data/moe_path/`: Grug-MoE v4 interpolation path data and current path-response analysis.
- `data/perturbations/`: proportional perturbation, DSP-gradient agreement, and proportional-controllability metadata.
- `data/snr/`: SNR and older IRT/factor diagnostic artifacts.
- `data/grug_v4_aggregate_reproduction/`: local reproduction attempts for the collaborator aggregate that produced the v4 mixture.
- `notes/`: current theory notes and research logbooks.
- `code/analysis_starter.py`: self-contained starter analysis script with PEP 723 dependencies.
- `PROMPT.md`: paste-ready prompt for ChatGPT Pro or another analysis session.
- `MANIFEST.json` and `MANIFEST.csv`: file list, sizes, sources, and SHA-256 hashes.

## Important Caveats

- The historical scale key `300m_6b` is sometimes displayed elsewhere as corrected `100M/6B`; this packet preserves existing file names.
- The Grug-MoE v4 path data are incomplete while follow-up evals are still being filled. Treat path analyses as interim.
- Some perturbation and eval coverage was still being backfilled at packet time. Use the included coverage summaries before making strong task-level claims.
- Factor rotations are not identifiable without conventions; varimax/factor names should be treated as diagnostic structure, not discovered ground truth.
- DSP surrogate optima are model predictions, not validation results. Trust-region and uncertainty-aware constraints are central to the intended dashboard.

## Suggested First Commands

From the packet root:

```bash
uv run --script code/analysis_starter.py --packet-root . --output-dir scratch_analysis
```

This script is intentionally lightweight. It checks core data shapes, summarizes missingness, runs a PCA/varimax diagnostic on the raw metric matrix, and summarizes the path-response and DSP-controllability tables when present.
"""


PROMPT_TEXT = """\
# Prompt to accompany the attached packet

I am working on a research project on **Partition and Swarm-Based Data Mixing** for LLM pretraining. I have attached a packet with raw data, derived diagnostics, theory notes, and starter code. Please treat the packet files as the primary source of truth and cite file names when you make claims.

## Gentle introduction

Our pretraining data are partitioned upstream into coarse data buckets/domains. A candidate training mixture is a two-phase schedule: phase 0 has weights over domains, and phase 1 has weights over domains. Because we can only choose weights over the existing partition, the feasible family is a low-dimensional convex family over coarse partitions, not the full space of possible token multisets or curricula.

The proportional mixture is a strong baseline because it preserves broad corpus coverage. We then run a swarm of many mixture candidates at proxy scale, evaluate a large metric panel, and fit surrogate models. The current canonical surrogate is a Domain Saturation-Penalty (DSP) model, especially the effective-exposure form. We also have factor-analysis aggregates intended to denoise the metric panel and capture latent capabilities more robustly than individual noisy metrics.

Conceptually, we are trying to separate:

1. **Measurement noise:** metric values can vary due to seed/eval noise.
2. **Projected controllability:** a benchmark may or may not have a gradient projected onto the current partition-mixture tangent space.
3. **Actuation/representation gaps:** the current partition basis may not expose the task-relevant latent data features, even if they exist in the corpus.
4. **Tradeoffs:** some mixtures improve code/math-like tasks while hurting broad commonsense or language tasks.

## Current goal

We want to design a **factor-DSP constrained optimization dashboard**. The UX idea is:

- Start from the proportional mixture's predicted performance.
- Let a user specify tasks/factors they want to improve and tasks/factors they want to guardrail.
- Fit/present a DSP surrogate with uncertainty and diagnostics.
- Return candidate mixtures subject to per-task/per-factor performance constraints.
- Clearly show predicted downsides, extrapolation risk, nearest observed mixtures, and which metrics are too noisy or weakly actuated to trust as direct constraints.

## Please analyze the packet

Focus on these questions:

1. **Aggregate/factor metric design.** Is the current factor-analysis approach statistically sound enough for denoising and capability aggregation? How should factors be weighted, rotated, stabilized, or selected? What bootstrap/stability diagnostics should we require before using a factor as an optimization target?

2. **Constraint dashboard formulation.** Propose a concrete optimization formulation for "improve these targets while not regressing these guardrails" using factor scores, per-task DSP predictions, uncertainty, and trust-region constraints around proportional/observed mixtures.

3. **Noisy and weakly actuated metrics.** How should low-SNR or low-controllability tasks enter the dashboard? Should they be excluded, used only as guardrails with slack, folded into factors, or handled by robust/DRO-style constraints?

4. **Theory critique.** Review the theory notes. Is the representation-gap / actuation-gap / projected-controllability framing rigorous? What definitions should be tightened for a statistical learning theory audience?

5. **DSP and surrogate modeling.** Evaluate whether DSP is the right functional form for this dashboard. Suggest improvements that preserve interpretability and avoid unjustified inductive bias. Explain how to quantify extrapolation risk.

6. **Next experiments.** Given limited compute and fixed partitions, recommend the most informative next experiments to improve the dashboard: path tests, central log-tilts, domain deletions, repeated seeds/noise panels, or alternative metric collection.

## Expected output

Please produce a concrete research memo, not generic advice. I would like:

- A prioritized list of changes to the factor aggregate and dashboard objective.
- A proposed statistical validation checklist.
- A recommended constrained optimization objective with equations.
- A critique of what conclusions are and are not supported by the current data.
- Specific packet files/tables you used.
- If you write code, keep it self-contained and compatible with the packet layout.
"""


ANALYSIS_STARTER = """\
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
# ]
# ///
\"\"\"Starter analysis for the factor-DSP data-mixing packet.

This script intentionally does not import Marin. It loads packet-local CSV files,
prints basic coverage diagnostics, runs a simple PCA + varimax diagnostic over
the raw metric panel, and summarizes path-response/controllability tables.
\"\"\"

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ID_PREFIXES = (
    "registry_",
    "run_",
    "scale",
    "cohort",
    "source_",
    "checkpoint",
    "wandb",
    "status",
    "row_",
    "is_",
    "noise_",
    "trainer_seed",
    "data_seed",
    "phase_",
)


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    \"\"\"Orthogonal varimax rotation for diagnostic factor loading inspection.\"\"\"
    n_rows, n_cols = loadings.shape
    rotation = np.eye(n_cols)
    previous = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        transform = loadings.T @ (rotated**3 - (gamma / n_rows) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        u, singular_values, vh = np.linalg.svd(transform)
        rotation = u @ vh
        current = singular_values.sum()
        if previous and current < previous * (1.0 + tol):
            break
        previous = current
    return loadings @ rotation


def metric_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if any(col.startswith(prefix) for prefix in ID_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            finite_fraction = np.isfinite(frame[col].to_numpy(dtype=float, na_value=np.nan)).mean()
            if finite_fraction >= 0.5:
                cols.append(col)
    return cols


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"missing optional file: {path}")
        return None
    return pd.read_csv(path)


def analyze_metric_matrix(packet_root: Path, output_dir: Path) -> None:
    matrix_path = packet_root / "data/raw_metric_matrix_300m/raw_metric_matrix_300m.csv"
    frame = pd.read_csv(matrix_path)
    print(f"raw metric matrix: {frame.shape} from {matrix_path}")

    metrics = metric_columns(frame)
    print(f"numeric metric-like columns with >=50% finite coverage: {len(metrics)}")

    missing = (
        frame[metrics]
        .isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "metric"})
        .sort_values("missing_fraction", ascending=False)
    )
    missing.to_csv(output_dir / "metric_missingness.csv", index=False)

    values = frame[metrics].replace([np.inf, -np.inf], np.nan)
    values = SimpleImputer(strategy="median").fit_transform(values)
    values = StandardScaler().fit_transform(values)

    n_components = min(12, values.shape[0] - 1, values.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(values)
    variance = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    variance.to_csv(output_dir / "pca_variance.csv", index=False)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    rotated = varimax(loadings[:, : min(5, n_components)])
    loading_frame = pd.DataFrame(rotated, columns=[f"varimax_{i}" for i in range(1, rotated.shape[1] + 1)])
    loading_frame.insert(0, "metric", metrics)
    loading_frame.to_csv(output_dir / "varimax_loadings.csv", index=False)

    score_frame = pd.DataFrame(scores[:, : min(5, n_components)], columns=[f"pc_{i}" for i in range(1, min(5, n_components) + 1)])
    for col in ["run_name", "registry_run_key", "source_experiment", "row_kind"]:
        if col in frame.columns:
            score_frame.insert(0, col, frame[col])
    score_frame.to_csv(output_dir / "pca_scores.csv", index=False)

    if {"pc_1", "pc_2"}.issubset(score_frame.columns):
        hover_cols = [col for col in ["run_name", "source_experiment", "row_kind"] if col in score_frame.columns]
        fig = px.scatter(score_frame, x="pc_1", y="pc_2", color="source_experiment" if "source_experiment" in score_frame.columns else None, hover_data=hover_cols)
        fig.write_html(output_dir / "pca_score_scatter.html")


def summarize_path_response(packet_root: Path, output_dir: Path) -> None:
    summary = load_optional_csv(packet_root / "data/moe_path/path_response_analysis/task_t_response_summary.csv")
    if summary is None:
        return
    sort_col = "mean_scale_slope" if "mean_scale_slope" in summary.columns else "pooled_pearson_r"
    cols = [col for col in ["task_alias", "task_group", "n_points", "mean_scale_slope", "pooled_pearson_r", "positive_scale_slope_fraction", "delta_t075_mean"] if col in summary.columns]
    ranked = summary[cols].sort_values(sort_col)
    ranked.to_csv(output_dir / "path_response_ranked.csv", index=False)
    print("\\nTasks most negative along the v4 interpolation path:")
    print(ranked.head(10).to_string(index=False))
    print("\\nTasks most positive along the v4 interpolation path:")
    print(ranked.tail(10).to_string(index=False))


def summarize_controllability(packet_root: Path, output_dir: Path) -> None:
    table = load_optional_csv(packet_root / "data/aggregate_metric_clean_slate_20260518/metric_controllability_effective_exposure_dsp.csv")
    if table is None:
        return
    score_cols = [col for col in ["metric", "suite", "recommended_role", "dsp_oof_spearman", "dsp_oof_r2", "dsp_controllability_score"] if col in table.columns]
    ranked = table[score_cols].sort_values("dsp_oof_spearman" if "dsp_oof_spearman" in table.columns else score_cols[-1], ascending=False)
    ranked.to_csv(output_dir / "dsp_controllability_ranked.csv", index=False)
    role_summary = ranked.groupby("recommended_role", dropna=False).size().rename("n").reset_index() if "recommended_role" in ranked.columns else pd.DataFrame()
    if not role_summary.empty:
        role_summary.to_csv(output_dir / "dsp_controllability_role_counts.csv", index=False)
    print("\\nTop DSP-fittable/controllable metrics:")
    print(ranked.head(15).to_string(index=False))
    print("\\nWeakest DSP-fittable/controllable metrics:")
    print(ranked.tail(15).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch_analysis"))
    args = parser.parse_args()

    packet_root = args.packet_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_metric_matrix(packet_root, output_dir)
    summarize_path_response(packet_root, output_dir)
    summarize_controllability(packet_root, output_dir)
    print(f"\\nWrote starter outputs to {output_dir}")


if __name__ == "__main__":
    main()
"""


COPY_FILES: tuple[tuple[str, str, str], ...] = (
    # Raw matrix and noise panels.
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv",
        "data/raw_metric_matrix_300m/raw_metric_matrix_300m.csv",
        "raw_metric_matrix_signal",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m_with_noise.csv",
        "data/raw_metric_matrix_300m/raw_metric_matrix_300m_with_noise.csv",
        "raw_metric_matrix_signal_and_noise",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m_with_proportional_noise.csv",
        "data/raw_metric_matrix_300m/raw_metric_matrix_300m_with_proportional_noise.csv",
        "raw_metric_matrix_with_proportional_noise",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/noise_baseline_run00097_variable_subset_300m.csv",
        "data/raw_metric_matrix_300m/noise_baseline_run00097_variable_subset_300m.csv",
        "noise_panel",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/noise_baseline_run00097_fixed_subset_300m.csv",
        "data/raw_metric_matrix_300m/noise_baseline_run00097_fixed_subset_300m.csv",
        "noise_panel",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/noise_baseline_proportional_variable_subset_300m.csv",
        "data/raw_metric_matrix_300m/noise_baseline_proportional_variable_subset_300m.csv",
        "noise_panel",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/summary.json",
        "data/raw_metric_matrix_300m/summary.json",
        "raw_metric_matrix_summary",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/README.md",
        "data/raw_metric_matrix_300m/README.md",
        "raw_metric_matrix_docs",
    ),
    # Aggregate/factor artifacts.
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_factor_loadings.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_factor_loadings.csv",
        "factor_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_factor_scores.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_factor_scores.csv",
        "factor_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_factor_summary.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_factor_summary.csv",
        "factor_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_factor_horn_parallel.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_factor_horn_parallel.csv",
        "factor_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_factor_item_table.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_factor_item_table.csv",
        "factor_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_item_table.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_item_table.csv",
        "aggregate_metric",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_candidate_summary.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_candidate_summary.csv",
        "aggregate_metric",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_candidate_scores.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_candidate_scores.csv",
        "aggregate_metric",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_candidate_correlations.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_candidate_correlations.csv",
        "aggregate_metric",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_candidate_effective_exposure_dsp_summary.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_candidate_effective_exposure_dsp_summary.csv",
        "aggregate_dsp",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_candidate_effective_exposure_dsp_summary.json",
        "data/aggregate_metric_clean_slate_20260518/aggregate_candidate_effective_exposure_dsp_summary.json",
        "aggregate_dsp",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_candidate_effective_exposure_dsp_raw_optimum_weights.csv",
        "data/aggregate_metric_clean_slate_20260518/aggregate_candidate_effective_exposure_dsp_raw_optimum_weights.csv",
        "aggregate_dsp",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/metric_controllability_effective_exposure_dsp.csv",
        "data/aggregate_metric_clean_slate_20260518/metric_controllability_effective_exposure_dsp.csv",
        "metric_controllability",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/metric_controllability_effective_exposure_dsp_summary.json",
        "data/aggregate_metric_clean_slate_20260518/metric_controllability_effective_exposure_dsp_summary.json",
        "metric_controllability",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/metric_controllability_effective_exposure_dsp_role_summary.csv",
        "data/aggregate_metric_clean_slate_20260518/metric_controllability_effective_exposure_dsp_role_summary.csv",
        "metric_controllability",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/metric_controllability_oof_ridge.csv",
        "data/aggregate_metric_clean_slate_20260518/metric_controllability_oof_ridge.csv",
        "metric_controllability",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/moe_task_scaling_r2_summary.csv",
        "data/aggregate_metric_clean_slate_20260518/moe_task_scaling_r2_summary.csv",
        "moe_metric_diagnostics",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/moe_task_snr_r2_join.csv",
        "data/aggregate_metric_clean_slate_20260518/moe_task_snr_r2_join.csv",
        "moe_metric_diagnostics",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/raw_ppl_snr_points.csv",
        "data/aggregate_metric_clean_slate_20260518/raw_ppl_snr_points.csv",
        "snr",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/reactive_metric_table.csv",
        "data/aggregate_metric_clean_slate_20260518/reactive_metric_table.csv",
        "metric_table",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/reactive_metric_table_best_by_item.csv",
        "data/aggregate_metric_clean_slate_20260518/reactive_metric_table_best_by_item.csv",
        "metric_table",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/reactive_metric_table_role_summary.csv",
        "data/aggregate_metric_clean_slate_20260518/reactive_metric_table_role_summary.csv",
        "metric_table",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/aggregate_metric_clean_slate_20260518.html",
        "data/aggregate_metric_clean_slate_20260518/aggregate_metric_clean_slate_20260518.html",
        "dashboard_html",
    ),
    # DSP model checks.
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_300m_20260510/summary.csv",
        "data/dsp/canonical_variants_300m_summary.csv",
        "dsp_summary",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_300m_20260510/summary.json",
        "data/dsp/canonical_variants_300m_summary.json",
        "dsp_summary",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_300m_20260510/report.md",
        "data/dsp/canonical_variants_300m_report.md",
        "dsp_report",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_raml_variants_300m_20260521/summary.csv",
        "data/dsp/raml_variants_300m_summary.csv",
        "dsp_summary",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_raml_variants_300m_20260521/report.md",
        "data/dsp/raml_variants_300m_report.md",
        "dsp_report",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_split_vs_effective_checks_20260514/fit_summary.csv",
        "data/dsp/split_vs_effective_fit_summary.csv",
        "dsp_summary",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_split_vs_effective_checks_20260514/domain_perturbation_prediction_summary.csv",
        "data/dsp/split_vs_effective_domain_perturbation_prediction_summary.csv",
        "dsp_perturbation",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_split_vs_effective_checks_20260514/domain_perturbation_predictions.csv",
        "data/dsp/split_vs_effective_domain_perturbation_predictions.csv",
        "dsp_perturbation",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_split_vs_effective_checks_20260514/report.md",
        "data/dsp/split_vs_effective_report.md",
        "dsp_report",
    ),
    # MoE/path tests.
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/grug_moe_mix_eval_metrics_long.csv",
        "data/moe_path/grug_moe_mix_eval_metrics_long.csv",
        "moe_tracks",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/grug_moe_mix_preferred_task_metrics.csv",
        "data/moe_path/grug_moe_mix_preferred_task_metrics.csv",
        "moe_tracks",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/grug_moe_mix_runs.csv",
        "data/moe_path/grug_moe_mix_runs.csv",
        "moe_tracks",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/grug_moe_mix_weights_long.csv",
        "data/moe_path/grug_moe_mix_weights_long.csv",
        "moe_tracks",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/grug_moe_v4_path_eval_metrics_long.csv",
        "data/moe_path/grug_moe_v4_path_eval_metrics_long.csv",
        "moe_path",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/grug_moe_v4_path_training_runs.csv",
        "data/moe_path/grug_moe_v4_path_training_runs.csv",
        "moe_path",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/img/path_task_delta_facets.html",
        "data/moe_path/path_task_delta_facets.html",
        "path_plot",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/img/path_task_scaling_facets.html",
        "data/moe_path/path_task_scaling_facets.html",
        "path_plot",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_path_response_analysis_20260525/summary.md",
        "data/moe_path/path_response_analysis/summary.md",
        "path_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_path_response_analysis_20260525/coverage_summary.csv",
        "data/moe_path/path_response_analysis/coverage_summary.csv",
        "path_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_path_response_analysis_20260525/task_t_mean_deltas.csv",
        "data/moe_path/path_response_analysis/task_t_mean_deltas.csv",
        "path_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_path_response_analysis_20260525/task_t_response_summary.csv",
        "data/moe_path/path_response_analysis/task_t_response_summary.csv",
        "path_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_path_response_analysis_20260525/task_t_mean_delta_facets.html",
        "data/moe_path/path_response_analysis/task_t_mean_delta_facets.html",
        "path_analysis_plot",
    ),
    # Perturbations and proportional controllability.
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/intervention_manifest.csv",
        "data/perturbations/proportional_perturbation_intervention_manifest.csv",
        "perturbation_manifest",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/training_manifest.csv",
        "data/perturbations/proportional_perturbation_training_manifest.csv",
        "perturbation_manifest",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/dsp_domain_perturbation_agreement_summary.csv",
        "data/perturbations/dsp_domain_perturbation_agreement_summary.csv",
        "perturbation_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/dsp_domain_perturbation_agreement.csv",
        "data/perturbations/dsp_domain_perturbation_agreement.csv",
        "perturbation_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/paired_bpb_effects.csv",
        "data/perturbations/paired_bpb_effects.csv",
        "perturbation_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_perturbation_scale_transfer_20260507/partial_60m_perturbation_results.csv",
        "data/perturbations/partial_60m_perturbation_results.csv",
        "perturbation_analysis",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/intervention_manifest.csv",
        "data/perturbations/proportional_controllability_intervention_manifest.csv",
        "controllability_design",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/log_tilt_target_multiplier_summary.csv",
        "data/perturbations/log_tilt_target_multiplier_summary.csv",
        "controllability_design",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_variable_subset_noise_collected_20260511.csv",
        "data/perturbations/proportional_variable_subset_noise_collected_20260511.csv",
        "noise_panel",
    ),
    # SNR and old IRT/factor diagnostics.
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/300m_irt_factor_analysis_20260501/summary.json",
        "data/snr/300m_irt_factor_analysis_summary.json",
        "snr_factor",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/300m_irt_factor_analysis_20260501/factor_noise_snr.csv",
        "data/snr/factor_noise_snr.csv",
        "snr_factor",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/300m_irt_factor_analysis_20260501/irt_item_noise_coverage.csv",
        "data/snr/irt_item_noise_coverage.csv",
        "snr_factor",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/300m_snr_fixed_vs_variable_20260501/300m_metric_snr_fixed_vs_variable.csv",
        "data/snr/300m_metric_snr_fixed_vs_variable.csv",
        "snr",
    ),
    # Grug v4 aggregate reproduction.
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/old_packet/report.md",
        "data/grug_v4_aggregate_reproduction/old_packet_report.md",
        "grug_v4_reproduction",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/old_packet/summary.json",
        "data/grug_v4_aggregate_reproduction/old_packet_summary.json",
        "grug_v4_reproduction",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/sent_raw_metric_matrix_300m_zip/report.md",
        "data/grug_v4_aggregate_reproduction/sent_raw_metric_matrix_report.md",
        "grug_v4_reproduction",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/sent_raw_metric_matrix_300m_zip/summary.json",
        "data/grug_v4_aggregate_reproduction/sent_raw_metric_matrix_summary.json",
        "grug_v4_reproduction",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/canonical_dsp_sent_zip/report.md",
        "data/grug_v4_aggregate_reproduction/canonical_dsp_sent_zip_report.md",
        "grug_v4_reproduction",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/canonical_dsp_sent_zip/summary.json",
        "data/grug_v4_aggregate_reproduction/canonical_dsp_sent_zip_summary.json",
        "grug_v4_reproduction",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/canonical_dsp_sent_zip/model.json",
        "data/grug_v4_aggregate_reproduction/canonical_dsp_sent_zip_model.json",
        "grug_v4_reproduction",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/phase_benefit_dsp_sent_zip/report.md",
        "data/grug_v4_aggregate_reproduction/phase_benefit_dsp_sent_zip_report.md",
        "grug_v4_reproduction",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/phase_benefit_dsp_sent_zip/summary.json",
        "data/grug_v4_aggregate_reproduction/phase_benefit_dsp_sent_zip_summary.json",
        "grug_v4_reproduction",
    ),
    # Self-contained reference code.
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/standalone_code/dsp_exact.py",
        "code/dsp_exact.py",
        "reference_code",
    ),
    (
        "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/issue5416_aggregate.py",
        "code/issue5416_aggregate.py",
        "reference_code",
    ),
    # Notes and theory.
    (
        ".agents/logbooks/partition-swarm-data-mixing-theory.md",
        "notes/logbooks/partition-swarm-data-mixing-theory.md",
        "logbook",
    ),
    (
        ".agents/logbooks/grug-moe-v4-path-test.md",
        "notes/logbooks/grug-moe-v4-path-test.md",
        "logbook",
    ),
    (
        ".agents/logbooks/grug-v4-aggregate-reproduction.md",
        "notes/logbooks/grug-v4-aggregate-reproduction.md",
        "logbook",
    ),
    (
        ".agents/logbooks/proportional-perturbation-scale-transfer.md",
        "notes/logbooks/proportional-perturbation-scale-transfer.md",
        "logbook",
    ),
    (
        ".agents/logbooks/proportional-controllability-300m.md",
        "notes/logbooks/proportional-controllability-300m.md",
        "logbook",
    ),
    (
        ".agents/logbooks/proportional-variable-subset-noise.md",
        "notes/logbooks/proportional-variable-subset-noise.md",
        "logbook",
    ),
)


EXTERNAL_FILES: tuple[tuple[Path, str, str], ...] = (
    (
        Path.home() / "Downloads" / "Representability in Stratified Sampling.md",
        "notes/Representability in Stratified Sampling.md",
        "external_note",
    ),
    (
        Path.home()
        / "Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper/theory.md",
        "notes/theory.md",
        "external_theory_note",
    ),
    (
        Path.home()
        / "Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper/representability_and_token_sampling.md",
        "notes/representability_and_token_sampling.md",
        "external_theory_note",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def copy_inputs() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for src_rel, dst_rel, kind in COPY_FILES:
        src = MARIN_ROOT / src_rel
        dst = PACKET_ROOT / dst_rel
        copy_file(src, dst)
        records.append({"source_path": str(src), "path": dst_rel, "kind": kind})

    for src, dst_rel, kind in EXTERNAL_FILES:
        dst = PACKET_ROOT / dst_rel
        copy_file(src, dst)
        records.append({"source_path": str(src), "path": dst_rel, "kind": kind})

    return records


def write_generated_files(records: list[dict[str, object]]) -> None:
    generated = (
        ("README.md", README_TEXT, "packet_docs"),
        ("PROMPT.md", PROMPT_TEXT, "prompt"),
        ("code/analysis_starter.py", ANALYSIS_STARTER, "starter_code"),
    )
    for dst_rel, text, kind in generated:
        dst = PACKET_ROOT / dst_rel
        write_text(dst, text)
        records.append({"source_path": "generated", "path": dst_rel, "kind": kind})


def write_manifest(records: list[dict[str, object]]) -> None:
    enriched: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: str(item["path"])):
        path = PACKET_ROOT / str(record["path"])
        enriched.append(
            {
                "path": str(record["path"]),
                "kind": str(record["kind"]),
                "source_path": str(record["source_path"]),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "packet_name": PACKET_NAME,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "marin_root": str(MARIN_ROOT),
        "file_count": len(enriched),
        "files": enriched,
    }
    write_text(PACKET_ROOT / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with (PACKET_ROOT / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "kind", "source_path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(enriched)


def write_archive() -> Path:
    for ds_store in PACKET_ROOT.rglob(".DS_Store"):
        ds_store.unlink()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    archive = shutil.make_archive(str(PACKET_ROOT), "zip", root_dir=PACKET_ROOT.parent, base_dir=PACKET_ROOT.name)
    return Path(archive)


def main() -> None:
    if PACKET_ROOT.exists():
        shutil.rmtree(PACKET_ROOT)
    PACKET_ROOT.mkdir(parents=True)
    records = copy_inputs()
    write_generated_files(records)
    for ds_store in PACKET_ROOT.rglob(".DS_Store"):
        ds_store.unlink()
    write_manifest(records)
    archive = write_archive()
    print(f"Built packet: {PACKET_ROOT}")
    print(f"Wrote archive: {archive}")


if __name__ == "__main__":
    main()
