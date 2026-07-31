# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Compute domain-deletion p-values directly from an augmented pctrl matrix.

This is a broader version of the earlier p-value histogram input builder.  It
does not require local-gradient columns, so it can include training-time
``eval/*`` metrics collected from W&B, such as ``eval/uncheatable_eval/*`` and
``eval/paloma/*``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_MATRIX = (
    SCRIPT_DIR
    / "reference_outputs"
    / "pctrl_training_eval_wandb_collect_20260623"
    / "pctrl_final_metric_matrix_with_training_eval.csv"
)
DEFAULT_NOISE_MATRIX = (
    SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m_with_proportional_noise.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "domain_ablation_pvalue_matrix_with_training_eval_20260623"

METRIC_PREFIXES = ("eval/", "lm_eval/", "teacher_forced/", "mcq_smooth/", "raw_ppl/")
NON_METRIC_SUFFIXES = {
    "bits",
    "bytes",
    "documents",
    "example_count",
    "target_bytes",
    "total_time",
    "loading_time",
}
SMOOTH_KIND_PRIORITY = {
    "bpb": 0,
    "macro_bpb": 1,
    "micro_bpb": 2,
    "loss": 3,
    "macro_loss": 4,
    "micro_loss": 5,
    "nll": 6,
    "choice_logprob_norm": 7,
    "choice_logprob": 8,
    "correct_vs_best_incorrect_margin": 9,
    "normalized_correct_vs_best_incorrect_margin": 10,
    "success_macro_bpb": 11,
    "failed_macro_bpb": 12,
    "coderforge_success_macro_bpb": 13,
    "coderforge_failed_macro_bpb": 14,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-matrix", type=Path, default=DEFAULT_INPUT_MATRIX)
    parser.add_argument("--noise-matrix", type=Path, default=DEFAULT_NOISE_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def metric_kind(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1]


def metric_family(metric: str) -> str:
    return metric.split("/", maxsplit=1)[0]


def is_metric_column(column: str) -> bool:
    if not column.startswith(METRIC_PREFIXES):
        return False
    kind = metric_kind(column)
    if kind in NON_METRIC_SUFFIXES:
        return False
    if "stderr" in kind:
        return False
    return True


def lower_is_better(metric: str) -> bool:
    kind = metric_kind(metric).lower()
    return (
        kind in {"bpb", "loss", "nll", "macro_bpb", "macro_loss", "macro_nll", "micro_bpb", "micro_loss", "micro_nll"}
        or kind.endswith("_bpb")
        or kind.endswith("_loss")
        or kind.endswith("_nll")
        or "perplexity" in kind
    )


def has_ambiguous_direction(metric: str) -> bool:
    kind = metric_kind(metric).lower()
    return any(token in kind for token in ("minus", "diff", "delta", "ratio", "gain", "reduction", "improvement"))


def is_smooth_proxy(metric: str) -> bool:
    if has_ambiguous_direction(metric):
        return False
    kind = metric_kind(metric).lower()
    return kind in SMOOTH_KIND_PRIORITY or kind.endswith("_bpb") or kind.endswith("_loss") or kind.endswith("_nll")


def smooth_priority(metric: str) -> tuple[int, str]:
    kind = metric_kind(metric).lower()
    if kind in SMOOTH_KIND_PRIORITY:
        return SMOOTH_KIND_PRIORITY[kind], metric
    if kind.endswith("_bpb"):
        return 30, metric
    if kind.endswith("_loss"):
        return 40, metric
    if kind.endswith("_nll"):
        return 50, metric
    return 999, metric


def benchmark_key(metric: str) -> str:
    kind = metric_kind(metric).lower()
    if kind in SMOOTH_KIND_PRIORITY and not kind.endswith("_bpb") and not kind.endswith("_loss"):
        return metric.rsplit("/", maxsplit=1)[0]
    if kind in {"bpb", "loss", "nll"}:
        return metric.rsplit("/", maxsplit=1)[0]
    if kind.startswith("macro_") or kind.startswith("micro_"):
        return metric
    if kind.endswith("_bpb") or kind.endswith("_loss") or kind.endswith("_nll"):
        return metric
    return metric.rsplit("/", maxsplit=1)[0]


def utility_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")
    return -values if lower_is_better(metric) else values


def proportional_reference_values(noise_matrix: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in noise_matrix.columns:
        return np.asarray([], dtype=float)
    baseline = noise_matrix[noise_matrix["run_name"].eq("baseline_proportional")]
    repeats = noise_matrix[noise_matrix["row_kind"].eq("noise_variable_subset_proportional")]
    reference = pd.concat([baseline, repeats], ignore_index=True)
    return utility_values(reference, metric).dropna().to_numpy(dtype=float)


def selected_smooth_metrics(matrix: pd.DataFrame, noise_matrix: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in matrix.columns
        if is_metric_column(column) and is_smooth_proxy(column) and column in noise_matrix.columns
    ]
    rows: list[dict[str, Any]] = []
    for metric in metric_columns:
        noise = proportional_reference_values(noise_matrix, metric)
        if len(noise) < 3:
            continue
        if not np.isfinite(noise).all() or float(np.std(noise, ddof=1)) <= 0:
            continue
        rows.append(
            {
                "benchmark_key": benchmark_key(metric),
                "metric": metric,
                "metric_family": metric_family(metric),
                "metric_kind": metric_kind(metric),
                "lower_is_better": lower_is_better(metric),
                "_priority": smooth_priority(metric),
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No smooth metrics with proportional noise coverage found.")
    metrics = metrics.sort_values(["benchmark_key", "_priority", "metric"])
    return metrics.groupby("benchmark_key", as_index=False, sort=False).first().drop(columns=["_priority"])


def compute_pvalues(matrix: pd.DataFrame, noise_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deletions = matrix[matrix["intervention_type"].eq("domain_deletion")].copy()
    if len(deletions) != 39:
        raise ValueError(f"Expected 39 domain-deletion rows, found {len(deletions)}.")
    selected = selected_smooth_metrics(matrix, noise_matrix)

    rows: list[dict[str, Any]] = []
    for selected_metric in selected.itertuples(index=False):
        metric = str(selected_metric.metric)
        reference = proportional_reference_values(noise_matrix, metric)
        if len(reference) < 3:
            continue
        base_utility = float(np.mean(reference))
        noise = reference
        noise_sd = float(np.std(noise, ddof=1))
        n_noise = int(len(noise))
        df = n_noise - 1
        predictive_sd = noise_sd * math.sqrt(1.0 + 1.0 / n_noise)
        for _, deletion in deletions.iterrows():
            value = pd.to_numeric(pd.Series([deletion[metric]]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            utility = -value if lower_is_better(metric) else value
            delta = float(utility - base_utility)
            t_stat = delta / predictive_sd
            p_harm = float(student_t.cdf(t_stat, df=df))
            p_improve = float(student_t.sf(t_stat, df=df))
            p_two_sided = float(2.0 * min(p_harm, p_improve))
            rows.append(
                {
                    "benchmark_key": selected_metric.benchmark_key,
                    "metric": metric,
                    "metric_family": selected_metric.metric_family,
                    "metric_kind": selected_metric.metric_kind,
                    "lower_is_better": bool(selected_metric.lower_is_better),
                    "target_domain": deletion["target_domain"],
                    "base_mass": float(deletion["base_mass"]),
                    "proportional_reference_utility": base_utility,
                    "domain_deletion_utility_delta": delta,
                    "noise_n": n_noise,
                    "noise_sd": noise_sd,
                    "predictive_sd": predictive_sd,
                    "t_statistic": t_stat,
                    "p_harm": min(max(p_harm, 0.0), 1.0),
                    "p_improve": min(max(p_improve, 0.0), 1.0),
                    "p_two_sided": min(max(p_two_sided, 0.0), 1.0),
                }
            )
    cell = pd.DataFrame(rows)
    if cell.empty:
        raise ValueError("No domain-deletion p-value cells computed.")

    summaries = []
    for metric, group in cell.groupby("metric", sort=False):
        n_domains = int(len(group))
        best = group.loc[group["p_harm"].idxmin()]
        summaries.append(
            {
                "benchmark_key": str(best["benchmark_key"]),
                "metric": metric,
                "metric_family": metric_family(metric),
                "metric_kind": metric_kind(metric),
                "n_domains": n_domains,
                "min_p_harm_raw": float(group["p_harm"].min()),
                "min_p_harm_bonferroni": float(min(1.0, group["p_harm"].min() * n_domains)),
                "best_harm_domain": str(best["target_domain"]),
                "best_harm_delta": float(best["domain_deletion_utility_delta"]),
                "best_harm_t": float(best["t_statistic"]),
                "fraction_harm_p_lt_0p05_raw": float((group["p_harm"] < 0.05).mean()),
                "fraction_domains_harm": float((group["domain_deletion_utility_delta"] < 0.0).mean()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(["min_p_harm_bonferroni", "min_p_harm_raw"])
    return cell, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(args.input_matrix, low_memory=False)
    noise_matrix = pd.read_csv(args.noise_matrix, low_memory=False)
    cell, summary = compute_pvalues(matrix, noise_matrix)
    cell_path = args.output_dir / "domain_ablation_cell_pvalues.csv"
    summary_path = args.output_dir / "domain_ablation_benchmark_min_pvalues.csv"
    cell.to_csv(cell_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        "input_matrix": str(args.input_matrix),
        "noise_matrix": str(args.noise_matrix),
        "cell_pvalues": str(cell_path),
        "benchmark_min_pvalues": str(summary_path),
        "cells": int(len(cell)),
        "benchmarks": int(cell["benchmark_key"].nunique()),
        "domains": int(cell["target_domain"].nunique()),
        "metrics": int(cell["metric"].nunique()),
        "eval_metrics": int(cell["metric"].str.startswith("eval/").sum() / 39),
        "uncheatable_metrics": int(cell["metric"].str.contains("uncheatable_eval").sum() / 39),
        "raw_p_harm_lt_0p05_cells": int((cell["p_harm"] < 0.05).sum()),
        "bonferroni_p_harm_lt_0p05_cells": int(
            (cell.groupby("metric")["p_harm"].transform(lambda s: np.minimum(1.0, s * len(s))) < 0.05).sum()
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
