# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Evaluate coverage-augmented DSP on extra 300M Table-9 checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_HELDOUT = (
    pooled.REFERENCE_OUTPUTS / "olmo_base_easy_extra_300m_heldout_eval_20260630/combined_300m_table9_heldout_panel.csv"
)
DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "coverage_dsp_extra_300m_heldout_20260709"
TARGET = "table9_macro_bpb"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def load_heldout(path: Path, domains: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[frame[TARGET].notna()].reset_index(drop=True)
    w0 = frame[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
    w1 = frame[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
    w0 /= w0.sum(axis=1, keepdims=True)
    w1 /= w1.sum(axis=1, keepdims=True)
    return frame, np.stack([w0, w1], axis=1), frame[TARGET].to_numpy(dtype=float)


def nearest_training_tv(training: np.ndarray, heldout: np.ndarray) -> np.ndarray:
    distances = np.empty(len(heldout), dtype=float)
    for index, weights in enumerate(heldout):
        distances[index] = float(np.min(0.5 * np.abs(training - weights[None, :, :]).sum(axis=2).mean(axis=1)))
    return distances


def metric_row(name: str, target: np.ndarray, prediction: np.ndarray) -> dict[str, float | str]:
    selected = int(np.argmin(prediction))
    tail_count = max(5, int(np.ceil(0.15 * len(target))))
    tail = np.argsort(prediction)[:tail_count]
    residual = prediction - target
    return {
        "model": name,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(spearmanr(target, prediction).statistic),
        "regret_at_1": float(target[selected] - np.min(target)),
        "selected_observed": float(target[selected]),
        "selected_predicted": float(prediction[selected]),
        "lower_tail_optimism": float(np.mean(np.maximum(-residual[tail], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maxiter", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training = pooled.load_300m_dataset("table9")
    frame, heldout_weights, heldout_target = load_heldout(args.heldout, training.domain_names)
    support_tv = nearest_training_tv(training.weights, heldout_weights)
    alpha0, alpha1 = coverage.phase_fractions(training)
    all_indices = np.arange(training.n)
    prediction_columns = {}
    rows = []
    for config in (
        coverage.FitConfig("effective_exposure", False),
        coverage.FitConfig("effective_exposure_coverage", True),
    ):
        model = coverage.fit_model(
            training,
            all_indices,
            config,
            linear_reg=coverage.dataset_linear_reg(training),
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        prediction = coverage.predict(model, heldout_weights, alpha0, alpha1)
        prediction_columns[config.name] = prediction
        rows.append(metric_row(config.name, heldout_target, prediction))

    predictions = frame[["run_name", "panel", "method", "diagnostic_group", "diagnostic_family", TARGET]].copy()
    predictions["nearest_training_tv"] = support_tv
    for name, values in prediction_columns.items():
        predictions[f"prediction_{name}"] = values
        predictions[f"residual_{name}"] = values - heldout_target
    summary = pd.DataFrame(rows)
    group_rows = []
    for group, indices in predictions.groupby("diagnostic_group", dropna=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        for name, values in prediction_columns.items():
            row = metric_row(name, heldout_target[idx], values[idx])
            row["diagnostic_group"] = group
            row["n_rows"] = len(idx)
            group_rows.append(row)
    grouped = pd.DataFrame(group_rows)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "heldout_summary.csv", index=False)
    grouped.to_csv(args.output_dir / "heldout_summary_by_group.csv", index=False)

    plot_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "observed": heldout_target,
                    "predicted": values,
                    "model": name,
                    "diagnostic_group": predictions["diagnostic_group"],
                }
            )
            for name, values in prediction_columns.items()
        ],
        ignore_index=True,
    )
    figure = px.scatter(
        plot_frame,
        x="observed",
        y="predicted",
        color="diagnostic_group",
        facet_col="model",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Extra 300M Table-9 heldout: predicted versus observed",
    )
    figure.write_html(
        args.output_dir / "heldout_predicted_vs_observed.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    print(summary.to_string(index=False))
    print(grouped.to_string(index=False))
    print(
        f"Heldout rows={len(frame)}, exact train overlaps={int(np.sum(support_tv < 1e-12))}, "
        f"median nearest-training TV={float(np.median(support_tv)):.6f}"
    )
    print(f"Wrote heldout evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
