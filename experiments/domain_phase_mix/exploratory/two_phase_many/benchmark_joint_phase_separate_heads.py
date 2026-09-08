# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Apply matched-phase grouped CV to separate-heads phase bowls."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_coverage_augmented_separate_heads as separate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "joint_phase_separate_heads_20260709"


def benchmark(
    dataset: pooled.Dataset,
    panel_name: str,
    seeds: list[int],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha0, alpha1 = coverage.phase_fractions(dataset)
    metric_rows = []
    parameter_rows = []
    for seed in seeds:
        folds = joint.grouped_folds(dataset.frame, seed, n_splits)
        predictions = {
            "separate_heads": np.zeros(dataset.n, dtype=float),
            "separate_heads_coverage": np.zeros(dataset.n, dtype=float),
        }
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            for name, use_coverage in (("separate_heads", False), ("separate_heads_coverage", True)):
                model = separate.fit_head(
                    dataset,
                    train_idx,
                    use_coverage=use_coverage,
                    alpha0=alpha0,
                    alpha1=alpha1,
                )
                predictions[name][test_idx] = separate.predict(model, dataset, test_idx, alpha0, alpha1)
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "fit_panel": panel_name,
                        "model": name,
                        "seed": seed,
                        "fold": fold_id,
                        "theta_tv": float(model.coverage_coef[0]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_aggregate": float(model.coverage_coef[1]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_phase1": float(model.coverage_coef[2]) if len(model.coverage_coef) else 0.0,
                    }
                )
        for name, prediction in predictions.items():
            row = asdict(pooled.metrics(dataset, name, seed, prediction, folds))
            row["fit_panel"] = panel_name
            row["n_groups"] = int(dataset.frame["phase_correspondence_key"].nunique())
            row["nominal_param_count"] = 4 * dataset.m + 3 + 3 * int(name.endswith("_coverage"))
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def evaluate_external(
    fit_dataset: pooled.Dataset,
    external: pooled.Dataset,
    panel_name: str,
) -> pd.DataFrame:
    alpha0, alpha1 = coverage.phase_fractions(fit_dataset)
    rows = []
    for name, use_coverage in (("separate_heads", False), ("separate_heads_coverage", True)):
        model = separate.fit_head(
            fit_dataset,
            np.arange(fit_dataset.n),
            use_coverage=use_coverage,
            alpha0=alpha0,
            alpha1=alpha1,
        )
        prediction = separate.predict(model, external, np.arange(external.n), alpha0, alpha1)
        row = joint.external_metrics(name, external.y, prediction)
        row["dataset"] = fit_dataset.name
        row["fit_panel"] = panel_name
        row["external_rows"] = external.n
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objectives", default="uncheatable,table9")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    metric_frames = []
    parameter_frames = []
    external_frames = []
    for objective in [part.strip() for part in args.objectives.split(",") if part.strip()]:
        target = joint.TARGET_COLUMNS[objective]
        original = joint.dataset_from_frame(objective, frame.loc[frame["split"].eq("train")].copy(), target)
        matched = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            target,
        )
        external = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy(),
            target,
        )
        for panel_name, dataset in (("original_290", original), ("joint_matched", matched)):
            metrics, parameters = benchmark(dataset, panel_name, pooled.parse_int_list(args.seeds), args.n_splits)
            metric_frames.append(metrics)
            parameter_frames.append(parameters)
            external_frames.append(evaluate_external(dataset, external, panel_name))
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = raw.groupby(["dataset", "fit_panel", "model"], as_index=False).agg(
        n_rows=("n_rows", "first"),
        n_groups=("n_groups", "first"),
        n_cv_seeds=("seed", "nunique"),
        oof_rmse_mean=("oof_rmse", "mean"),
        oof_rmse_std=("oof_rmse", "std"),
        oof_spearman_mean=("oof_spearman", "mean"),
        oof_spearman_std=("oof_spearman", "std"),
        fold_mean_regret_at_1_mean=("fold_mean_regret_at_1", "mean"),
        lower_tail_optimism_mean=("lower_tail_optimism", "mean"),
        low_tail_rmse_mean=("low_tail_rmse", "mean"),
    )
    parameters = pd.concat(parameter_frames, ignore_index=True)
    external = pd.concat(external_frames, ignore_index=True)
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    external.to_csv(args.output_dir / "external_two_phase_heldout_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(external.to_string(index=False))
    print(f"Wrote matched-phase separate-heads benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
