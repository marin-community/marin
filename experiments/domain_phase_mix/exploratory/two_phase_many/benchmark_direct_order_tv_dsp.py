# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Fit the phase-TV correction directly from aggregate-matched ordering deltas.

The backbone is effective-exposure DSP. For each training correspondence pair,
the residual ordering effect is

    (y_two - y_tied) - (f_two - f_tied).

A single nonnegative coefficient maps phase TV to that residual. Tied schedules
have zero phase TV, so the correction changes only the predicted ordering
effect and cannot alter the aggregate-matched control.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_matched_phase_ordering as ordering,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "direct_order_tv_dsp_20260710"


@dataclass(frozen=True)
class DirectTVModel:
    base: geometry.CoverageModel
    theta_tv: float


def phase_tv(weights: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(weights[:, 0, :] - weights[:, 1, :]).sum(axis=1)


def complete_pair_indices(frame: pd.DataFrame, allowed: set[int]) -> list[tuple[int, int]]:
    rows = frame.reset_index().rename(columns={"index": "row_index"})
    rows = rows.loc[rows["row_index"].isin(allowed)]
    pairs = []
    for _key, group in rows.groupby("phase_correspondence_key", sort=False):
        tied = group.loc[group["policy_family"].eq("single_phase"), "row_index"].tolist()
        two_phase = group.loc[group["policy_family"].eq("two_phase"), "row_index"].tolist()
        if len(tied) == 1 and len(two_phase) == 1:
            pairs.append((int(tied[0]), int(two_phase[0])))
    return pairs


def fit_direct_tv(
    dataset: pooled.Dataset,
    train_idx: np.ndarray,
    maxiter: int,
    coarse_top_k: int,
) -> DirectTVModel:
    config = geometry.FitConfig("effective_exposure", False, "effective_exposure")
    base = geometry.fit_model(
        dataset,
        train_idx,
        config,
        linear_reg=geometry.dataset_linear_reg(dataset),
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )
    alpha0, alpha1 = geometry.phase_fractions(dataset)
    base_prediction = geometry.predict(base, dataset.weights, alpha0, alpha1)
    pairs = complete_pair_indices(dataset.frame, set(np.asarray(train_idx, dtype=int)))
    if not pairs:
        raise ValueError("No complete aggregate-matched pairs in training fold")
    tied_idx = np.asarray([pair[0] for pair in pairs], dtype=int)
    two_idx = np.asarray([pair[1] for pair in pairs], dtype=int)
    observed_delta = dataset.y[two_idx] - dataset.y[tied_idx]
    predicted_delta = base_prediction[two_idx] - base_prediction[tied_idx]
    feature = phase_tv(dataset.weights[two_idx])
    denominator = float(feature @ feature)
    theta_tv = 0.0 if denominator <= 0.0 else max(0.0, float(feature @ (observed_delta - predicted_delta)) / denominator)
    return DirectTVModel(base=base, theta_tv=theta_tv)


def predict(model: DirectTVModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    alpha0, alpha1 = geometry.phase_fractions(dataset)
    return geometry.predict(model.base, dataset.weights[indices], alpha0, alpha1) + model.theta_tv * phase_tv(
        dataset.weights[indices]
    )


def benchmark_dataset(
    dataset: pooled.Dataset,
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pair_rows = []
    parameter_rows = []
    for seed in seeds:
        folds = joint.grouped_folds(dataset.frame, seed, n_splits)
        direct_prediction = np.zeros(dataset.n, dtype=float)
        base_prediction = np.zeros(dataset.n, dtype=float)
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            model = fit_direct_tv(dataset, train_idx, maxiter, coarse_top_k)
            direct_prediction[test_idx] = predict(model, dataset, test_idx)
            alpha0, alpha1 = geometry.phase_fractions(dataset)
            base_prediction[test_idx] = geometry.predict(
                model.base,
                dataset.weights[test_idx],
                alpha0,
                alpha1,
            )
            parameter_rows.append(
                {
                    "dataset": dataset.name,
                    "seed": seed,
                    "fold": fold_id,
                    "theta_tv": model.theta_tv,
                    "gamma": float(model.base.base.params["gamma"]),
                }
            )
        for name, prediction, parameter_count in (
            ("effective_exposure", base_prediction, 4 * dataset.m + 2),
            ("effective_exposure_direct_tv", direct_prediction, 4 * dataset.m + 3),
        ):
            row = asdict(pooled.metrics(dataset, name, seed, prediction, folds))
            row["nominal_param_count"] = parameter_count
            metric_rows.append(row)
            pairs = ordering.pair_frame(dataset, prediction, name)
            proportional = dataset.frame.loc[
                dataset.frame["phase_correspondence_key"].eq("baseline_proportional")
                & dataset.frame["split"].eq("train"),
                joint.TARGET_COLUMNS[dataset.name.removeprefix("300m_")],
            ].to_numpy(dtype=float)
            pair_metric = ordering.pair_metrics(pairs, float(np.std(proportional, ddof=1)))
            pair_metric["seed"] = seed
            pair_rows.append(pair_metric)
    return pd.DataFrame(metric_rows), pd.DataFrame(pair_rows), pd.DataFrame(parameter_rows)


def external_evaluation(
    fit_dataset: pooled.Dataset,
    external: pooled.Dataset,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    model = fit_direct_tv(fit_dataset, np.arange(fit_dataset.n), maxiter, coarse_top_k)
    rows = []
    alpha0, alpha1 = geometry.phase_fractions(fit_dataset)
    predictions = {
        "effective_exposure": geometry.predict(model.base, external.weights, alpha0, alpha1),
        "effective_exposure_direct_tv": predict(model, external, np.arange(external.n)),
    }
    for name, prediction in predictions.items():
        row = joint.external_metrics(name, external.y, prediction)
        row["dataset"] = fit_dataset.name
        row["external_rows"] = external.n
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    metric_frames = []
    pair_frames = []
    parameter_frames = []
    external_frames = []
    for objective, target in joint.TARGET_COLUMNS.items():
        dataset = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            target,
        )
        external = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy(),
            target,
        )
        metrics, pairs, parameters = benchmark_dataset(
            dataset,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            args.maxiter,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        pair_frames.append(pairs)
        parameter_frames.append(parameters)
        external_frames.append(external_evaluation(dataset, external, args.maxiter, args.coarse_top_k))

    metrics = pd.concat(metric_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    external = pd.concat(external_frames, ignore_index=True)
    summary = pooled.summarize(metrics)
    metrics.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    pairs.to_csv(args.output_dir / "matched_pair_delta_metrics.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    external.to_csv(args.output_dir / "external_two_phase_heldout_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(pairs.groupby(["dataset", "model"]).mean(numeric_only=True).reset_index().to_string(index=False))
    print(external.to_string(index=False))
    print(f"Wrote direct-order TV benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
