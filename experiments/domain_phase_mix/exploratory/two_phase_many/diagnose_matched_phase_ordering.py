# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Diagnose ordering-effect calibration on matched one/two-phase pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr, spearmanr

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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "matched_phase_ordering_diagnostic_20260709"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def safe_correlation(fn, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(fn(x, y).statistic)


def pair_frame(dataset: pooled.Dataset, prediction: np.ndarray, model_name: str) -> pd.DataFrame:
    frame = dataset.frame.copy()
    frame["observed"] = dataset.y
    frame["predicted"] = prediction
    single = frame.loc[frame["policy_family"].eq("single_phase")].copy()
    two = frame.loc[frame["split"].eq("train") & frame["packet_panel"].eq("augmented_fit_panel")].copy()
    single = single.set_index("phase_correspondence_key")
    two = two.set_index("phase_correspondence_key")
    keys = single.index.intersection(two.index)
    return pd.DataFrame(
        {
            "dataset": dataset.name,
            "model": model_name,
            "phase_correspondence_key": keys,
            "single_run_name": single.loc[keys, "run_name"].to_numpy(),
            "two_run_name": two.loc[keys, "run_name"].to_numpy(),
            "single_observed": single.loc[keys, "observed"].to_numpy(dtype=float),
            "two_observed": two.loc[keys, "observed"].to_numpy(dtype=float),
            "single_predicted": single.loc[keys, "predicted"].to_numpy(dtype=float),
            "two_predicted": two.loc[keys, "predicted"].to_numpy(dtype=float),
        }
    ).assign(
        observed_ordering_delta=lambda value: value["two_observed"] - value["single_observed"],
        predicted_ordering_delta=lambda value: value["two_predicted"] - value["single_predicted"],
    )


def pair_metrics(pairs: pd.DataFrame, noise_sd: float) -> dict[str, float | int | str]:
    observed = pairs["observed_ordering_delta"].to_numpy(dtype=float)
    predicted = pairs["predicted_ordering_delta"].to_numpy(dtype=float)
    threshold = np.sqrt(2.0) * noise_sd
    reliable = np.abs(observed) > threshold
    return {
        "dataset": str(pairs["dataset"].iloc[0]),
        "model": str(pairs["model"].iloc[0]),
        "n_pairs": len(pairs),
        "noise_sd": noise_sd,
        "pair_noise_sd": threshold,
        "observed_delta_mean": float(np.mean(observed)),
        "observed_delta_median": float(np.median(observed)),
        "predicted_delta_mean": float(np.mean(predicted)),
        "delta_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "delta_mae": float(np.mean(np.abs(predicted - observed))),
        "delta_spearman": safe_correlation(spearmanr, observed, predicted),
        "delta_pearson": safe_correlation(pearsonr, observed, predicted),
        "sign_accuracy": float(np.mean(np.sign(observed) == np.sign(predicted))),
        "reliable_pair_count": int(np.sum(reliable)),
        "reliable_sign_accuracy": (
            float(np.mean(np.sign(observed[reliable]) == np.sign(predicted[reliable])))
            if np.any(reliable)
            else float("nan")
        ),
    }


def subset_metrics(
    dataset: pooled.Dataset,
    prediction: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
) -> dict[str, float | int | str]:
    subset = np.flatnonzero(dataset.frame["split"].eq("train") & dataset.frame["packet_panel"].eq("augmented_fit_panel"))
    target = dataset.y[subset]
    predicted = prediction[subset]
    residual = predicted - target
    fold_regrets = []
    subset_set = set(subset.tolist())
    for _train, test in folds:
        test_subset = np.asarray([index for index in test if index in subset_set], dtype=int)
        if len(test_subset) == 0:
            continue
        selected = test_subset[int(np.argmin(prediction[test_subset]))]
        fold_regrets.append(float(dataset.y[selected] - np.min(dataset.y[test_subset])))
    tail_count = max(5, int(np.ceil(0.15 * len(subset))))
    tail = np.argsort(predicted)[:tail_count]
    return {
        "dataset": dataset.name,
        "model": model_name,
        "n_rows": len(subset),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": safe_correlation(spearmanr, target, predicted),
        "fold_mean_regret_at_1": float(np.mean(fold_regrets)),
        "lower_tail_optimism": float(np.mean(np.maximum(-residual[tail], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
    }


def bootstrap_spearman_difference(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(target), size=len(target))
        left_rho = safe_correlation(spearmanr, target[indices], left[indices])
        right_rho = safe_correlation(spearmanr, target[indices], right[indices])
        if np.isfinite(left_rho) and np.isfinite(right_rho):
            differences.append(left_rho - right_rho)
    values = np.asarray(differences, dtype=float)
    return float(np.mean(values)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--coarse-top-k", type=int, default=1)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    pair_frames = []
    pair_metric_rows = []
    subset_rows = []
    oof_frames = []
    external_rows = []
    for objective, target_column in joint.TARGET_COLUMNS.items():
        fit_frame = frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy()
        external_frame = frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy()
        dataset = joint.dataset_from_frame(objective, fit_frame, target_column)
        external = joint.dataset_from_frame(objective, external_frame, target_column)
        folds = joint.grouped_folds(dataset.frame, seed=0, n_splits=5)
        alpha0, alpha1 = coverage.phase_fractions(dataset)
        predictions = {
            "effective_exposure": np.zeros(dataset.n, dtype=float),
            "effective_exposure_geometry": np.zeros(dataset.n, dtype=float),
            "split_saturation_penalty": np.zeros(dataset.n, dtype=float),
            "split_saturation_penalty_geometry": np.zeros(dataset.n, dtype=float),
            "separate_heads": np.zeros(dataset.n, dtype=float),
        }
        phase_configs = (
            coverage.FitConfig("effective_exposure", False, "effective_exposure"),
            coverage.FitConfig("effective_exposure_geometry", True, "effective_exposure", (0, 1)),
            coverage.FitConfig("split_saturation_penalty", False, "split_saturation_penalty"),
            coverage.FitConfig(
                "split_saturation_penalty_geometry",
                True,
                "split_saturation_penalty",
                (0, 1),
            ),
        )
        for train_idx, test_idx in folds:
            for config in phase_configs:
                model = coverage.fit_model(
                    dataset,
                    train_idx,
                    config,
                    linear_reg=coverage.dataset_linear_reg(dataset),
                    maxiter=args.maxiter,
                    coarse_top_k=args.coarse_top_k,
                )
                predictions[config.name][test_idx] = coverage.predict(model, dataset.weights[test_idx], alpha0, alpha1)
            separate_model = separate.fit_head(
                dataset,
                train_idx,
                use_coverage=False,
                alpha0=alpha0,
                alpha1=alpha1,
            )
            predictions["separate_heads"][test_idx] = separate.predict(separate_model, dataset, test_idx, alpha0, alpha1)

        proportional = dataset.frame.loc[
            dataset.frame["phase_correspondence_key"].eq("baseline_proportional") & dataset.frame["split"].eq("train"),
            target_column,
        ].to_numpy(dtype=float)
        noise_sd = float(np.std(proportional, ddof=1))
        for model_name, prediction in predictions.items():
            pairs = pair_frame(dataset, prediction, model_name)
            pair_frames.append(pairs)
            pair_metric_rows.append(pair_metrics(pairs, noise_sd))
            subset_rows.append(subset_metrics(dataset, prediction, folds, model_name))
            oof = dataset.frame[
                ["run_name", "split", "policy_family", "packet_panel", "phase_correspondence_key"]
            ].copy()
            oof["dataset"] = dataset.name
            oof["model"] = model_name
            oof["observed"] = dataset.y
            oof["predicted"] = prediction
            oof_frames.append(oof)

        external_predictions = {}
        for config in phase_configs:
            model = coverage.fit_model(
                dataset,
                np.arange(dataset.n),
                config,
                linear_reg=coverage.dataset_linear_reg(dataset),
                maxiter=args.maxiter,
                coarse_top_k=args.coarse_top_k,
            )
            external_predictions[config.name] = coverage.predict(model, external.weights, alpha0, alpha1)
        separate_model = separate.fit_head(
            dataset,
            np.arange(dataset.n),
            use_coverage=False,
            alpha0=alpha0,
            alpha1=alpha1,
        )
        external_predictions["separate_heads"] = separate.predict(
            separate_model, external, np.arange(external.n), alpha0, alpha1
        )
        reference_name = "effective_exposure_geometry"
        for model_name, prediction in external_predictions.items():
            if model_name == reference_name:
                continue
            mean_diff, lower, upper = bootstrap_spearman_difference(
                external.y,
                prediction,
                external_predictions[reference_name],
                args.n_bootstrap,
                seed=0,
            )
            external_rows.append(
                {
                    "dataset": dataset.name,
                    "left_model": model_name,
                    "right_model": reference_name,
                    "observed_spearman_difference": (
                        safe_correlation(spearmanr, external.y, prediction)
                        - safe_correlation(spearmanr, external.y, external_predictions[reference_name])
                    ),
                    "bootstrap_mean_difference": mean_diff,
                    "bootstrap_ci95_lower": lower,
                    "bootstrap_ci95_upper": upper,
                    "n_rows": external.n,
                }
            )

    pairs = pd.concat(pair_frames, ignore_index=True)
    pair_metrics_frame = pd.DataFrame(pair_metric_rows)
    subset = pd.DataFrame(subset_rows)
    oof = pd.concat(oof_frames, ignore_index=True)
    external = pd.DataFrame(external_rows)
    pairs.to_csv(args.output_dir / "matched_pair_deltas.csv", index=False)
    pair_metrics_frame.to_csv(args.output_dir / "matched_pair_delta_metrics.csv", index=False)
    subset.to_csv(args.output_dir / "two_phase_subset_oof_metrics.csv", index=False)
    oof.to_csv(args.output_dir / "joint_panel_oof_predictions.csv", index=False)
    external.to_csv(args.output_dir / "external_paired_bootstrap.csv", index=False)
    figure = px.scatter(
        pairs,
        x="observed_ordering_delta",
        y="predicted_ordering_delta",
        color="model",
        facet_col="dataset",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Matched ordering effect: observed versus OOF predicted",
    )
    figure.write_html(
        args.output_dir / "matched_pair_ordering_calibration.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    print(pair_metrics_frame.to_string(index=False))
    print(subset.to_string(index=False))
    print(external.to_string(index=False))
    print(f"Wrote matched ordering diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
