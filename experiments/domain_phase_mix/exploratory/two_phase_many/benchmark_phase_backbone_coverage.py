# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Compare mechanistic DSP phase backbones with mixture-geometry terms."""

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
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "phase_backbone_coverage_20260709"


def configs() -> tuple[coverage.FitConfig, ...]:
    return (
        coverage.FitConfig("canonical", False, "canonical"),
        coverage.FitConfig("canonical_geometry", True, "canonical", (0, 1)),
        coverage.FitConfig("effective_exposure", False, "effective_exposure"),
        coverage.FitConfig(
            "effective_exposure_geometry",
            True,
            "effective_exposure",
            (0, 1),
        ),
        coverage.FitConfig(
            "effective_exposure_tv",
            True,
            "effective_exposure",
            (0,),
        ),
        coverage.FitConfig("split_saturation_penalty", False, "split_saturation_penalty"),
        coverage.FitConfig(
            "split_saturation_penalty_geometry",
            True,
            "split_saturation_penalty",
            (0, 1),
        ),
        coverage.FitConfig(
            "split_saturation_penalty_tv",
            True,
            "split_saturation_penalty",
            (0,),
        ),
    )


def base_parameter_count(domain_count: int, variant_name: str) -> int:
    global_phase_parameters = 2 if variant_name == "split_saturation_penalty" else 1
    return 4 * domain_count + 1 + global_phase_parameters


def folds_for(dataset: pooled.Dataset, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if "phase_correspondence_key" in dataset.frame.columns:
        return joint.grouped_folds(dataset.frame, seed, n_splits)
    return pooled.dataset_folds(dataset, seed, n_splits)


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: tuple[coverage.FitConfig, ...],
    seeds: list[int],
    n_splits: int,
    maxiter_300m: int,
    maxiter_production: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha0, alpha1 = coverage.phase_fractions(dataset)
    rows = []
    parameter_rows = []
    for seed in seeds:
        folds = folds_for(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in model_configs}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in model_configs:
                model = coverage.fit_model(
                    dataset,
                    train_idx,
                    config,
                    linear_reg=coverage.dataset_linear_reg(dataset),
                    maxiter=(maxiter_production if dataset.name == "production_uncheatable" else maxiter_300m),
                    coarse_top_k=coarse_top_k,
                )
                predictions[config.name][test_idx] = coverage.predict(model, dataset.weights[test_idx], alpha0, alpha1)
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "gamma": float(model.base.params.get("gamma", np.nan)),
                        "gamma_saturation": float(model.base.params.get("gamma_saturation", np.nan)),
                        "gamma_penalty": float(model.base.params.get("gamma_penalty", np.nan)),
                        "theta_tv": float(model.coverage_coef[0]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_aggregate": float(model.coverage_coef[1]) if len(model.coverage_coef) else 0.0,
                    }
                )
        for config in model_configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["nominal_param_count"] = base_parameter_count(dataset.m, config.variant_name) + len(
                config.coverage_indices
            ) * int(config.use_coverage)
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(parameter_rows)


def external_evaluation(
    fit_dataset: pooled.Dataset,
    external: pooled.Dataset,
    model_configs: tuple[coverage.FitConfig, ...],
    maxiter_300m: int,
    maxiter_production: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    alpha0, alpha1 = coverage.phase_fractions(fit_dataset)
    rows = []
    for config in model_configs:
        model = coverage.fit_model(
            fit_dataset,
            np.arange(fit_dataset.n),
            config,
            linear_reg=coverage.dataset_linear_reg(fit_dataset),
            maxiter=(maxiter_production if fit_dataset.name == "production_uncheatable" else maxiter_300m),
            coarse_top_k=coarse_top_k,
        )
        prediction = coverage.predict(model, external.weights, alpha0, alpha1)
        row = joint.external_metrics(config.name, external.y, prediction)
        row["dataset"] = fit_dataset.name
        row["external_rows"] = external.n
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--datasets",
        default="300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument(
        "--models",
        default=",".join(config.name for config in configs()),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter-300m", type=int, default=8)
    parser.add_argument("--maxiter-production", type=int, default=0)
    parser.add_argument("--coarse-top-k", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    dataset_by_name = {}
    external_by_name = {}
    for objective, target in joint.TARGET_COLUMNS.items():
        dataset = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            target,
        )
        dataset_by_name[dataset.name] = dataset
        external_by_name[f"300m_{objective}"] = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy(),
            target,
        )
    production = pooled.load_production_dataset()
    dataset_by_name[production.name] = production
    selected_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected_names).difference(dataset_by_name))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    datasets = [dataset_by_name[name] for name in selected_names]
    config_by_name = {config.name: config for config in configs()}
    selected_models = [part.strip() for part in args.models.split(",") if part.strip()]
    unknown_models = sorted(set(selected_models).difference(config_by_name))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    model_configs = tuple(config_by_name[name] for name in selected_models)

    raw_frames = []
    parameter_frames = []
    external_frames = []
    for dataset in datasets:
        metrics, parameters = benchmark_dataset(
            dataset,
            model_configs,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            args.maxiter_300m,
            args.maxiter_production,
            args.coarse_top_k,
        )
        raw_frames.append(metrics)
        parameter_frames.append(parameters)
        if dataset.name in external_by_name:
            external_frames.append(
                external_evaluation(
                    dataset,
                    external_by_name[dataset.name],
                    model_configs,
                    args.maxiter_300m,
                    args.maxiter_production,
                    args.coarse_top_k,
                )
            )
    raw = pd.concat(raw_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    external = pd.concat(external_frames, ignore_index=True) if external_frames else pd.DataFrame()
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    external.to_csv(args.output_dir / "external_two_phase_heldout_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(external.to_string(index=False))
    print(f"Wrote backbone benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
