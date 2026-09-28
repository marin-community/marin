# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark a phase-invariant DSP backbone plus a low-dimensional order residual.

The aggregate backbone is fit only on tied schedules when matched one/two-phase
rows are available. For each two-phase schedule, the order residual is fit to
the directly observed paired difference from its tied counterpart. This keeps
aggregate quality and phase ordering statistically separate.

The order residual uses up to four pooled features, all zero for tied phases:

* phase total variation;
* phase-1 concentration relative to aggregate concentration;
* late placement of buckets with remaining DSP benefit;
* late placement of buckets under DSP overexposure pressure.

Only four global nonnegative coefficients are added regardless of the number
of buckets. The bounded late/early contrast prevents arbitrarily large credit
from extreme schedules.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

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
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import (  # noqa: E402
    dsp_exact as dsp,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "aggregate_order_dsp_20260710"
SCALE_FLOOR = 1e-10
BASE_CONFIG = geometry.FitConfig(
    "aggregate_no_phase_hhi",
    True,
    "no_phase",
    (1,),
)
NOISE_SD = {
    "300m_uncheatable": 0.0011270969148995812,
    "300m_table9": 0.003325782675083218,
}


@dataclass(frozen=True)
class OrderConfig:
    name: str
    feature_names: tuple[str, ...]
    l2: float = 0.1


@dataclass(frozen=True)
class AggregateOrderModel:
    base: geometry.CoverageModel
    config: OrderConfig
    feature_scale: np.ndarray
    order_coef: np.ndarray
    alpha0: float
    alpha1: float


def configs() -> tuple[OrderConfig, ...]:
    return (
        OrderConfig("aggregate_only", ()),
        OrderConfig("aggregate_geometry_order", ("phase_tv", "phase1_concentration")),
        OrderConfig(
            "aggregate_value_room_order",
            ("phase_tv", "late_benefit", "late_penalty"),
        ),
        OrderConfig(
            "aggregate_value_room_geometry_order",
            ("phase_tv", "phase1_concentration", "late_benefit", "late_penalty"),
        ),
    )


def tied_weights(weights: np.ndarray, alpha0: float, alpha1: float) -> np.ndarray:
    aggregate = alpha0 * weights[:, 0, :] + alpha1 * weights[:, 1, :]
    return np.stack([aggregate, aggregate], axis=1)


def tied_dataset(dataset: pooled.Dataset, alpha0: float, alpha1: float) -> pooled.Dataset:
    return pooled.Dataset(
        name=dataset.name,
        frame=dataset.frame,
        y=dataset.y,
        weights=tied_weights(dataset.weights, alpha0, alpha1),
        c0=dataset.c0,
        c1=dataset.c1,
        domain_names=dataset.domain_names,
    )


def order_feature_frame(
    base: geometry.CoverageModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> pd.DataFrame:
    weights = dataset.weights[indices]
    w0 = weights[:, 0, :]
    w1 = weights[:, 1, :]
    aggregate_weights = alpha0 * w0 + alpha1 * w1
    e0 = w0 * dataset.c0[None, :]
    e1 = w1 * dataset.c1[None, :]
    aggregate_exposure = aggregate_weights * (dataset.c0 + dataset.c1)[None, :]
    contrast = np.tanh(np.log1p(e1 / alpha1) - np.log1p(e0 / alpha0))

    rho = np.asarray(base.base.params["rho"], dtype=float)[None, :]
    remaining_benefit = base.base.benefit_coef[None, :] * rho * np.exp(-rho * aggregate_exposure)
    tau = np.asarray(base.base.params["tau"], dtype=float)[None, :]
    penalty_argument = np.log1p(aggregate_exposure) - tau
    penalty_softplus = dsp.softplus(penalty_argument)
    penalty_sigmoid = 1.0 / (1.0 + np.exp(-penalty_argument))
    penalty_pressure = (
        base.base.penalty_coef[None, :] * 2.0 * penalty_softplus * penalty_sigmoid / (1.0 + aggregate_exposure)
    )

    return pd.DataFrame(
        {
            "phase_tv": 0.5 * np.abs(w0 - w1).sum(axis=1),
            "phase1_concentration": np.sum(w1**2, axis=1) - np.sum(aggregate_weights**2, axis=1),
            # A positive coefficient gives useful late placement negative loss.
            "late_benefit": -np.sum(contrast * remaining_benefit, axis=1),
            # A positive coefficient charges late placement under repetition pressure.
            "late_penalty": np.sum(contrast * penalty_pressure, axis=1),
        },
        index=indices,
    )


def fit_order_head(
    design: np.ndarray,
    target: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, np.ndarray]:
    if design.shape[1] == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), SCALE_FLOOR)
    normalized = design / scale[None, :]
    if l2 > 0.0:
        normalized = np.vstack([normalized, np.sqrt(l2) * np.eye(design.shape[1])])
        target = np.concatenate([target, np.zeros(design.shape[1], dtype=float)])
    coef, _residual = nnls(normalized, target)
    return scale, np.asarray(coef, dtype=float)


def matched_order_training_rows(
    dataset: pooled.Dataset,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset.frame.copy()
    frame["_row_index"] = np.arange(dataset.n)
    frame["_target"] = dataset.y
    in_train = frame["_row_index"].isin(set(train_indices.tolist()))
    single = frame.loc[in_train & frame["policy_family"].eq("single_phase")]
    two = frame.loc[in_train & frame["split"].eq("train") & frame["packet_panel"].eq("augmented_fit_panel")]
    single_target = single.groupby("phase_correspondence_key")["_target"].mean()
    paired = two.loc[two["phase_correspondence_key"].isin(single_target.index)].copy()
    delta = paired["_target"].to_numpy(dtype=float) - paired["phase_correspondence_key"].map(single_target).to_numpy(
        dtype=float
    )
    return paired["_row_index"].to_numpy(dtype=int), delta


def fit_model(
    dataset: pooled.Dataset,
    train_indices: np.ndarray,
    config: OrderConfig,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> AggregateOrderModel:
    alpha0, alpha1 = geometry.phase_fractions(dataset)
    tied = tied_dataset(dataset, alpha0, alpha1)
    has_pairs = "policy_family" in dataset.frame.columns
    if has_pairs:
        single = dataset.frame["policy_family"].eq("single_phase").to_numpy()
        base_indices = train_indices[single[train_indices]]
        if len(base_indices) < 20:
            raise ValueError("Matched aggregate fit has too few tied training rows")
    else:
        base_indices = train_indices
    base = geometry.fit_model(
        tied,
        base_indices,
        BASE_CONFIG,
        linear_reg=geometry.dataset_linear_reg(dataset),
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )

    if not config.feature_names:
        return AggregateOrderModel(
            base=base,
            config=config,
            feature_scale=np.zeros(0, dtype=float),
            order_coef=np.zeros(0, dtype=float),
            alpha0=alpha0,
            alpha1=alpha1,
        )

    if has_pairs:
        order_indices, order_target = matched_order_training_rows(dataset, train_indices)
    else:
        order_indices = train_indices
        aggregate_prediction = geometry.predict(
            base,
            tied.weights[order_indices],
            alpha0,
            alpha1,
        )
        order_target = dataset.y[order_indices] - aggregate_prediction
    features = (
        order_feature_frame(base, dataset, order_indices, alpha0, alpha1)
        .loc[:, config.feature_names]
        .to_numpy(dtype=float)
    )
    feature_scale, order_coef = fit_order_head(features, order_target, config.l2)
    return AggregateOrderModel(
        base=base,
        config=config,
        feature_scale=feature_scale,
        order_coef=order_coef,
        alpha0=alpha0,
        alpha1=alpha1,
    )


def predict(
    model: AggregateOrderModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
) -> np.ndarray:
    tied = tied_weights(dataset.weights[indices], model.alpha0, model.alpha1)
    prediction = geometry.predict(
        model.base,
        tied,
        model.alpha0,
        model.alpha1,
    )
    if model.config.feature_names:
        features = (
            order_feature_frame(
                model.base,
                dataset,
                indices,
                model.alpha0,
                model.alpha1,
            )
            .loc[:, model.config.feature_names]
            .to_numpy(dtype=float)
        )
        prediction = prediction + (features / model.feature_scale[None, :]) @ model.order_coef
    return prediction


def folds_for(
    dataset: pooled.Dataset,
    seed: int,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if "phase_correspondence_key" in dataset.frame.columns:
        return joint.grouped_folds(dataset.frame, seed, n_splits)
    return pooled.dataset_folds(dataset, seed, n_splits)


def benchmark_dataset(
    dataset: pooled.Dataset,
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    parameter_rows = []
    prediction_frames = []
    pair_metric_rows = []
    for seed in seeds:
        folds = folds_for(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in configs()}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in configs():
                model = fit_model(
                    dataset,
                    train_idx,
                    config,
                    maxiter=maxiter,
                    coarse_top_k=coarse_top_k,
                )
                predictions[config.name][test_idx] = predict(model, dataset, test_idx)
                row: dict[str, float | int | str] = {
                    "dataset": dataset.name,
                    "model": config.name,
                    "seed": seed,
                    "fold": fold_id,
                    "gamma": float(model.base.base.params.get("gamma", np.nan)),
                    "theta_hhi_aggregate": float(model.base.coverage_coef[1]),
                }
                for name, coef in zip(config.feature_names, model.order_coef, strict=True):
                    row[f"theta_{name}"] = float(coef)
                parameter_rows.append(row)
        for config in configs():
            prediction = predictions[config.name]
            metric = asdict(pooled.metrics(dataset, config.name, seed, prediction, folds))
            metric["nominal_param_count"] = 4 * dataset.m + 2 + len(config.feature_names)
            metric_rows.append(metric)
            prediction_frame = dataset.frame[
                [
                    column
                    for column in (
                        "run_name",
                        "policy_family",
                        "split",
                        "packet_panel",
                        "phase_correspondence_key",
                    )
                    if column in dataset.frame.columns
                ]
            ].copy()
            prediction_frame["dataset"] = dataset.name
            prediction_frame["model"] = config.name
            prediction_frame["seed"] = seed
            prediction_frame["observed"] = dataset.y
            prediction_frame["predicted"] = prediction
            prediction_frames.append(prediction_frame)
            if dataset.name in NOISE_SD:
                pairs = ordering.pair_frame(dataset, prediction, config.name)
                pair_metric = ordering.pair_metrics(pairs, NOISE_SD[dataset.name])
                pair_metric["seed"] = seed
                pair_metric_rows.append(pair_metric)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(parameter_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(pair_metric_rows),
    )


def external_evaluation(
    fit_dataset: pooled.Dataset,
    external: pooled.Dataset,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    rows = []
    train_indices = np.arange(fit_dataset.n)
    for config in configs():
        model = fit_model(
            fit_dataset,
            train_indices,
            config,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        prediction = predict(model, external, np.arange(external.n))
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
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--datasets", default="300m_uncheatable,300m_table9,production_uncheatable")
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

    metric_frames = []
    parameter_frames = []
    prediction_frames = []
    pair_metric_frames = []
    external_frames = []
    for dataset in datasets:
        maxiter = args.maxiter_production if dataset.name == "production_uncheatable" else args.maxiter_300m
        metrics, parameters, predictions, pair_metrics = benchmark_dataset(
            dataset,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            maxiter,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
        prediction_frames.append(predictions)
        if not pair_metrics.empty:
            pair_metric_frames.append(pair_metrics)
        if dataset.name in external_by_name:
            external_frames.append(
                external_evaluation(
                    dataset,
                    external_by_name[dataset.name],
                    maxiter,
                    args.coarse_top_k,
                )
            )

    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(metrics)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pair_metrics = pd.concat(pair_metric_frames, ignore_index=True)
    external = pd.concat(external_frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pair_metrics.to_csv(args.output_dir / "matched_pair_delta_metrics.csv", index=False)
    external.to_csv(args.output_dir / "external_two_phase_heldout_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(
        pair_metrics.groupby(["dataset", "model"])[["delta_rmse", "delta_spearman", "sign_accuracy"]].mean().to_string()
    )
    print(external.to_string(index=False))
    print(f"Wrote aggregate/order DSP benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
