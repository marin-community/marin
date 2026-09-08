# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate"]
# ///
"""Matched, training-only alternatives for the frozen Delphi selection benchmark."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.linalg import solve
from scipy.spatial.distance import cdist, pdist

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark

RIDGES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
LENGTH_SCALES = (0.5, 1.0, 2.0)
MAX_KERNEL_DOF = 80
VARIANCE_RIDGE = 1.0


class Representation(StrEnum):
    SHARE = "share"
    SQRT_SHARE = "sqrt_share"
    LOG_EPOCH = "log_epoch"


class Estimator(StrEnum):
    RIDGE = "ridge"
    LINEAR_KERNEL = "linear_kernel"
    MATERN = "matern32"
    WLS = "loo_residual_wls"


class Pooling(StrEnum):
    MACRO = "macro"
    SHARED = "shared"
    TASKWISE = "taskwise"
    RANK3 = "rank3"


class Objective(StrEnum):
    RMSE = "rmse"
    REGRET = "regret_at_1"
    SHORTLIST = "best_of_5_regret"


@dataclasses.dataclass(frozen=True)
class Specification:
    name: str
    representation: Representation
    estimator: Estimator = Estimator.RIDGE
    pooling: Pooling = Pooling.MACRO
    objective: Objective = Objective.RMSE
    parent: str = "ridge_log_epoch_macro"
    change: str = ""


SPECS = (
    Specification("ridge_share_macro", Representation.SHARE, parent="ridge_sqrt_share_macro", change="share geometry"),
    Specification("ridge_sqrt_share_macro", Representation.SQRT_SHARE, change="square-root share geometry"),
    Specification(
        "ridge_log_epoch_macro", Representation.LOG_EPOCH, parent="wspu_direct_macro", change="signed log-epoch ridge"
    ),
    Specification(
        "ridge_log_epoch_shared",
        Representation.LOG_EPOCH,
        pooling=Pooling.SHARED,
        change="common task ridge; exact linearity control",
    ),
    Specification(
        "ridge_log_epoch_taskwise", Representation.LOG_EPOCH, pooling=Pooling.TASKWISE, change="per-task ridge selection"
    ),
    Specification(
        "ridge_log_epoch_rank3",
        Representation.LOG_EPOCH,
        pooling=Pooling.RANK3,
        parent="ridge_log_epoch_shared",
        change="rank-three training-response projection",
    ),
    Specification(
        "matern_share_macro",
        Representation.SHARE,
        Estimator.MATERN,
        parent="matern_sqrt_share_macro",
        change="kernel share geometry",
    ),
    Specification(
        "matern_sqrt_share_macro",
        Representation.SQRT_SHARE,
        Estimator.MATERN,
        parent="linear_kernel_sqrt_share_macro",
        change="nonadditive kernel; maximum 80 effective degrees of freedom",
    ),
    Specification(
        "linear_kernel_sqrt_share_macro",
        Representation.SQRT_SHARE,
        Estimator.LINEAR_KERNEL,
        parent="ridge_sqrt_share_macro",
        change="unstandardized geometry; additive control for the Matern kernel",
    ),
    Specification(
        "ridge_log_epoch_wls",
        Representation.LOG_EPOCH,
        Estimator.WLS,
        change="training-only leave-one-out residual variance weights",
    ),
    Specification(
        "ridge_log_epoch_regret", Representation.LOG_EPOCH, objective=Objective.REGRET, change="inner regret selection"
    ),
    Specification(
        "ridge_log_epoch_shortlist",
        Representation.LOG_EPOCH,
        objective=Objective.SHORTLIST,
        change="inner best-of-five selection",
    ),
)


@dataclasses.dataclass(frozen=True)
class Prediction:
    values: np.ndarray
    standard_error: np.ndarray
    dof: float
    effective_rows: float
    parameters: dict


def design(weights: np.ndarray, exposures: np.ndarray, representation: Representation) -> np.ndarray:
    if representation == Representation.SHARE:
        return weights
    if representation == Representation.SQRT_SHARE:
        return np.sqrt(weights)
    if representation == Representation.LOG_EPOCH:
        return np.log1p(exposures)
    raise ValueError(representation)


def variance_descriptors(weights: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1)
    return np.column_stack(
        [entropy, weights.max(axis=1), np.square(weights).sum(axis=1), np.log1p(exposures.max(axis=1))]
    )


def precision_weights(matrix: np.ndarray, response: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
    """Feasible WLS with a four-covariate variance model learned from training LOO residuals."""
    centered = matrix - matrix.mean(axis=0)
    scaled = centered / np.maximum(matrix.std(axis=0), 1e-8)
    inverse = solve(
        scaled.T @ scaled + VARIANCE_RIDGE * np.eye(matrix.shape[1]), np.eye(matrix.shape[1]), assume_a="pos"
    )
    fitted = response.mean() + scaled @ inverse @ scaled.T @ (response - response.mean())
    leverage = 1 / len(matrix) + np.einsum("ij,jk,ik->i", scaled, inverse, scaled)
    squared = np.square((response - fitted) / np.maximum(1 - leverage, 0.05))
    log_variance = np.log(squared + 0.1 * squared.mean() + 1e-12)
    descriptors = (descriptors - descriptors.mean(axis=0)) / np.maximum(descriptors.std(axis=0), 1e-8)
    coefficient = solve(
        descriptors.T @ descriptors + VARIANCE_RIDGE * np.eye(descriptors.shape[1]),
        descriptors.T @ (log_variance - log_variance.mean()),
        assume_a="pos",
    )
    precision = np.exp(np.clip(-descriptors @ coefficient, -np.log(4), np.log(4)))
    return precision / precision.mean()


def ridge_prediction(
    matrix: np.ndarray,
    response: np.ndarray,
    query: np.ndarray,
    ridge: float,
    precision: np.ndarray,
    aggregate_weights: np.ndarray,
) -> Prediction:
    """Signed ridge with a free intercept and training-only standardization."""
    mass = precision.sum()
    mean_x = np.average(matrix, axis=0, weights=precision)
    scale_x = np.sqrt(np.average(np.square(matrix - mean_x), axis=0, weights=precision))
    scale_x = np.maximum(scale_x, 1e-8)
    x = (matrix - mean_x) / scale_x
    q = (query - mean_x) / scale_x
    mean_y = np.average(response, axis=0, weights=precision)
    gram = x.T @ (precision[:, None] * x)
    inverse = solve(gram + ridge * np.eye(matrix.shape[1]), np.eye(matrix.shape[1]), assume_a="pos")
    beta = inverse @ x.T @ (precision[:, None] * (response - mean_y))
    values = mean_y + q @ beta
    dof = float(1 + np.trace(gram @ inverse))
    residual = (response - mean_y - x @ beta) @ aggregate_weights
    variance = float(np.dot(precision, residual**2) / max(mass - dof, 1))
    error = np.sqrt(np.maximum(variance * (1 / mass + np.einsum("ij,jk,ik->i", q, inverse, q)), 0))
    return Prediction(
        values,
        error,
        dof,
        float(mass**2 / np.square(precision).sum()),
        {
            "coefficient": beta.tolist(),
            "intercept": mean_y.tolist(),
            "feature_mean": mean_x.tolist(),
            "feature_scale": scale_x.tolist(),
        },
    )


def matern_prediction(
    matrix: np.ndarray,
    response: np.ndarray,
    query: np.ndarray,
    ridge: float,
    length_scale: float,
    aggregate_weights: np.ndarray,
    estimator: Estimator,
) -> Prediction:
    bandwidth = float(np.median(pdist(matrix))) * length_scale
    if bandwidth <= 0:
        raise ValueError("Kernel training coordinates have no variation")
    if estimator == Estimator.LINEAR_KERNEL:
        kernel = matrix @ matrix.T
        cross = query @ matrix.T
        query_diagonal = np.square(query).sum(axis=1)
    else:
        distance = np.sqrt(3) * cdist(matrix, matrix) / bandwidth
        kernel = (1 + distance) * np.exp(-distance)
        distance = np.sqrt(3) * cdist(query, matrix) / bandwidth
        cross = (1 + distance) * np.exp(-distance)
        query_diagonal = np.ones(len(query))
    column_mean = kernel.mean(axis=0)
    total_mean = kernel.mean()
    centered = kernel - column_mean[None] - column_mean[:, None] + total_mean
    centered_cross = cross - cross.mean(axis=1)[:, None] - column_mean[None] + total_mean
    inverse = solve(centered + ridge * np.eye(len(matrix)), np.eye(len(matrix)), assume_a="pos")
    alpha = inverse @ (response - response.mean(axis=0))
    values = response.mean(axis=0) + centered_cross @ alpha
    dof = float(1 + np.trace(centered @ inverse))
    residual = (response - response.mean(axis=0) - centered @ alpha) @ aggregate_weights
    variance = float(np.square(residual).sum() / max(len(matrix) - dof, 1))
    leverage = (
        query_diagonal
        - 2 * cross.mean(axis=1)
        + total_mean
        - np.einsum("ij,jk,ik->i", centered_cross, inverse, centered_cross)
    )
    error = np.sqrt(np.maximum(variance * (1 / len(matrix) + leverage / ridge), 0))
    return Prediction(values, error, dof, float(len(matrix)), {"bandwidth": bandwidth})


def predict(
    spec: Specification,
    x: np.ndarray,
    outcomes: np.ndarray,
    descriptors: np.ndarray,
    aggregate_weights: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    ridge: float,
    length_scale: float,
) -> Prediction:
    response = outcomes[train]
    weights = aggregate_weights
    if spec.pooling == Pooling.MACRO:
        response = (response @ weights)[:, None]
        weights = np.ones(1)
    elif spec.pooling == Pooling.RANK3:
        mean = response.mean(axis=0)
        scale = np.maximum(response.std(axis=0), 1e-8)
        standardized = (response - mean) / scale
        _, _, vectors = np.linalg.svd(standardized, full_matrices=False)
        response = mean + (standardized @ vectors[:3].T @ vectors[:3]) * scale
    if spec.estimator in (Estimator.MATERN, Estimator.LINEAR_KERNEL):
        return matern_prediction(x[train], response, query, ridge, length_scale, weights, spec.estimator)
    precision = np.ones(len(train))
    if spec.estimator == Estimator.WLS:
        precision = precision_weights(x[train], outcomes[train] @ aggregate_weights, descriptors[train])
    return ridge_prediction(x[train], response, query, ridge, precision, weights)


def fit_alternative(output: Path, spec: Specification, target: str, fold: int, repeat: int, fingerprint: str) -> str:
    path = output / "alternative_shards" / spec.name / target / f"r{repeat}_f{fold}.npz"
    if path.exists() and str(benchmark.read_npz(path)["fingerprint"]) == fingerprint:
        return "cached"
    start = time.monotonic()
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
    split = benchmark.partition(output, fold, repeat)
    x = design(data["weights"], data["exposures"], spec.representation)
    bank_x = design(bank["weights"], bank["exposures"], spec.representation)
    descriptors = variance_descriptors(data["weights"], data["exposures"])
    outcomes = data[f"{target}_outcomes"]
    weights = data[f"{target}_aggregation_weights"]
    aggregate = outcomes @ weights
    candidates = [
        (ridge, length)
        for ridge in RIDGES
        for length in (LENGTH_SCALES if spec.estimator == Estimator.MATERN else (1.0,))
    ]
    tables = []
    choices = []
    for ridge, length in candidates:
        errors = []
        regrets = []
        task_errors = []
        valid = True
        for train, test in split.inner:
            fit = predict(spec, x, outcomes, descriptors, weights, train, x[test], ridge, length)
            if spec.estimator == Estimator.MATERN and fit.dof > MAX_KERNEL_DOF:
                valid = False
                break
            prediction = fit.values[:, 0] if spec.pooling == Pooling.MACRO else fit.values @ weights
            errors.extend(np.square(prediction - aggregate[test]))
            task_errors.append(np.mean(np.square(fit.values - outcomes[test]), axis=0))
            order = np.argsort(prediction, kind="stable")
            k = 5 if spec.objective == Objective.SHORTLIST else 1
            regrets.append(float(aggregate[test][order[:k]].min() - aggregate[test].min()))
        if not valid:
            continue
        rmse = float(np.sqrt(np.mean(errors)))
        score = rmse if spec.objective == Objective.RMSE else float(np.mean(regrets))
        tables.append({"ridge": ridge, "length": length, "rmse": rmse, "score": score})
        choices.append((score, rmse, -ridge, length, np.mean(task_errors, axis=0)))
    if not choices:
        raise ValueError(f"No admissible candidate for {spec.name}")
    query = np.vstack([x[split.test], bank_x, x[split.train]])
    if spec.pooling == Pooling.TASKWISE:
        error_matrix = np.stack([choice[-1] for choice in choices])
        indices = error_matrix.argmin(axis=0)
        predictions = []
        dofs = []
        for task, index in enumerate(indices):
            choice = choices[index]
            fit = predict(
                spec, x, outcomes[:, task : task + 1], descriptors, np.ones(1), split.train, query, -choice[2], choice[3]
            )
            predictions.append(fit.values[:, 0])
            dofs.append(fit.dof)
        values = np.stack(predictions, axis=1)
        prediction = values @ weights
        error = np.zeros(len(query))
        dof = float(np.mean(dofs))
        selected = {"task_ridges": [-choices[i][2] for i in indices]}
        effective_rows = float(len(split.train))
        parameters = {}
    else:
        choice = min(choices, key=lambda item: item[:4])
        fit = predict(spec, x, outcomes, descriptors, weights, split.train, query, -choice[2], choice[3])
        if spec.estimator == Estimator.MATERN and fit.dof > MAX_KERNEL_DOF:
            raise ValueError("Final fit exceeds frozen kernel complexity bound")
        prediction = fit.values[:, 0] if spec.pooling == Pooling.MACRO else fit.values @ weights
        error = fit.standard_error
        dof = fit.dof
        effective_rows = fit.effective_rows
        selected = {"ridge": -choice[2], "length": choice[3]}
        parameters = fit.parameters
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite predictions from {spec.name}")
    test_end = len(split.test)
    bank_end = test_end + len(bank_x)
    benchmark.harness.atomic_save(
        path,
        {
            "fingerprint": fingerprint,
            "prediction": prediction[:test_end],
            "bank_prediction": prediction[test_end:bank_end],
            "train_prediction": prediction[bank_end:],
            "standard_error": error[:test_end],
            "bank_standard_error": error[test_end:bank_end],
            "test": split.test,
            "train": split.train,
            "dof": dof,
            "effective_rows": effective_rows,
            "selected_json": json.dumps(selected),
            "cv_json": json.dumps(tables),
            "parameters_json": json.dumps(parameters),
            "elapsed": time.monotonic() - start,
        },
    )
    return "fitted"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=benchmark.DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, choices=(1, 5), default=1)
    args = parser.parse_args()
    fingerprint = benchmark.source_fingerprint(args.output_dir, (Path(__file__),))
    benchmark.write_json(
        args.output_dir / "alternative_protocol.json",
        {
            "specifications": [dataclasses.asdict(spec) for spec in SPECS],
            "ridge_grid": RIDGES,
            "kernel_length_scales": LENGTH_SCALES,
            "kernel_dof_limit": MAX_KERNEL_DOF,
            "variance_ridge": VARIANCE_RIDGE,
            "fingerprint": fingerprint,
            "selection_ties": "score, aggregate RMSE, descending ridge, ascending length scale",
        },
    )
    tasks = [
        (spec, target, fold, repeat)
        for spec in SPECS
        for target in benchmark.TARGETS
        for repeat in range(args.repeats)
        for fold in ((-1, 0, 1, 2, 3, 4) if repeat == 0 else (0, 1, 2, 3, 4))
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        counts = Parallel(n_jobs=args.workers, verbose=10)(
            delayed(fit_alternative)(args.output_dir, *task, fingerprint) for task in tasks
        )
    print(pd.Series(counts).value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
