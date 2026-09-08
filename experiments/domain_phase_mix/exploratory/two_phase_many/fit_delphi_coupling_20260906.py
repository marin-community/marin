# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "joblib", "scikit-learn", "tabulate"]
# ///
"""Offline taskwise coupling ablation with identical anchored bucket main effects."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
from enum import StrEnum
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.linalg import null_space, solve
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as primitives,
)

DEFAULT_OUTPUT = benchmark.REFERENCE / "delphi_coupling_followup_20260906" / "coupling"
RIDGES = (0.1, 1.0, 10.0)
EXP_LIMIT = 50.0
WEIBULL_RATE = 1.0
WEIBULL_POWER = 1.0
HARM_THRESHOLD = float(np.log(2.0))


class Basis(StrEnum):
    SHARES = "shares"
    WEIBULL = "fixed_weibull"


class Link(StrEnum):
    IDENTITY = "identity"
    ADDITIVE = "additive_exp"
    COUPLED = "coupled_exp"


@dataclasses.dataclass(frozen=True)
class Specification:
    basis: Basis
    link: Link

    @property
    def name(self) -> str:
        return f"{self.basis}_{self.link}"


SPECS = tuple(Specification(basis, link) for basis in Basis for link in Link)


@dataclasses.dataclass(frozen=True)
class Design:
    training: np.ndarray
    query: np.ndarray
    projection: np.ndarray
    anchor: np.ndarray
    scale: np.ndarray
    basis: Basis


@dataclasses.dataclass(frozen=True)
class TaskFit:
    parameters: np.ndarray
    outcome_scale: float
    objective: float
    success: bool
    gradient_norm: float
    iterations: int
    start_objectives: tuple[float, ...]
    start_successes: tuple[bool, ...]
    message: str


def bucket_basis(weights: np.ndarray, inventory: np.ndarray, basis: Basis) -> np.ndarray:
    if basis == Basis.SHARES:
        return weights[:, :, None]
    exposure = weights * inventory
    return np.stack(
        [
            -primitives.weibull_response(exposure, WEIBULL_RATE, WEIBULL_POWER),
            primitives.softplus_harm(exposure, HARM_THRESHOLD),
        ],
        axis=-1,
    )


def design_matrix(weights: np.ndarray, query: np.ndarray, inventory: np.ndarray, basis: Basis) -> Design:
    """Center bucket features at a training-only feasible anchor and fix the share gauge."""
    anchor = weights.mean(axis=0)
    reference = bucket_basis(anchor[None], inventory, basis)
    training = bucket_basis(weights, inventory, basis)
    scale = np.maximum(training.std(axis=0), 1e-8)
    training = (training - reference) / scale
    transformed_query = (bucket_basis(query, inventory, basis) - reference) / scale
    width = int(np.prod(training.shape[1:]))
    projection = null_space(scale.reshape(1, -1)) if basis == Basis.SHARES else np.eye(width)
    return Design(training, transformed_query, projection, anchor, scale, basis)


def response_jacobian(
    parameters: np.ndarray, matrix: np.ndarray, projection: np.ndarray, link: Link
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return normalized BPB responses and exact parameter derivatives for each anchored law."""
    flat = matrix.reshape(len(matrix), -1)
    coefficient = (projection @ parameters[1:]).reshape(matrix.shape[1:])
    scores = np.einsum("nbk,bk->nb", matrix, coefficient)
    if link == Link.IDENTITY:
        value = parameters[0] + scores.sum(axis=1)
        jacobian = np.column_stack([np.ones(len(matrix)), flat @ projection])
        return value, jacobian, 0
    if link == Link.COUPLED:
        logits = parameters[0] + scores.sum(axis=1)
        value = np.exp(np.clip(logits, -EXP_LIMIT, EXP_LIMIT))
        active = np.abs(logits) < EXP_LIMIT
        jacobian = value[:, None] * np.column_stack([np.ones(len(matrix)), flat @ projection])
        jacobian *= active[:, None]
        return value, jacobian, int((~active).sum())
    active = np.abs(scores) < EXP_LIMIT
    exponential = np.exp(np.clip(scores, -EXP_LIMIT, EXP_LIMIT))
    amplitude = np.exp(parameters[0])
    value = amplitude * (1 + np.expm1(np.clip(scores, -EXP_LIMIT, EXP_LIMIT)).sum(axis=1))
    slope = (amplitude * exponential * active)[:, :, None] * matrix
    jacobian = np.column_stack([value, slope.reshape(len(matrix), -1) @ projection])
    return value, jacobian, int((~active).any(axis=1).sum())


def linear_parameters(matrix: np.ndarray, response: np.ndarray, ridge: float, basis: Basis) -> np.ndarray:
    if basis == Basis.WEIBULL:
        specification = primitives.HeadSpec(kind=primitives.HeadKind.NNLS, scale_columns=False)
        design = primitives.Design(matrix, np.ones(matrix.shape[1]), tuple(map(str, range(matrix.shape[1]))))
        head = primitives.fit_head(design, response, ridge, specification)
        return np.r_[head.intercept, head.coefficients]
    center = matrix.mean(axis=0)
    centered = matrix - center
    coefficient = solve(
        centered.T @ centered + ridge * np.eye(matrix.shape[1]),
        centered.T @ (response - response.mean()),
        assume_a="pos",
    )
    return np.r_[response.mean() - center @ coefficient, coefficient]


def fit_task(design: Design, response: np.ndarray, link: Link, ridge: float) -> TaskFit:
    """Fit raw-BPB least squares and a common relative-amplitude ridge penalty."""
    if np.any(response <= 0):
        raise ValueError("Coupling benchmark requires positive BPB observations")
    outcome_scale = float(response.mean())
    target = response / outcome_scale
    linear_matrix = design.training.reshape(len(response), -1) @ design.projection

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        prediction, jacobian, _ = response_jacobian(parameters, design.training, design.projection, link)
        error = prediction - target
        value = float(error @ error + ridge * (parameters[1:] @ parameters[1:]))
        gradient = 2 * jacobian.T @ error
        gradient[1:] += 2 * ridge * parameters[1:]
        return value, gradient

    if link == Link.IDENTITY:
        parameters = linear_parameters(linear_matrix, target, ridge, design.basis)
        value, gradient = objective(parameters)
        projected = gradient.copy()
        if design.basis == Basis.WEIBULL:
            projected[1:][(parameters[1:] <= 1e-8) & (projected[1:] > 0)] = 0
        return TaskFit(
            parameters, outcome_scale, value, True, float(np.abs(projected).max()), 1, (value,), (True,), "linear solve"
        )

    zero_start = np.zeros(linear_matrix.shape[1] + 1)
    log_start = linear_parameters(linear_matrix, np.log(target), ridge, design.basis)
    bounds = [(-20.0, 20.0)] + [(0.0, None) if design.basis == Basis.WEIBULL else (None, None)] * linear_matrix.shape[1]
    fits = [
        minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
        )
        for start in (zero_start, log_start)
    ]
    fit = min(fits, key=lambda result: float(result.fun))
    gradient = np.asarray(fit.jac).copy()
    if design.basis == Basis.WEIBULL:
        gradient[1:][(fit.x[1:] <= 1e-8) & (gradient[1:] > 0)] = 0
    return TaskFit(
        fit.x,
        outcome_scale,
        float(fit.fun),
        bool(fit.success),
        float(np.abs(gradient).max()),
        int(fit.nit),
        tuple(float(result.fun) for result in fits),
        tuple(bool(result.success) for result in fits),
        str(fit.message),
    )


def local_dof(fit: TaskFit, design: Design, link: Link, ridge: float) -> float:
    _, jacobian, _ = response_jacobian(fit.parameters, design.training, design.projection, link)
    active = (
        np.r_[True, fit.parameters[1:] > 1e-8] if design.basis == Basis.WEIBULL else np.ones(len(fit.parameters), bool)
    )
    jacobian = jacobian[:, active]
    gram = jacobian.T @ jacobian
    penalty = np.diag(np.r_[0.0, np.full(gram.shape[0] - 1, ridge)])
    return float(np.trace(solve(gram + penalty, gram, assume_a="pos")))


def fit_matrix(design: Design, responses: np.ndarray, link: Link, ridge: float) -> tuple[np.ndarray, list[TaskFit]]:
    fits = [fit_task(design, responses[:, column], link, ridge) for column in range(responses.shape[1])]
    values = np.column_stack(
        [response_jacobian(fit.parameters, design.query, design.projection, link)[0] * fit.outcome_scale for fit in fits]
    )
    return values, fits


def fit_specification(output: Path, spec: Specification, target: str, fold: int, repeat: int, fingerprint: str) -> str:
    path = output / "shards" / spec.name / target / f"r{repeat}_f{fold}.npz"
    if path.exists() and str(benchmark.read_npz(path)["fingerprint"]) == fingerprint:
        return "cached"
    started = time.monotonic()
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
    split = benchmark.partition(output, fold, repeat)
    outcomes = data[f"{target}_outcomes"]
    aggregate_weights = data[f"{target}_aggregation_weights"]
    table = []
    for ridge in RIDGES:
        errors = []
        all_fits = []
        for train, test in split.inner:
            design = design_matrix(data["weights"][train], data["weights"][test], data["inventory"], spec.basis)
            values, fits = fit_matrix(design, outcomes[train], spec.link, ridge)
            errors.extend(((values - outcomes[test]) @ aggregate_weights) ** 2)
            all_fits.extend(fits)
        table.append(
            {
                "ridge": ridge,
                "aggregate_rmse": float(np.sqrt(np.mean(errors))),
                "unsuccessful": sum(not fit.success for fit in all_fits),
                "maximum_gradient": max(fit.gradient_norm for fit in all_fits),
                "fits": len(all_fits),
            }
        )
    choice = min(table, key=lambda row: (row["aggregate_rmse"], -row["ridge"]))
    ridge = float(choice["ridge"])
    query = np.vstack([data["weights"][split.test], bank["weights"], data["weights"][split.train]])
    design = design_matrix(data["weights"][split.train], query, data["inventory"], spec.basis)
    values, fits = fit_matrix(design, outcomes[split.train], spec.link, ridge)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite output from {spec.name}")
    removed = (
        np.column_stack(
            [
                response_jacobian(fit.parameters, design.query, design.projection, Link.ADDITIVE)[0] * fit.outcome_scale
                for fit in fits
            ]
        )
        if spec.link == Link.COUPLED
        else values
    )
    prediction = values @ aggregate_weights
    removed_prediction = removed @ aggregate_weights
    test_end = len(split.test)
    bank_end = test_end + len(bank["weights"])
    diagnostics = []
    for column, fit in enumerate(fits):
        _, _, clipped = response_jacobian(fit.parameters, design.query, design.projection, spec.link)
        diagnostics.append(
            {
                "component": column,
                "objective": fit.objective,
                "converged": fit.success,
                "projected_gradient_norm": fit.gradient_norm,
                "iterations": fit.iterations,
                "dof": local_dof(fit, design, spec.link, ridge),
                "start_objectives": fit.start_objectives,
                "start_successes": fit.start_successes,
                "message": fit.message,
                "clipped_query_rows": clipped,
                "negative_query_rows": int((values[:, column] < 0).sum()),
            }
        )
    benchmark.harness.atomic_save(
        path,
        {
            "fingerprint": fingerprint,
            "prediction": prediction[:test_end],
            "bank_prediction": prediction[test_end:bank_end],
            "train_prediction": prediction[bank_end:],
            "atomic_prediction": values[:test_end],
            "atomic_bank_prediction": values[test_end:bank_end],
            "removed_prediction": removed_prediction[:test_end],
            "removed_bank_prediction": removed_prediction[test_end:bank_end],
            "test": split.test,
            "train": split.train,
            "parameters": np.stack([fit.parameters for fit in fits]),
            "outcome_scale": np.array([fit.outcome_scale for fit in fits]),
            "anchor": design.anchor,
            "feature_scale": design.scale,
            "projection": design.projection,
            "selected_json": json.dumps(choice),
            "cv_json": json.dumps(table),
            "diagnostics_json": json.dumps(diagnostics),
            "elapsed": time.monotonic() - started,
        },
    )
    return "fitted"


def prepare(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "inputs").exists():
        benchmark.verify_inputs(source)
        shutil.copytree(source / "inputs", output / "inputs")
        shutil.copy2(source / "input_hashes.json", output / "input_hashes.json")
    benchmark.verify_inputs(output)
    protocol = {
        "specifications": [dataclasses.asdict(spec) for spec in SPECS],
        "ridge_grid": RIDGES,
        "shape": {"rate": WEIBULL_RATE, "power": WEIBULL_POWER, "harm_threshold": HARM_THRESHOLD},
        "anchor": "feature image of training-mean mixture; column SD from training only",
        "share_gauge": "coefficients orthogonal to feature scales, 38 slope contrasts for 39 buckets",
        "sign_constraints": "signed share contrasts; nonnegative benefit/harm coefficients for fixed-Weibull basis",
        "fit_loss": "per-task raw BPB squared error divided by training mean BPB squared, plus ridge*sum(beta^2)",
        "selection": "one common ridge per target/fold chosen by inner aggregate RMSE; descending-ridge tie break",
        "starts": "zero slopes and log-target ridge; both optimized to the same raw-BPB objective",
        "floor": 0,
        "optimizer": "L-BFGS-B; maxiter1500,ftol1e-12,gtol1e-7,maxls40; log-amplitude bounds[-20,20]",
        "numerical_exponent_limit": EXP_LIMIT,
        "negative_additive_predictions": "retained and reported, never clipped",
        "extra_ablation": "coupled coefficients evaluated with additive formula without refitting",
        "fit_data": "canonical280 only",
        "development_data": "frozen old bank, prior CC/epoch-cap/pilot evidence only; no running ladder",
        "evaluation": (
            "original blocked nested folds, both targets, all requested metrics and connected-source contrasts"
        ),
        "successor_gate": (
            "screen only; promotion requires both-target nonworsening archive regret, "
            "Table9 source-paired regret CI below zero vs WSPU and OLMix, no invalid predictions, "
            "then original five-repeat confirmation; retrospective bank evidence cannot confirm a successor"
        ),
    }
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("The frozen coupling protocol changed; use a new output directory")
    benchmark.write_json(path, protocol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=benchmark.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("prepare", "fit"), default="fit")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, choices=(1, 5), default=1)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir)
    fingerprint = benchmark.source_fingerprint(args.output_dir, (Path(__file__),))
    provenance = json.loads((args.output_dir / f"provenance_{fingerprint[:16]}.json").read_text())
    for relative, expected in provenance["sources"].items():
        destination = args.output_dir / "source_snapshot" / relative
        source = benchmark.REPO_ROOT / relative
        if destination.resolve() == source.resolve():
            continue
        content = source.read_bytes()
        if benchmark.sha256(source) != expected:
            raise ValueError(f"Source changed while snapshotting: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    if args.stage == "prepare":
        print({"prepared": str(args.output_dir), "fingerprint": fingerprint})
        return
    tasks = [
        (spec, target, fold, repeat)
        for spec in SPECS
        for target in benchmark.TARGETS
        for repeat in range(args.repeats)
        for fold in ((-1, 0, 1, 2, 3, 4) if repeat == 0 else (0, 1, 2, 3, 4))
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.workers, verbose=10)(
            delayed(fit_specification)(args.output_dir, spec, target, fold, repeat, fingerprint)
            for spec, target, fold, repeat in tasks
        )
    print({status: results.count(status) for status in sorted(set(results))})


if __name__ == "__main__":
    main()
