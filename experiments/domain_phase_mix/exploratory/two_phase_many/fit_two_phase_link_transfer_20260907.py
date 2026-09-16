# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Audit and fit tied-centered temporal departures on the structured 300M panel.

Aggregate spines are produced separately with fully nested tied-only fits.
No target from an outer or inner validation counterpart trains its predictor.
All computations are local; proposed policies are not measured outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
from two_phase_link_residual_20260907 import (
    PENALTY_GRID,
    TemporalArm,
    fit_bpb_contrasts,
    predict_bpb_delta,
    run_structural_checks,
    temporal_basis,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/two_phase_link_transfer_20260907"
OBJECTIVES = ("uncheatable", "table9")
ARMS = ("aggregate", "damage-only", "benefit+damage")
CONTEXTS = ("final", "outer0", "outer1", "outer2")
NOISE_REFERENCE = {"uncheatable": 0.001127, "table9": 0.003330}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def inputs(output: str) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = Path(output) / "inputs"
    spec = importlib.util.spec_from_file_location("link2_single_phase", root / "single_phase.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with np.load(root / "panel.npz", allow_pickle=False) as data:
        panel = {key: data[key].copy() for key in data.files}
    with np.load(root / "splits.npz", allow_pickle=False) as data:
        splits = {key: data[key].copy() for key in data.files}
    return module, panel, splits


def load_spine(output: Path, context: str, objective: str, component: int) -> Any:
    module, _, _ = inputs(str(output))
    path = output / "spines" / context / f"{objective}_c{component}.json"
    return module.TaskFit.from_json(json.loads(path.read_text()))


def basis_and_prediction(module: Any, panel: dict[str, np.ndarray], spine: Any) -> tuple[np.ndarray, ...]:
    shape = spine.shape
    width = len(panel["buckets"])
    basis = temporal_basis(
        panel["phase0"],
        panel["phase1"],
        panel["c0"],
        panel["c1"],
        spine.head.coefficients[:width],
        spine.head.coefficients[width:],
        lambda exposure: module.benefit(exposure, shape["rate"], shape["power"]),
        lambda exposure: module.harm(exposure, shape["threshold"]),
    )
    matrix = module.design_matrix(panel["epochs"], shape)
    eta = spine.head.intercept + matrix @ spine.head.coefficients
    deficit = np.exp(np.clip(eta, -30, 30))
    assert np.max(np.abs(basis.columns[panel["physical_tied"]]), initial=0) < 1e-12
    return spine.head.floor + deficit, deficit, basis.columns


def selected_pairs(panel: dict[str, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
    left_mask, right_mask = np.isin(left, rows), np.isin(right, rows)
    assert np.array_equal(left_mask, right_mask)
    return left[left_mask], right[left_mask]


def training_rows(panel: dict[str, np.ndarray], splits: dict[str, np.ndarray], context: str) -> np.ndarray:
    return np.arange(len(panel["runs"])) if context == "final" else splits[f"{context}_train"]


def fit_task(argument: tuple[str, str, str, int]) -> dict[str, Any]:
    output_string, context, objective, component = argument
    output = Path(output_string)
    module, panel, splits = inputs(output_string)
    path = output / "temporal" / context / f"{objective}_c{component}.json"
    source_paths = [
        Path(__file__),
        HERE / "two_phase_link_residual_20260907.py",
        output / "inputs/panel.npz",
        output / "inputs/splits.npz",
    ]
    source_paths += [
        output / "spines" / prefix / f"{objective}_c{component}.json"
        for prefix in [context, *[f"{context}_inner{i}" for i in range(3)]]
    ]
    identity = {str(p): file_hash(p) for p in source_paths}
    if path.exists():
        record = json.loads(path.read_text())
        if record["input_hashes"] != identity:
            raise ValueError(f"cached temporal task changed: {path}")
        return {"context": context, "objective": objective, "component": component, "cached": True}
    response = panel[f"{objective}_outcomes"][:, component]
    sweep: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS[1:]}
    for arm in ARMS[1:]:
        for penalty in PENALTY_GRID:
            error, count = 0.0, 0
            for inner in range(3):
                prefix = f"{context}_inner{inner}"
                spine = load_spine(output, prefix, objective, component)
                _, q, columns = basis_and_prediction(module, panel, spine)
                train_a, train_t = selected_pairs(panel, splits[f"{prefix}_train"])
                test_a, test_t = selected_pairs(panel, splits[f"{prefix}_test"])
                if not len(test_a):
                    continue
                if len(train_a):
                    fitted = fit_bpb_contrasts(
                        q[train_a], columns[train_a], response[train_a] - response[train_t], penalty, TemporalArm(arm)
                    )
                    if not fitted.success:
                        raise RuntimeError(
                            f"temporal fit did not converge: {context}/{objective}/{component}/{inner}/{penalty}"
                        )
                    prediction = predict_bpb_delta(q[test_a], columns[test_a], fitted.theta)
                else:
                    prediction = np.zeros(len(test_a))
                error += float(np.sum((prediction - (response[test_a] - response[test_t])) ** 2))
                count += len(test_a)
            if count == 0:
                raise ValueError(f"no held-out phase contrasts for {context}")
            sweep[arm].append({"penalty": "inf" if math.isinf(penalty) else penalty, "mse": error / count, "n": count})
    spine = load_spine(output, context, objective, component)
    _, q, columns = basis_and_prediction(module, panel, spine)
    pair_a, pair_t = selected_pairs(panel, training_rows(panel, splits, context))
    fits = {}
    for arm in ARMS[1:]:
        winner = min(enumerate(sweep[arm]), key=lambda item: (item[1]["mse"], -item[0]))[1]
        penalty = float(winner["penalty"])
        fitted = fit_bpb_contrasts(
            q[pair_a], columns[pair_a], response[pair_a] - response[pair_t], penalty, TemporalArm(arm)
        )
        if not fitted.success:
            raise RuntimeError(f"final temporal fit did not converge: {context}/{objective}/{component}")
        fits[arm] = fitted.to_json() | {"inner_mse": winner["mse"]}
    write_json(
        path,
        {
            "context": context,
            "objective": objective,
            "component": component,
            "component_name": str(panel[f"{objective}_components"][component]),
            "input_hashes": identity,
            "fits": fits,
            "sweep": sweep,
            "train_asymmetric_rows": pair_a.tolist(),
            "train_tied_rows": pair_t.tolist(),
        },
    )
    return {
        "context": context,
        "objective": objective,
        "component": component,
        "penalties": {arm: fits[arm]["penalty"] for arm in fits},
    }


def safe_rank(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(observed) < 3 or np.std(predicted) < 1e-14 or np.std(observed) < 1e-14:
        return None
    return float(spearmanr(observed, predicted).statistic)


def metrics(observed: np.ndarray, predicted: np.ndarray, tied: np.ndarray | None = None) -> dict[str, Any]:
    order = np.argsort(predicted, kind="stable")
    physical_tied = np.zeros(len(order), bool) if tied is None else tied
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and predicted[order[end]] <= predicted[order[begin]] + 1e-12:
            end += 1
        group = order[begin:end]
        order[begin:end] = group[np.lexsort((group, ~physical_tied[group]))]
        begin = end
    best = float(np.min(observed))
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "bias_predicted_minus_measured": float(np.mean(predicted - observed)),
        "spearman": safe_rank(observed, predicted),
        "regret1": float(observed[order[0]] - best),
        "regret3": float(np.min(observed[order[:3]]) - best),
        "regret5": float(np.min(observed[order[:5]]) - best),
        "selected_rank": int(np.sum(observed < observed[order[0]]) + 1),
        "optimism_at_pick": float(observed[order[0]] - predicted[order[0]]),
        "tie_policy": "predictions within 1e-12: physically tied first, then original row order",
    }


def evaluate(output: Path) -> None:
    module, panel, _ = inputs(str(output))
    records, pair_records, parameter_records = [], [], []
    for objective in OBJECTIVES:
        weights = panel[f"{objective}_aggregation_weights"]
        response = panel[f"{objective}_outcomes"] @ weights
        predictions = {arm: np.full(len(response), np.nan) for arm in ARMS}
        final_predictions = {arm: np.zeros(len(response)) for arm in ARMS}
        for context in CONTEXTS:
            eligible = np.ones(len(response), bool) if context == "final" else panel["outer_fold"] == int(context[-1])
            combined = {arm: np.zeros(len(response)) for arm in ARMS}
            for component, weight in enumerate(weights):
                spine = load_spine(output, context, objective, component)
                base, q, columns = basis_and_prediction(module, panel, spine)
                record = json.loads((output / "temporal" / context / f"{objective}_c{component}.json").read_text())
                combined["aggregate"] += weight * base
                for arm in ARMS[1:]:
                    fit = record["fits"][arm]
                    theta = np.asarray(fit["theta"])
                    combined[arm] += weight * (base + predict_bpb_delta(q, columns, theta))
                    parameter_records.append(
                        {
                            "objective": objective,
                            "component": component,
                            "context": context,
                            "arm": arm,
                            "u": theta[0],
                            "v": theta[1],
                            "penalty": fit["penalty"],
                            "phase_inner_mse": fit["inner_mse"],
                            "max_abs_log_correction": float(np.max(np.abs(columns @ theta))),
                        }
                    )
            for arm in ARMS:
                if context == "final":
                    final_predictions[arm] = combined[arm]
                else:
                    predictions[arm][eligible] = combined[arm][eligible]
        table = pd.DataFrame(
            {
                "row": np.arange(len(response)),
                "run": panel["runs"],
                "group": panel["groups"],
                "fold": panel["outer_fold"],
                "tied": panel["physical_tied"],
                "measured": response,
                **predictions,
                **{f"final_{arm}": value for arm, value in final_predictions.items()},
            }
        )
        table.to_csv(output / f"predictions_{objective}.csv", index=False)
        scoreable = ~panel["calibration_mask"]
        for arm in ARMS:
            assert np.isfinite(predictions[arm][scoreable]).all()
            for population, mask in [
                ("all", scoreable),
                ("tied", scoreable & panel["physical_tied"]),
                ("asymmetric", scoreable & ~panel["physical_tied"]),
            ]:
                records.append(
                    {
                        "objective": objective,
                        "arm": arm,
                        "population": population,
                        **metrics(response[mask], predictions[arm][mask], panel["physical_tied"][mask]),
                        "selection_scope": "stitched outer-fold predictions; not a continuous policy validation",
                    }
                )
        a, t = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
        paired = pd.DataFrame(
            {
                "asymmetric_row": a,
                "tied_row": t,
                "group": panel["groups"][a],
                "fold": panel["outer_fold"][a],
                "measured_delta": response[a] - response[t],
                **{arm: predictions[arm][a] - predictions[arm][t] for arm in ARMS},
            }
        )
        for arm in ARMS:
            paired.loc[np.abs(paired[arm]) < 1e-12, arm] = 0.0
        assert np.max(np.abs(paired["aggregate"])) == 0.0
        paired.to_csv(output / f"pairs_{objective}.csv", index=False)
        truth = response[a] - response[t]
        for arm in ARMS:
            pred = paired[arm].to_numpy()
            decision_regret = np.where(pred < 0, np.maximum(truth, 0), np.maximum(-truth, 0))
            pair_records.append(
                {
                    "objective": objective,
                    "arm": arm,
                    "n": len(truth),
                    "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                    "bias": float(np.mean(pred - truth)),
                    "spearman": safe_rank(truth, pred),
                    "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(truth))),
                    "mean_binary_decision_regret": float(np.mean(decision_regret)),
                    "asymmetric_pick_fraction": float(np.mean(pred < 0)),
                }
            )
    pd.DataFrame(records).to_csv(output / "oof_metrics.csv", index=False)
    pd.DataFrame(pair_records).to_csv(output / "pair_metrics.csv", index=False)
    pd.DataFrame(parameter_records).to_csv(output / "temporal_parameters.csv", index=False)
    print(pd.DataFrame(pair_records).to_string(index=False), flush=True)


def signal_error(scale: float, q: np.ndarray, columns: np.ndarray, direction: np.ndarray, signal_rms: float) -> float:
    delta = predict_bpb_delta(q, columns, direction * scale)
    return float(np.sqrt(np.mean(delta**2)) - signal_rms)


def synthetic_audit(output: Path, draws: int) -> None:
    """Measure conditional low-dimensional recovery with an estimated tied spine.

    Shapes, ridge and floor are held fixed in this cheap identification check.
    Synthetic tied outcomes refit amplitudes; no real asymmetric outcomes are
    used. Noise is a local sensitivity, not an estimated global noise law.
    """
    module, panel, splits = inputs(str(output))
    structural = run_structural_checks(lambda e: module.benefit(e, 0.5, 0.7), lambda e: module.harm(e, 2.0))
    write_json(output / "structural_checks.json", structural)
    records, geometry = [], []
    rng = np.random.default_rng(20260907)
    a = panel["pair_asymmetric_rows"]
    for objective in OBJECTIVES:
        for component in range(len(panel[f"{objective}_components"])):
            spine = load_spine(output, "final", objective, component)
            base, q, columns = basis_and_prediction(module, panel, spine)
            derivative = q[a, None] * columns[a]
            norms = np.linalg.norm(derivative, axis=0)
            active = norms > 1e-12
            scaled = derivative[:, active] / norms[active]
            singular = np.linalg.svd(scaled, compute_uv=False)
            rank = int(np.sum(singular > 1e-8))
            geometry.append(
                {
                    "objective": objective,
                    "component": component,
                    "active_columns": int(active.sum()),
                    "rank": rank,
                    "smallest_singular": float(singular[-1]) if len(singular) else 0,
                    "condition": float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 1e-12 else None,
                }
            )
            if not rank:
                continue
            global_component = component if objective == "uncheatable" else component + 7
            noise_sd = max(float(panel["anchor_repeat_sd"][global_component]), NOISE_REFERENCE[objective])
            signal_rms = 0.0039 if objective == "table9" else 0.0013
            matrix = module.design_matrix(panel["epochs"], spine.shape)
            for direction_id, raw in enumerate([(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)]):
                direction = np.divide(np.asarray(raw), norms, out=np.zeros(2), where=active)
                if np.linalg.norm(derivative @ direction) < 1e-10:
                    continue

                arguments = (q[a], columns[a], direction, signal_rms)
                high = 1.0
                while signal_error(high, *arguments) < 0 and high < 1e6:
                    high *= 2
                theta = direction * brentq(signal_error, 0, high, args=arguments)
                signal = predict_bpb_delta(q, columns, theta)
                for draw in range(draws):
                    tied_noise = rng.normal(0, noise_sd, len(base))
                    asym_noise = rng.normal(0, noise_sd, len(base))
                    for truth_kind in ("signal", "null"):
                        truth = signal if truth_kind == "signal" else np.zeros(len(base))
                        predicted = np.full(len(base), np.nan)
                        for fold in range(3):
                            train_rows = splits[f"outer{fold}_train"]
                            train_tied = train_rows[panel["physical_tied"][train_rows]]
                            train_a, train_t = selected_pairs(panel, train_rows)
                            test_a, _ = selected_pairs(panel, splits[f"outer{fold}_test"])
                            if not len(test_a):
                                continue
                            log_response = np.log(
                                np.maximum(
                                    base[train_tied] + tied_noise[train_tied] - spine.head.floor, module.DEFICIT_FLOOR
                                )
                            )
                            intercept, coefficients = module.nonnegative_solve(
                                matrix[train_tied], log_response, spine.ridge
                            )
                            fitted_head = module.Head(intercept, coefficients, spine.head.floor)
                            fitted_spine = module.TaskFit(
                                spine.component,
                                spine.shape,
                                spine.ridge,
                                spine.kappa,
                                spine.flat_profile,
                                spine.inner_cv_rmse,
                                fitted_head,
                            )
                            _, fitted_q, fitted_columns = basis_and_prediction(module, panel, fitted_spine)
                            observed = truth[train_a] + asym_noise[train_a] - tied_noise[train_t]
                            fitted = fit_bpb_contrasts(fitted_q[train_a], fitted_columns[train_a], observed, 0.0)
                            if not fitted.success:
                                raise RuntimeError("synthetic recovery did not converge")
                            predicted[test_a] = predict_bpb_delta(fitted_q[test_a], fitted_columns[test_a], fitted.theta)
                        assert np.isfinite(predicted[a]).all()
                        records.append(
                            {
                                "objective": objective,
                                "component": component,
                                "direction": direction_id,
                                "draw": draw,
                                "truth": truth_kind,
                                "noise_sd": noise_sd,
                                "signal_rms": signal_rms,
                                "rmse_to_truth": float(np.sqrt(np.mean((predicted[a] - truth[a]) ** 2))),
                                "rmse_signal_ratio": float(
                                    np.sqrt(np.mean((predicted[a] - truth[a]) ** 2)) / signal_rms
                                ),
                                "predicted_gain_rms": float(np.sqrt(np.mean(np.minimum(predicted[a], 0) ** 2))),
                                "scope": (
                                    "amplitudes re-estimated on noisy synthetic tied data; "
                                    "shapes/ridge/floor fixed; local noise sensitivity"
                                ),
                            }
                        )
            pd.DataFrame(records).to_csv(output / "synthetic_recovery.csv", index=False)
            pd.DataFrame(geometry).to_csv(output / "temporal_geometry.csv", index=False)
            print(json.dumps(geometry[-1]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("synthetic", "fit", "evaluate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draws", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "synthetic":
        synthetic_audit(args.output, args.draws)
    elif args.phase == "evaluate":
        evaluate(args.output)
    else:
        _, panel, _ = inputs(str(args.output))
        jobs = [
            (str(args.output), context, objective, component)
            for context in CONTEXTS
            for objective in OBJECTIVES
            for component in range(len(panel[f"{objective}_components"]))
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fit_task, job) for job in jobs]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
