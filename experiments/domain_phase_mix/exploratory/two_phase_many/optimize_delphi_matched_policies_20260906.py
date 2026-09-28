# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate", "threadpoolctl"]
# ///
"""Compare frozen Delphi surrogates under identical offline policy constraints.

Execute this file by absolute path with PYTHONPATH set to the earlier benchmark's
reproduction_sources directory. Reconstruction requires those pinned imports.
Only canonical outcomes and frozen bank predictions/features are read; proposed
continuous policies have no measured outcomes and are never submitted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import linprog, minimize
from scipy.special import xlogy
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix import olmix_loglinear_fit
from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import single_phase_observatory_models_20260902 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
DEFAULT_FROZEN = REFERENCE / "delphi_offline_selection_20260906"
DEFAULT_OUTPUT = REFERENCE / "delphi_coupling_followup_20260906" / "continuous_policies"
MODEL_IDS = ("weibull_softplus_unscaled", "olmix_loglinear_taskwise")
CAPS = (4, 6, 8, 16)
KL_VALUES = (0.0, 0.005, 0.02)
START_SEED = 20260906
PANEL_STARTS = 4
PARITY_TOLERANCE = 1e-8
FEASIBILITY_TOLERANCE = 1e-7
OPTIMIZER_OPTIONS = {"maxiter": 500, "ftol": 1e-11, "eps": 1e-8}


class Surrogate(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def target(self) -> str: ...

    def predict(self, weights: np.ndarray) -> np.ndarray: ...


@dataclasses.dataclass(frozen=True)
class WspuSurrogate:
    target: str
    inventory: np.ndarray
    aggregation: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    rate: np.ndarray
    power: np.ndarray
    threshold: np.ndarray
    model_id: str = "weibull_softplus_unscaled"

    def predict(self, weights: np.ndarray) -> np.ndarray:
        exposure = np.atleast_2d(weights)[:, None, :] * self.inventory[None, None, :]
        benefit = -np.expm1(-((self.rate[None, :, None] * np.maximum(exposure, 0)) ** self.power[None, :, None]))
        harm = np.logaddexp(0.0, np.log1p(np.maximum(exposure, 0)) - self.threshold[None, :, None]) ** 2
        buckets = len(self.inventory)
        values = self.intercept[None, :] - np.einsum("ncb,cb->nc", benefit, self.coefficients[:, :buckets])
        values += np.einsum("ncb,cb->nc", harm, self.coefficients[:, buckets:])
        return values @ self.aggregation


@dataclasses.dataclass(frozen=True)
class OlmixSurrogate:
    target: str
    aggregation: np.ndarray
    log_c: np.ndarray
    coefficients: np.ndarray
    model_id: str = "olmix_loglinear_taskwise"

    def predict(self, weights: np.ndarray) -> np.ndarray:
        limit = olmix_loglinear_fit.MAX_LOG_MAGNITUDE
        logits = np.clip(np.atleast_2d(weights) @ self.coefficients.T, -limit, limit)
        values = np.exp(np.clip(self.log_c[None, :], -limit, limit)) + np.exp(logits)
        return values @ self.aggregation


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def pinned_sources(frozen: Path) -> dict[str, str]:
    root = (frozen / "reproduction_sources").resolve()
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename or not str(filename).endswith(".py"):
            continue
        path = Path(filename).resolve()
        if path == Path(__file__).resolve() or "experiments" not in path.parts:
            continue
        if not path.is_relative_to(root):
            raise ValueError(f"Unpinned experiment import: {path}. Run the absolute script with PYTHONPATH={root}")
        sources[str(path.relative_to(root))] = benchmark.sha256(path)
    for module in (benchmark, harness, models, registry, olmix_loglinear_fit):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError(f"Required model source is not pinned: {module.__file__}")
    return sources


def freeze_protocol(frozen: Path, output: Path) -> tuple[dict, str]:
    """Persist the complete read set and policy protocol before refits or solves."""
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    if data["weights"].shape != (280, 39):
        raise ValueError("This diagnostic requires the frozen canonical 280-by-39 panel")
    paths = [frozen / "inputs" / "panel.npz"]
    paths.extend(frozen / "inputs" / f"{target}_bank_features.npz" for target in benchmark.TARGETS)
    paths.extend(
        frozen / "baseline_shards" / model / target / f"r0_f-1_c{component}.npz"
        for model in MODEL_IDS
        for target in benchmark.TARGETS
        for component in range(data[f"{target}_outcomes"].shape[1])
    )
    protocol = {
        "version": 1,
        "scope": "offline sensitivity diagnostic; canonical 280 final fits; no observed continuous-policy losses",
        "read_set": {str(path.relative_to(frozen)): benchmark.sha256(path) for path in paths},
        "pinned_sources": pinned_sources(frozen),
        "script_sha256": benchmark.sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "models": list(MODEL_IDS),
        "targets": list(benchmark.TARGETS),
        "caps": list(CAPS),
        "cap_definition": "0 <= weight_b <= min(1, cap / frozen_inventory_b); sum weights = 1",
        "kl_values": list(KL_VALUES),
        "objective": "aggregate surrogate BPB + kl_value * sum(weight * log(weight / natural))",
        "natural": "normalize(1 / frozen_inventory)",
        "starts": "natural and four feature-only canonical rows, independently projected for each cap",
        "panel_start_indices": np.random.default_rng(START_SEED).choice(280, PANEL_STARTS, replace=False).tolist(),
        "optimizer": {"method": "SLSQP", "numerical_derivatives_for_both_models": True, **OPTIMIZER_OPTIONS},
        "parity_tolerance": PARITY_TOLERANCE,
        "feasibility_tolerance": FEASIBILITY_TOLERANCE,
        "selection": "minimum objective among converged feasible restarts; if none, mark best feasible unresolved",
        "outcomes_read": "canonical panel outcomes only; bank shard predictions are used only for reconstruction parity",
        "forbidden": [
            "bank labels",
            "current registry data",
            "WSPU cross-scale ladder",
            "training or evaluation launches",
        ],
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("The frozen continuous-policy protocol changed; use a new output directory")
    else:
        atomic_json(path, protocol)
        (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    return protocol, digest


def reconstruct_component(frozen: Path, output: Path, model_id: str, target: str, component: int, digest: str) -> str:
    destination = output / "reconstructed_heads" / model_id / target / f"c{component}.npz"
    if destination.exists():
        saved = benchmark.read_npz(destination)
        if str(saved["protocol_hash"]) != digest:
            raise ValueError(f"Stale reconstructed head: {destination}")
        return "cached"
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    bank = benchmark.read_npz(frozen / "inputs" / f"{target}_bank_features.npz")
    original = benchmark.read_npz(frozen / "baseline_shards" / model_id / target / f"r0_f-1_c{component}.npz")
    component_name = str(data[f"{target}_components"][component])
    entry = registry.ENTRY_BY_ID[model_id]
    features = dataclasses.replace(
        registry.apply_transform(
            benchmark.feature_set(data, benchmark.PANEL, data["weights"], data["exposures"]), entry
        ),
        component=component_name,
    )
    query = registry.apply_transform(
        benchmark.feature_set(data, f"{benchmark.PANEL}|frozen-bank", bank["weights"], bank["exposures"]), entry
    )
    model = entry.build(features)
    response = data[f"{target}_outcomes"][:, component]
    started = time.monotonic()
    if model_id == MODEL_IDS[0]:
        if not isinstance(model, models.GridModel):
            raise ValueError("Frozen WSPU is not the expected grid model")
        shape = json.loads(str(original["shape_json"]))
        design = model.design(features, shape)
        expected_names = tuple(f"bucket_signal:{i}" for i in range(39)) + tuple(
            f"bucket_overexposure:{i}" for i in range(39)
        )
        if design.names != expected_names:
            raise ValueError("Frozen WSPU design is not the expected 78-column additive basis")
        head = models.fit_head(design, response, float(original["ridge"]), model.head_for(shape))
        fit = models.Fitted(shape, float(original["ridge"]), head, {})
        payload = {"intercept": head.intercept, "coefficients": head.coefficients, "shape_json": json.dumps(shape)}
    else:
        task = harness.FitTask(model_id, benchmark.PANEL, target, component, component_name, 0, 0)
        seed = harness._seed(task)
        fit = model.fit(features, response, np.arange(280), (), seed)
        payload = {
            "log_c": fit.head.log_c,
            "coefficients": np.asarray(fit.head.coefficients),
            "huber_loss": fit.head.huber_loss,
            "seed": seed,
        }
    predictions = model.predict(fit, query, np.arange(len(bank["weights"])))
    train_predictions = model.predict(fit, features, np.arange(280))
    bank_error = float(np.max(np.abs(predictions - original["bank_prediction"])))
    train_error = float(np.max(np.abs(train_predictions - original["train_prediction"])))
    if not np.isfinite(predictions).all() or max(bank_error, train_error) > PARITY_TOLERANCE:
        raise ValueError(
            f"Frozen prediction parity failed for {model_id}/{target}/{component}: {bank_error}, {train_error}"
        )
    harness.atomic_save(
        destination,
        {
            **payload,
            "protocol_hash": digest,
            "bank_max_absolute_error": bank_error,
            "train_max_absolute_error": train_error,
            "elapsed": time.monotonic() - started,
            "component": component_name,
        },
    )
    return "fitted"


def load_surrogates(frozen: Path, output: Path) -> dict[tuple[str, str], Surrogate]:
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    result: dict[tuple[str, str], Surrogate] = {}
    for target in benchmark.TARGETS:
        count = data[f"{target}_outcomes"].shape[1]
        for model_id in MODEL_IDS:
            parts = [
                benchmark.read_npz(output / "reconstructed_heads" / model_id / target / f"c{component}.npz")
                for component in range(count)
            ]
            coefficients = np.stack([part["coefficients"] for part in parts])
            aggregation = data[f"{target}_aggregation_weights"]
            if model_id == MODEL_IDS[0]:
                shapes = [json.loads(str(part["shape_json"])) for part in parts]
                result[model_id, target] = WspuSurrogate(
                    target,
                    data["inventory"],
                    aggregation,
                    np.array([float(part["intercept"]) for part in parts]),
                    coefficients,
                    np.array([shape["rate"] for shape in shapes]),
                    np.array([shape["power"] for shape in shapes]),
                    np.array([shape["threshold"] for shape in shapes]),
                )
            else:
                result[model_id, target] = OlmixSurrogate(
                    target, aggregation, np.array([float(part["log_c"]) for part in parts]), coefficients
                )
    return result


def verify_prediction_rules(frozen: Path, output: Path, surrogates: dict[tuple[str, str], Surrogate]) -> None:
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    rows = []
    for (model_id, target), surrogate in surrogates.items():
        bank = benchmark.read_npz(frozen / "inputs" / f"{target}_bank_features.npz")
        parts = [
            benchmark.read_npz(frozen / "baseline_shards" / model_id / target / f"r0_f-1_c{component}.npz")
            for component in range(data[f"{target}_outcomes"].shape[1])
        ]
        original_bank = (
            np.column_stack([part["bank_prediction"] for part in parts]) @ data[f"{target}_aggregation_weights"]
        )
        original_train = (
            np.column_stack([part["train_prediction"] for part in parts]) @ data[f"{target}_aggregation_weights"]
        )
        bank_error = float(np.max(np.abs(surrogate.predict(bank["weights"]) - original_bank)))
        train_error = float(np.max(np.abs(surrogate.predict(data["weights"]) - original_train)))
        if max(bank_error, train_error) > PARITY_TOLERANCE:
            raise ValueError(f"Vectorized prediction parity failed: {model_id}/{target}, {bank_error}, {train_error}")
        rows.append(
            {
                "model": model_id,
                "target": target,
                "bank_max_absolute_error": bank_error,
                "train_max_absolute_error": train_error,
            }
        )
    pd.DataFrame(rows).to_csv(output / "prediction_parity.csv", index=False)


def project_start(weights: np.ndarray, upper: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Project a start by a convex quadratic solve, without legacy materializer imports."""
    if np.max(weights - upper) <= 1e-12 and weights.min() >= 0 and abs(weights.sum() - 1) <= 1e-12:
        return weights.copy()
    result = minimize(
        lambda candidate: 0.5 * float(np.square(candidate - weights).sum()),
        feasible,
        jac=lambda candidate: candidate - weights,
        bounds=list(zip(np.zeros(len(upper)), upper, strict=True)),
        constraints={
            "type": "eq",
            "fun": lambda candidate: candidate.sum() - 1,
            "jac": lambda candidate: np.ones_like(candidate),
        },
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-13},
    )
    if not result.success or max(abs(result.x.sum() - 1), -result.x.min(), np.max(result.x - upper)) > 1e-10:
        raise ValueError(f"Start projection failed: {result.message}")
    return result.x


def categorical_kl(weights: np.ndarray, natural: np.ndarray) -> float:
    return float(xlogy(weights, weights / natural).sum())


def optimize_start(
    predict: Callable[[np.ndarray], np.ndarray], start: np.ndarray, natural: np.ndarray, upper: np.ndarray, kl: float
) -> tuple[np.ndarray, dict]:
    """Solve one common constrained objective and preserve its convergence evidence."""

    def objective(weights: np.ndarray) -> float:
        return float(predict(weights[None])[0]) + kl * categorical_kl(weights, natural)

    started = time.monotonic()
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=list(zip(np.zeros(len(upper)), upper, strict=True)),
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1,
            "jac": lambda weights: np.ones_like(weights),
        },
        options=OPTIMIZER_OPTIONS,
    )
    violation = float(max(abs(result.x.sum() - 1), -result.x.min(), np.max(result.x - upper)))
    weights = project_start(result.x, upper, natural)
    value = objective(weights)
    if not np.isfinite(value):
        raise ValueError("Continuous optimizer produced a nonfinite objective")
    return weights, {
        "success": bool(result.success and violation <= FEASIBILITY_TOLERANCE),
        "solver_success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "raw_feasibility_violation": violation,
        "endpoint_projection_tv": float(np.abs(weights - result.x).sum() / 2),
        "objective": value,
        "surrogate_bpb": float(predict(weights[None])[0]),
        "kl_divergence": categorical_kl(weights, natural),
        "start_objective": objective(start),
        "elapsed": time.monotonic() - started,
    }


def support_distances(panel: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Measure nearest-row and convex-hull distances in total variation."""
    rows, buckets = panel.shape
    result = linprog(
        np.r_[np.zeros(rows), np.ones(buckets)],
        A_ub=np.vstack([np.hstack([panel.T, -np.eye(buckets)]), np.hstack([-panel.T, -np.eye(buckets)])]),
        b_ub=np.r_[weights, -weights],
        A_eq=np.r_[np.ones(rows), np.zeros(buckets)][None],
        b_eq=[1],
        bounds=[(0, None)] * (rows + buckets),
        method="highs",
    )
    if not result.success:
        raise ValueError(result.message)
    return {
        "hull_distance_tv": float(result.fun / 2),
        "nearest_panel_tv": float(np.abs(panel - weights).sum(axis=1).min() / 2),
    }


def run_policies(frozen: Path, output: Path, protocol: dict, digest: str) -> None:
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    surrogates = load_surrogates(frozen, output)
    verify_prediction_rules(frozen, output, surrogates)
    natural = 1 / data["inventory"]
    natural /= natural.sum()
    start_rows = [natural, *data["weights"][protocol["panel_start_indices"]]]
    starts = {
        cap: [project_start(weight, np.minimum(1, cap / data["inventory"]), natural) for weight in start_rows]
        for cap in CAPS
    }
    harness.atomic_save(output / "starts.npz", {f"cap_{cap}": np.stack(rows) for cap, rows in starts.items()})
    completed = 0
    for (model_id, target), surrogate in surrogates.items():
        for cap in CAPS:
            upper = np.minimum(1, cap / data["inventory"])
            for kl_index, kl in enumerate(KL_VALUES):
                for start_index, start in enumerate(starts[cap]):
                    path = output / "policy_shards" / model_id / target / f"cap{cap}_kl{kl_index}_start{start_index}.npz"
                    if path.exists():
                        if str(benchmark.read_npz(path)["protocol_hash"]) != digest:
                            raise ValueError(f"Stale policy shard: {path}")
                        continue
                    weights, diagnostics = optimize_start(surrogate.predict, start, natural, upper, kl)
                    harness.atomic_save(
                        path,
                        {"weights": weights, "diagnostics_json": json.dumps(diagnostics), "protocol_hash": digest},
                    )
                    completed += 1
                print(f"policies {model_id}/{target} cap={cap} kl={kl:g}; new restarts={completed}", flush=True)
    summarize(frozen, output, surrogates, natural)


def summarize(frozen: Path, output: Path, surrogates: dict[tuple[str, str], Surrogate], natural: np.ndarray) -> None:
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    restarts = []
    selected = []
    chosen_weights = {}
    for model_id, target in surrogates:
        for cap in CAPS:
            for kl_index, kl in enumerate(KL_VALUES):
                rows = []
                for start_index in range(PANEL_STARTS + 1):
                    part = benchmark.read_npz(
                        output / "policy_shards" / model_id / target / f"cap{cap}_kl{kl_index}_start{start_index}.npz"
                    )
                    diagnostics = json.loads(str(part["diagnostics_json"]))
                    row = {
                        "model": model_id,
                        "target": target,
                        "cap": cap,
                        "kl_coefficient": kl,
                        "start": start_index,
                        **diagnostics,
                    }
                    rows.append((row, part["weights"]))
                    restarts.append(row)
                converged = [item for item in rows if item[0]["success"]]
                best, weights = min(converged or rows, key=lambda item: (item[0]["objective"], item[0]["start"]))
                all_values = [item[0]["objective"] for item in rows]
                converged_values = [item[0]["objective"] for item in converged]
                all_distances = [np.abs(first[1] - second[1]).sum() / 2 for first in rows for second in rows]
                record = {
                    **best,
                    "successful_restarts": len(converged),
                    "total_restarts": len(rows),
                    "selection_status": "converged_candidate" if converged else "unresolved_best_feasible_candidate",
                    "all_restart_objective_spread": float(max(all_values) - min(all_values)),
                    "converged_restart_objective_spread": (
                        float(max(converged_values) - min(converged_values)) if converged else None
                    ),
                    "all_restart_max_policy_tv": float(max(all_distances)),
                    "max_exposure": float(np.max(weights * data["inventory"])),
                    "effective_buckets": float(np.exp(-xlogy(weights, weights).sum())),
                    "distance_from_natural_tv": float(np.abs(weights - natural).sum() / 2),
                    "buckets_above_panel_max": int((weights > data["weights"].max(axis=0) + 1e-8).sum()),
                    **support_distances(data["weights"], weights),
                    "measured_uncheatable_bpb": None,
                    "measured_table9_bpb": None,
                    "measurement_status": "unknown_not_run",
                }
                for (prediction_model, prediction_target), surrogate in surrogates.items():
                    record[f"prediction::{prediction_model}::{prediction_target}"] = float(
                        surrogate.predict(weights[None])[0]
                    )
                record.update(
                    {f"weight::{bucket}": float(weight) for bucket, weight in zip(data["buckets"], weights, strict=True)}
                )
                selected.append(record)
                chosen_weights[model_id, target, cap, kl] = weights
    pd.DataFrame(restarts).to_csv(output / "restart_diagnostics.csv", index=False)
    table = pd.DataFrame(selected)
    table.to_csv(output / "selected_policies.csv", index=False)
    comparisons = []
    for target in benchmark.TARGETS:
        for cap in CAPS:
            for kl in KL_VALUES:
                first = chosen_weights[MODEL_IDS[0], target, cap, kl]
                second = chosen_weights[MODEL_IDS[1], target, cap, kl]
                row = {
                    "target": target,
                    "cap": cap,
                    "kl_coefficient": kl,
                    "wspu_olmix_policy_tv": float(np.abs(first - second).sum() / 2),
                }
                for model_id in MODEL_IDS:
                    predictions = surrogates[model_id, target].predict(np.stack([first, second]))
                    row[f"{model_id}::predicted_wspu_minus_olmix_bpb"] = float(predictions[0] - predictions[1])
                comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(output / "matched_policy_comparisons.csv", index=False)
    atomic_json(
        output / "summary.json",
        {
            "selected_policies": len(selected),
            "restarts": len(restarts),
            "successful_restarts": int(pd.DataFrame(restarts).success.sum()),
            "unresolved_cells": int((table.successful_restarts == 0).sum()),
            "maximum_raw_feasibility_violation": float(pd.DataFrame(restarts).raw_feasibility_violation.max()),
            "maximum_converged_objective_spread": float(table.converged_restart_objective_spread.max()),
            "read_scope": "canonical 280 outcomes; frozen bank features and predictions only for parity",
            "interpretation": (
                "Policy sensitivity and surrogate disagreement only; all proposed-policy measured losses are unknown."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("freeze", "reconstruct", "policies", "all"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    frozen = args.frozen_dir.resolve()
    output = args.output_dir.resolve()
    protocol, digest = freeze_protocol(frozen, output)
    with threadpool_limits(limits=1):
        if args.stage in ("reconstruct", "all"):
            data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
            tasks = [
                (model_id, target, component)
                for model_id in MODEL_IDS
                for target in benchmark.TARGETS
                for component in range(data[f"{target}_outcomes"].shape[1])
            ]
            with parallel_config(backend="loky", inner_max_num_threads=1):
                statuses = Parallel(n_jobs=args.workers, verbose=10)(
                    delayed(reconstruct_component)(frozen, output, *task, digest) for task in tasks
                )
            print(pd.Series(statuses).value_counts().to_dict(), flush=True)
            verify_prediction_rules(frozen, output, load_surrogates(frozen, output))
        if args.stage in ("policies", "all"):
            run_policies(frozen, output, protocol, digest)


if __name__ == "__main__":
    main()
