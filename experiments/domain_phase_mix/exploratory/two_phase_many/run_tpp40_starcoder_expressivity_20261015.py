# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""All-training StarCoder estimation diagnostic within frozen model families.

This optimistic diagnostic estimates no held-out performance or model ceiling.
It changes the objective and search, preserving the registered functions, signs,
parameter bounds, native clipping, and complete aggregate-null candidates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_models_20261015 as models

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_starcoder_expressivity_20261015"
PRIOR = ROOT / "reference_outputs/tpp40_tangent_starcoder_20261015"
DATA = ROOT / "reference_outputs/two_phase_benchmark_20260921/v3/prepared/starcoder_wsd80_dense"
FAMILIES = models.MODEL_NAMES


@dataclass(frozen=True)
class Panel:
    early: np.ndarray
    late: np.ndarray
    phase0: np.ndarray
    phase1: np.ndarray
    alpha: float
    rows: pd.DataFrame
    outcomes: pd.DataFrame


@dataclass(frozen=True)
class State:
    """Exact model parameters independent of any search or regularization."""

    family: str
    alpha: float
    shape: dict[str, float]
    strength: float
    floor: float
    intercept: float
    coefficients: np.ndarray

    def design(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        parameters = {} if self.family == "aggregate_mariner" else {"kappa": self.strength}
        matrix, offset = models.design_and_offset(early, late, self.alpha, self.shape, self.family, parameters)
        assert not np.any(offset)
        return matrix

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        head = models.mariner.Head(self.intercept, self.coefficients, self.floor)
        return head.predict(self.design(early, late))

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "alpha": self.alpha,
            "shape": self.shape,
            "strength": self.strength,
            "floor": self.floor,
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> State:
        return cls(
            record["family"],
            float(record["alpha"]),
            {key: float(value) for key, value in record["shape"].items()},
            float(record["strength"]),
            float(record["floor"]),
            float(record["intercept"]),
            np.asarray(record["coefficients"], dtype=float),
        )


def registration() -> dict[str, Any]:
    return json.loads((OUTPUT / "registration.json").read_text())


def load_panel() -> Panel:
    metadata = json.loads((DATA / "metadata.json").read_text())
    rows = pd.read_csv(DATA / "rows.csv", keep_default_na=False)
    outcomes = pd.read_csv(DATA / "outcomes.csv")
    with np.load(DATA / "policies.npz") as arrays:
        result = Panel(
            arrays["phase0_epochs"],
            arrays["phase1_epochs"],
            arrays["phase0"],
            arrays["phase1"],
            float(metadata["alpha"]),
            rows,
            outcomes,
        )
    assert metadata["buckets"] == ["nemotron_full", "starcoder"]
    assert len(rows) == len(outcomes) == registration()["expected_rows"]
    assert result.early.shape == result.late.shape == (len(rows), 2)
    assert rows.row_id.is_unique
    assert np.isfinite(outcomes[registration()["targets"]].to_numpy()).all()
    return result


def source_identity() -> dict[str, Any]:
    paths = [
        Path(__file__),
        OUTPUT / "registration.json",
        PRIOR / "result_manifest.json",
        PRIOR / "registration.json",
        PRIOR / "execution_identity.json",
        PRIOR / "complete.json",
        Path(models.__file__),
        Path(models.frozen.__file__),
        Path(models.frozen.mechanisms.__file__),
        Path(models.frozen.temporal.__file__),
        Path(models.mariner.__file__),
        Path(benchmark.__file__),
        benchmark.REPO / "pyproject.toml",
        benchmark.REPO / "uv.lock",
        *(DATA / filename for filename in ("policies.npz", "rows.csv", "outcomes.csv", "metadata.json", "folds.csv")),
    ]
    prior_identity = json.loads((PRIOR / "execution_identity.json").read_text())
    prior_seal = json.loads((PRIOR / "result_manifest.json").read_text())
    for task in range(len(registration()["targets"])):
        for fold in registration()["folds"]:
            for family in FAMILIES:
                folder = PRIOR / f"fits/fold{fold}_task{task}/{family}"
                paths.extend([folder / "fit.json", folder / "complete.json"])
    sources = {str(path.resolve()): benchmark.sha256(path) for path in paths}
    for path, value in sources.items():
        expected = prior_identity["sources"].get(path)
        if expected is not None and value != expected:
            raise ValueError(f"A frozen prior source changed: {path}")
    for path in paths:
        if path.is_relative_to(PRIOR) and path.name != "result_manifest.json":
            sealed = prior_seal["files"][str(path.relative_to(PRIOR))]["sha256"]
            if sources[str(path.resolve())] != sealed:
                raise ValueError(f"Sealed prior output changed: {path}")
    prior_registration = json.loads((PRIOR / "registration.json").read_text())
    assert registration()["targets"] == prior_registration["targets"]
    assert registration()["families"] == list(FAMILIES)
    return {"sources": sources, "environment": benchmark.environment_identity()}


def prepare() -> None:
    identity = source_identity()
    panel = load_panel()
    path = OUTPUT / "execution_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Existing preparation differs; preserve it and review before changing identity")
        print("Cached preparation identity verified", flush=True)
        return
    benchmark.write_json(path, identity)
    benchmark.write_json(
        OUTPUT / "preparation.json",
        {
            "rows": len(panel.rows),
            "targets": registration()["targets"],
            "families": list(FAMILIES),
            "strata_counts": {name: int(mask.sum()) for name, mask in strata(panel).items()},
            "all_rows_used_for_fitting": True,
            "scientific_predictions_or_fits_computed": False,
            "identity": benchmark.receipt(path),
        },
    )
    print("Prepared geometry, source identity and descriptive strata only", flush=True)


def verified_identity() -> dict[str, str]:
    saved = json.loads((OUTPUT / "execution_identity.json").read_text())
    if saved != source_identity():
        raise ValueError("Execution sources or environment differ from preparation")
    return benchmark.receipt(OUTPUT / "execution_identity.json")


def require_approval(identity: dict[str, str]) -> None:
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    if (
        approval["execution_identity_sha256"] != identity["sha256"]
        or approval["registration_sha256"] != benchmark.sha256(OUTPUT / "registration.json")
        or approval["scope"] != "TP40-027 all-training diagnostic only"
    ):
        raise ValueError("Execution approval does not match this diagnostic")


def floor_contract(observed: np.ndarray) -> tuple[float, float]:
    """Recover native median/minimum gap including its positive-gap fallback."""
    median = float(np.median(observed))
    floor = models.mariner.floor_value(observed, models.mariner.FloorSpec(float("nan"), 0.0, 1.0))
    return median, median - floor


def head_response_jacobian(parameters: np.ndarray, matrix: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the native clipped link and its conditional head derivatives."""
    linear = parameters[0] + matrix @ parameters[1:]
    deficit = np.exp(np.clip(linear, -models.mariner.LOG_CLIP, models.mariner.LOG_CLIP))
    active = np.abs(linear) < models.mariner.LOG_CLIP
    jacobian = deficit[:, None] * np.column_stack([np.ones(len(matrix)), matrix]) * active[:, None]
    return floor + deficit, jacobian


def saved_state(task: int, fold: int, family: str) -> State:
    path = PRIOR / f"fits/fold{fold}_task{task}/{family}/fit.json"
    fit = models.JointFit.from_json(json.loads(path.read_text()))
    return State(
        family,
        fit.alpha,
        fit.shape,
        float(fit.parameters.get("kappa", 0.0)),
        fit.head.floor,
        fit.head.intercept,
        fit.head.coefficients,
    )


def state_diagnostics(state: State, panel: Panel, observed: np.ndarray) -> dict[str, Any]:
    matrix = state.design(panel.early, panel.late)
    linear = state.intercept + matrix @ state.coefficients
    predicted = state.predict(panel.early, panel.late)
    native_benefit = models.mariner.benefit(panel.early + panel.late, state.shape["rate"], state.shape["power"])
    if not np.isfinite(predicted).all():
        raise FloatingPointError("Nonfinite diagnostic prediction")
    return {
        "model": state.to_json(),
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "prediction": predicted.tolist(),
        "clipped_prediction_rows": np.flatnonzero(np.abs(linear) >= models.mariner.LOG_CLIP).tolist(),
        "clipped_prediction_fraction": float(np.mean(np.abs(linear) >= models.mariner.LOG_CLIP)),
        "native_benefit_endpoints_zero": int(np.sum(native_benefit == 0)),
        "native_benefit_endpoints_one": int(np.sum(native_benefit == 1)),
        "transformed_benefit_endpoints_zero": int(np.sum(matrix[:, :2] == 0)),
        "transformed_benefit_endpoints_one": int(np.sum(matrix[:, :2] == -1)),
        "floor_above_outcome_count": int(np.sum(observed < state.floor)),
        "nonnegative_coefficients": bool(np.all(state.coefficients >= 0)),
        "finite_parameters": bool(
            np.isfinite([state.floor, state.intercept, state.strength, *state.shape.values(), *state.coefficients]).all()
        ),
    }


def solver_record(result: Any, lower: np.ndarray, upper: np.ndarray, initial_residual: np.ndarray) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "njev": int(result.njev or 0),
        "cost": float(result.cost),
        "initial_cost": float(0.5 * initial_residual @ initial_residual),
        "cost_change": float(result.cost - 0.5 * initial_residual @ initial_residual),
        "optimality": float(result.optimality),
        "active_mask": result.active_mask.tolist(),
        "feasible": bool(np.all(result.x >= lower) and np.all(result.x <= upper)),
        "parameters": result.x.tolist(),
    }


def solve_head(state: State, panel: Panel, observed: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    matrix = state.design(panel.early, panel.late)
    start = np.r_[state.intercept, state.coefficients]
    lower, upper = np.r_[-np.inf, np.zeros(4)], np.full(5, np.inf)
    result = least_squares(
        lambda values: head_response_jacobian(values, matrix, state.floor)[0] - observed,
        start,
        jac=lambda values: head_response_jacobian(values, matrix, state.floor)[1],
        bounds=(lower, upper),
        **config,
    )
    endpoint = replace(state, intercept=float(result.x[0]), coefficients=result.x[1:])
    return {
        "start": state_diagnostics(state, panel, observed),
        "endpoint": state_diagnostics(endpoint, panel, observed),
        "solver": solver_record(result, lower, upper, state.predict(panel.early, panel.late) - observed),
        "fixed_floor_shape_and_strength": True,
    }


def joint_start(state: State, observed: np.ndarray, bounds: dict[str, list[float]]) -> tuple[np.ndarray, dict[str, Any]]:
    median, gap = floor_contract(observed)
    original_gamma = (median - state.floor) / gap
    gamma = float(np.clip(original_gamma, *bounds["gamma"]))
    parameters = np.r_[
        state.intercept, state.coefficients, gamma, [state.shape[key] for key in ("rate", "power", "threshold")]
    ]
    if state.family != "aggregate_mariner":
        parameters = np.r_[parameters, state.strength]
    return parameters, {
        "full_panel_median": median,
        "native_full_panel_gap": gap,
        "original_floor": state.floor,
        "implied_gamma_before_projection": original_gamma,
        "gamma_after_projection": gamma,
        "projected_floor": median - gamma * gap,
        "floor_was_projected": bool(gamma != original_gamma),
        "intercept_and_amplitudes_changed_during_projection": False,
    }


def joint_state(template: State, values: np.ndarray, observed: np.ndarray) -> State:
    median, gap = floor_contract(observed)
    return replace(
        template,
        intercept=float(values[0]),
        coefficients=values[1:5],
        floor=median - float(values[5]) * gap,
        shape=dict(zip(("rate", "power", "threshold"), map(float, values[6:9]), strict=True)),
        strength=0.0 if template.family == "aggregate_mariner" else float(values[9]),
    )


def solve_joint(
    state: State, panel: Panel, observed: np.ndarray, config: dict[str, Any], bounds: dict[str, list[float]]
) -> dict[str, Any]:
    start, projection = joint_start(state, observed, bounds)
    names = ("gamma", "rate", "power", "threshold")
    if state.family != "aggregate_mariner":
        names += ("kappa",)
    lower = np.r_[-np.inf, np.zeros(4), [bounds[name][0] for name in names]]
    upper = np.r_[np.full(5, np.inf), [bounds[name][1] for name in names]]
    projected = joint_state(state, start, observed)
    result = least_squares(
        lambda values: joint_state(state, values, observed).predict(panel.early, panel.late) - observed,
        start,
        jac="3-point",
        bounds=(lower, upper),
        **config,
    )
    endpoint = joint_state(state, result.x, observed)
    return {
        "before_projection": state_diagnostics(state, panel, observed),
        "projection": projection,
        "start": state_diagnostics(projected, panel, observed),
        "endpoint": state_diagnostics(endpoint, panel, observed),
        "solver": solver_record(result, lower, upper, projected.predict(panel.early, panel.late) - observed),
    }


def cached_record(path: Path, identity: dict[str, str]) -> dict[str, Any] | None:
    receipt_path = path.with_suffix(".complete.json")
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt["identity"] != identity or receipt["output"] != benchmark.receipt(path):
        raise ValueError(f"Invalid completed diagnostic record: {path}")
    return json.loads(path.read_text())


def save_record(path: Path, payload: dict[str, Any], identity: dict[str, str]) -> None:
    benchmark.write_json(path, payload)
    benchmark.write_json(path.with_suffix(".complete.json"), {"identity": identity, "output": benchmark.receipt(path)})


def best_record(records: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    candidates = []
    for name, record in records:
        if "solver" in record and not record["solver"]["feasible"]:
            raise ValueError("An infeasible endpoint cannot enter the diagnostic menu")
        for endpoint in ("start", "endpoint"):
            if endpoint in record:
                candidates.append((float(record[endpoint]["rmse"]), name, endpoint, record[endpoint]))
    _, start_name, endpoint_name, selected = min(candidates, key=lambda item: item[:3])
    return {
        **selected,
        "selected_start": start_name,
        "selected_endpoint": endpoint_name,
        "selection_rule": "minimum all-training BPB RMSE; name then endpoint tie-break; includes untouched starts",
    }


def run_task(task: int) -> dict[str, Any]:
    identity = verified_identity()
    require_approval(identity)
    panel = load_panel()
    spec = registration()
    target = spec["targets"][task]
    observed = panel.outcomes[target].to_numpy(dtype=float)
    task_root = OUTPUT / f"fits/task{task}"
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            states = {f"fold{fold}": saved_state(task, fold, family) for fold in spec["folds"]}
            stages: dict[str, list[tuple[str, dict[str, Any]]]] = {"A_saved": [], "B_head": [], "C_joint": []}
            for name, state in states.items():
                saved_path = task_root / family / "A_saved" / f"{name}.json"
                saved = cached_record(saved_path, identity)
                if saved is None:
                    saved = {"start": state_diagnostics(state, panel, observed)}
                    save_record(saved_path, saved, identity)
                stages["A_saved"].append((name, saved))
                path = task_root / family / "B_head" / f"{name}.json"
                record = cached_record(path, identity)
                if record is None:
                    record = solve_head(state, panel, observed, spec["solver"]["B_head"])
                    save_record(path, record, identity)
                stages["B_head"].append((name, record))
            for stage in ("A_saved", "B_head"):
                if family != "aggregate_mariner":
                    retained = selected[("aggregate_mariner", stage)]
                    null = replace(State.from_json(retained["model"]), family=family, strength=0.0)
                    stages[stage].append(("retained_aggregate", {"start": state_diagnostics(null, panel, observed)}))
                selected[(family, stage)] = best_record(stages[stage])
            starts = {
                name: State.from_json(best_record([(name, record)])["model"])
                for name, record in stages["B_head"]
                if name != "retained_aggregate"
            }
            if family != "aggregate_mariner":
                null = replace(
                    State.from_json(selected[("aggregate_mariner", "C_joint")]["model"]), family=family, strength=0.0
                )
                starts["retained_aggregate"] = null
                stages["C_joint"].append(
                    ("retained_aggregate_untouched", {"start": state_diagnostics(null, panel, observed)})
                )
            for name, state in starts.items():
                path = task_root / family / "C_joint" / f"{name}.json"
                record = cached_record(path, identity)
                if record is None:
                    record = solve_joint(state, panel, observed, spec["solver"]["C_joint"], spec["bounds"])
                    save_record(path, record, identity)
                stages["C_joint"].append((name, record))
            selected[(family, "C_joint")] = best_record(stages["C_joint"])
            for stage, records in stages.items():
                record = {
                    "target": target,
                    "family": family,
                    "stage": stage,
                    "all_rows_training": True,
                    "selected": selected[(family, stage)],
                    "starts": [name for name, _ in records],
                    "retained_aggregate_is_separate_nested_baseline": family != "aggregate_mariner",
                    "nonconverged_starts": [
                        name for name, item in records if "solver" in item and not item["solver"]["success"]
                    ],
                }
                save_record(task_root / family / f"{stage}_selected.json", record, identity)
            print(
                json.dumps(
                    {
                        "target": target,
                        "family": family,
                        "training_rmse": {stage: selected[(family, stage)]["rmse"] for stage in stages},
                    }
                ),
                flush=True,
            )
    return {"task": task, "target": target, "complete": True}


def strata(panel: Panel) -> dict[str, np.ndarray]:
    """Predefined policy-only subsets; none are excluded from fitting."""
    regions = {"all": np.ones(len(panel.rows), dtype=bool)}
    tied = models.frozen.temporal.tied_rows(panel.early, panel.late, panel.alpha)
    regions.update({"tied": tied, "asymmetric": ~tied})
    for index, name in enumerate(("web", "code")):
        late = panel.phase1[:, index]
        regions[f"{name}_late_zero"] = late == 0
        regions[f"{name}_late_low_positive"] = (late > 0) & (late <= 0.05)
        regions[f"{name}_late_interior"] = (late > 0.05) & (late < 0.95)
        regions[f"{name}_late_high_below_one"] = (late >= 0.95) & (late < 1)
        regions[f"{name}_late_one"] = late == 1
    aggregate = 0.8 * panel.phase0[:, 1] + 0.2 * panel.phase1[:, 1]
    edges = registration()["descriptive_strata"]["aggregate_bin_edges"]
    for index, (low, high) in enumerate(itertools.pairwise(edges)):
        mask = (aggregate >= low) & (aggregate <= high if index == len(edges) - 2 else aggregate < high)
        regions[f"code_aggregate_{low:g}_{high:g}"] = mask
    for value in registration()["descriptive_strata"]["nominal_fibers"]:
        regions[f"nominal_code_fiber_{value:g}"] = np.isclose(aggregate, value, rtol=0, atol=1e-10)
    return regions


def training_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - observed
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "signed_bias": float(np.mean(error)),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def summarize() -> None:
    identity = verified_identity()
    require_approval(identity)
    panel = load_panel()
    scores, predictions, statuses = [], [], []
    for task, target in enumerate(registration()["targets"]):
        observed = panel.outcomes[target].to_numpy(dtype=float)
        for family in FAMILIES:
            for stage in ("A_saved", "B_head", "C_joint"):
                path = OUTPUT / f"fits/task{task}/{family}/{stage}_selected.json"
                record = cached_record(path, identity)
                if record is None:
                    raise ValueError(f"Incomplete diagnostic: {path}")
                prediction = np.asarray(record["selected"]["prediction"])
                state = State.from_json(record["selected"]["model"])
                np.testing.assert_array_equal(prediction, state.predict(panel.early, panel.late))
                for name, mask in strata(panel).items():
                    if mask.any():
                        scores.append(
                            {
                                "target": target,
                                "family": family,
                                "stage": stage,
                                "stratum": name,
                                "all_rows_training": True,
                                **training_metrics(observed[mask], prediction[mask]),
                            }
                        )
                frame = panel.rows.copy()
                frame["target"], frame["family"], frame["stage"] = target, family, stage
                frame["observed"], frame["predicted"] = observed, prediction
                frame["role"] = "all_training_diagnostic"
                predictions.append(frame)
                statuses.append(
                    {
                        "target": target,
                        "family": family,
                        "stage": stage,
                        "selected_start": record["selected"]["selected_start"],
                        "nonconverged_starts": record["nonconverged_starts"],
                        "selected_clipped_rows": len(record["selected"]["clipped_prediction_rows"]),
                        "selected_floor_above_outcomes": record["selected"]["floor_above_outcome_count"],
                    }
                )
    pd.DataFrame(scores).to_csv(OUTPUT / "training_scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT / "training_predictions.csv", index=False)
    benchmark.write_json(OUTPUT / "solver_statuses.json", statuses)
    benchmark.write_json(
        OUTPUT / "summary_manifest.json",
        {
            "identity": identity,
            "claim": (
                "Feasible C winners bound attainable all-training RMSE from above within C bounds. "
                "A/B use historical fixed floors, possibly outside C bounds. No result is an expressivity ceiling."
            ),
            "out_of_fold_or_generalization_claim": False,
            "promotion": False,
            "outputs": [
                benchmark.receipt(OUTPUT / name)
                for name in ("training_scores.csv", "training_predictions.csv", "solver_statuses.json")
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    if args.command == "summarize":
        summarize()
        return
    tasks = list(range(len(registration()["targets"]))) if args.task is None else [args.task]
    if args.workers < 1 or not set(tasks).issubset(range(len(registration()["targets"]))):
        raise ValueError("Invalid task or worker count")
    started = time.monotonic()
    if args.workers == 1:
        for task in tasks:
            run_task(task)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed(executor.submit(run_task, task) for task in tasks):
                print(json.dumps(future.result()), flush=True)
    print(json.dumps({"elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
