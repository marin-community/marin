# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Compare frozen, fixed-floor response, and joint-floor response heads offline.

Reuses the frozen procedure's fold-specific shape/ridge choices. No held-out
outcome participates in fitting. This is a conditional head ablation, not a
retuned replacement-method benchmark. Durable per-task outputs permit resume.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/joint_floor_comparison_20260908"
STANDALONE = Path("/Users/calvinxu/Projects/Work/Marin/mixture-selection")
REFERENCE = HERE / "reference_outputs/delphi_corrected_screen_20260908"
REFERENCE_MODEL = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"
VARIANTS = ("frozen", "response_fixed", "response_joint")
INTERVENTIONS = {"conditional_epoch_dose_response", "archive::delphi_baseline_mixtures_issue6607_20260623"}
PARITY_TOLERANCE = 1e-8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@cache
def load_module(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def inputs(output: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    folder = Path(output) / "inputs"
    baseline = load_module(str(folder / "frozen_mixture_selection.py"), "frozen_mixture_selection")
    solver = load_module(str(folder / "joint_floor_solver.py"), "joint_floor_solver")
    swarm = baseline.read_swarm(folder / "swarm_weights.csv", folder / "swarm_outcomes.csv", folder / "buckets.csv")
    objectives = baseline.read_objectives(folder / "objectives.csv")
    return baseline, solver, swarm, objectives


def prepare(output: Path) -> None:
    folder = output / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    source_map = {folder / p.name: p for p in (STANDALONE / "data").glob("*.csv")}
    source_map[folder / "frozen_mixture_selection.py"] = STANDALONE / "mixture_selection.py"
    source_map[folder / "joint_floor_solver.py"] = HERE / "joint_floor_solver_20260908.py"
    shard_root = REFERENCE / "baseline_shards" / REFERENCE_MODEL
    for path in shard_root.glob("*/*.npz"):
        source_map[folder / "shards" / path.relative_to(shard_root)] = path
    for destination, source in source_map.items():
        if destination.exists():
            if sha256(destination) != sha256(source):
                raise ValueError(f"frozen input differs from current source: {source}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    manifest = {
        "protocol": "Fixed fold-specific shape/ridge; frozen NNLS/CV floor, BPB fixed-floor control, BPB joint floor.",
        "gamma_bounds": [1, 6],
        "noise_margin_sds": 3,
        "response_scale": "mean(y_train - frozen_floor), fixed across gamma and starts",
        "response_objective": "0.5 * (sum(((prediction-y)/scale)^2) + ridge*sum(coefficients^2))",
        "gamma_starts": "incumbent,1,1.5,2.5,6; joint also starts at fixed-response solution",
        "outer_repeat": 0,
        "outer_folds": 5,
        "candidate_tuning": "none; joint gamma minimizes training objective inside [1,6], no boundary reset",
        "evaluation_role": "Frozen bank and earlier 23 validations are retrospective development diagnostics.",
        "new_lm_training": False,
        "files": {str(p.relative_to(output)): {"source": str(s), "sha256": sha256(p)} for p, s in source_map.items()},
    }
    write_json(output / "manifest.json", manifest)
    print(f"prepared {len(source_map)} frozen files", flush=True)


def frozen_head(baseline: Any, swarm: Any, output: Path, objective: str, index: int, fold: int) -> tuple[Any, ...]:
    shard = output / "inputs/shards" / objective / f"r0_f{fold}_c{index}.npz"
    with np.load(shard, allow_pickle=False) as data:
        train, test = data["train"].copy(), data["test"].copy()
        shape = json.loads(str(data["shape_json"]))
        ridge = float(data["ridge"])
        diagnostics = json.loads(str(data["diagnostics_json"]))
        saved_train, saved_test = data["train_prediction"].copy(), data["prediction"].copy()
    component = baseline.read_objectives(output / "inputs/objectives.csv")[objective].components[index]
    anchor = baseline.read_anchors(output / "inputs/anchors.csv", objective)[component]
    matrix = baseline.design_matrix(swarm.exposures, shape)
    response = swarm.outcomes[component].to_numpy(float)
    gamma = float(diagnostics["kappa"])
    head = baseline.fit_head(
        matrix[train], response[train], ridge, baseline.FloorSpec(anchor.proportional, anchor.repeat_sd, gamma)
    )
    error = float(np.max(np.abs(head.predict(matrix[train]) - saved_train)))
    if len(test):
        error = max(error, float(np.max(np.abs(head.predict(matrix[test]) - saved_test))))
    if error > PARITY_TOLERANCE:
        raise ValueError(f"frozen parity {objective}/{component}/{fold}: {error}")
    if not np.all(np.isin(np.flatnonzero(swarm.calibration), train)) or np.any(swarm.calibration[test]):
        raise ValueError("calibration row crossed validation boundary")
    return component, train, test, shape, ridge, diagnostics, anchor, matrix, response, head, error


def head_payload(head: Any, gamma: float) -> dict[str, Any]:
    return {"intercept": head.intercept, "coefficients": head.coefficients.tolist(), "floor": head.floor, "gamma": gamma}


def fit_one(job: tuple[str, str, int, int, int, float]) -> dict[str, Any]:
    output_name, objective, index, fold, max_nfev, tolerance = job
    output = Path(output_name)
    path = output / "tasks" / objective / f"r0_f{fold}_c{index}.json"
    if path.exists():
        return {"cached": True, "path": str(path)}
    started = time.monotonic()
    baseline, solver, swarm, _ = inputs(output_name)
    component, train, test, shape, ridge, diagnostics, anchor, matrix, response, frozen, parity = frozen_head(
        baseline, swarm, output, objective, index, fold
    )
    gamma = float(diagnostics["kappa"])
    common = dict(
        matrix=matrix[train],
        response=response[train],
        ridge=ridge,
        anchor=anchor.proportional,
        noise_sd=anchor.repeat_sd,
        incumbent_gamma=gamma,
        conditional_solve=baseline.nonnegative_solve,
        max_nfev=max_nfev,
        tolerance=tolerance,
    )
    fixed = solver.fit_response_head(**common, mode="fixed")
    warm = solver.ResponseStart(fixed.intercept, fixed.coefficients, gamma)
    joint = solver.fit_response_head(**common, mode="joint", extra_starts=(warm,))
    if joint.cost > fixed.cost + 1e-8 * max(1, fixed.cost):
        raise ValueError(f"joint objective worse than nested fixed solution: {component} {fold}")
    variants = {}
    for name, head in zip(VARIANTS, (frozen, fixed, joint), strict=True):
        item = head_payload(head, gamma if name == "frozen" else head.gamma)
        item["test_prediction"] = head.predict(matrix[test]).tolist()
        item["train_rmse"] = float(np.sqrt(np.mean((head.predict(matrix[train]) - response[train]) ** 2)))
        if name != "frozen":
            item["solver"] = {
                k: v
                for k, v in dataclasses.asdict(head).items()
                if k not in {"coefficients", "intercept", "floor", "gamma"}
            }
        variants[name] = item
    result = {
        "objective": objective,
        "component": component,
        "component_index": index,
        "fold": fold,
        "train": train.tolist(),
        "test": test.tolist(),
        "shape": shape,
        "ridge": ridge,
        "anchor": dataclasses.asdict(anchor),
        "frozen_diagnostics": diagnostics,
        "frozen_parity": parity,
        "max_nfev": max_nfev,
        "tolerance": tolerance,
        "variants": variants,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(path, result)
    return {
        "cached": False,
        "path": str(path),
        "elapsed": result["elapsed_seconds"],
        "fixed_ok": fixed.success,
        "joint_ok": joint.success,
        "gamma": joint.gamma,
    }


def fit_all(args: argparse.Namespace) -> None:
    _, _, _, objectives = inputs(str(args.output))
    folds = [-1] if args.final_only else [-1, 0, 1, 2, 3, 4]
    jobs = [
        (str(args.output), o.name, i, f, args.max_nfev, args.tolerance)
        for f in folds
        for o in objectives.values()
        for i in range(len(o.components))
    ]
    if args.limit:
        jobs = jobs[: args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_one, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            print(json.dumps({"done": count, "total": len(jobs), **future.result()}), flush=True)


def fitted_objective(output: Path, objective: str, variant: str) -> Any:
    baseline, _, swarm, objectives = inputs(str(output))
    tasks = []
    for index, component in enumerate(objectives[objective].components):
        row = json.loads((output / "tasks" / objective / f"r0_f-1_c{index}.json").read_text())
        head = row["variants"][variant]
        tasks.append(
            baseline.TaskFit(
                component,
                row["shape"],
                row["ridge"],
                head["gamma"],
                False,
                row["frozen_diagnostics"]["inner_cv_rmse"],
                baseline.Head(head["intercept"], np.asarray(head["coefficients"]), head["floor"]),
            )
        )
    return baseline.ObjectiveFit(objective, swarm.buckets, swarm.inventory, objectives[objective].weights, tuple(tasks))


def metric_values(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    order = np.argsort(predicted, kind="stable")
    pick = int(order[0])
    return {
        "n": len(measured),
        "rank": int(np.sum(measured <= measured[pick])),
        "regret": float(measured[pick] - measured.min()),
        "selected_measured": float(measured[pick]),
        "selected_predicted": float(predicted[pick]),
        "optimism": float(measured[pick] - predicted[pick]),
        "regret_best5": float(measured[order[:5]].min() - measured.min()),
        "regret_best10": float(measured[order[:10]].min() - measured.min()),
        "rmse": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "bias_pred_minus_measured": float(np.mean(predicted - measured)),
        "spearman": float(spearmanr(predicted, measured).statistic),
    }


def evaluate(output: Path) -> None:
    _, _, swarm, objectives = inputs(str(output))
    bank_rows, oof_rows, component_rows, solver_rows, fresh_rows = [], [], [], [], []
    for objective in objectives.values():
        bank = pd.read_csv(output / "inputs" / f"heldout_bank_{objective.name}.csv")
        weights = bank[list(swarm.buckets)].to_numpy(float)
        measured = bank.measured_mean_bpb.to_numpy(float)
        max_epochs = (weights * swarm.inventory).max(axis=1)
        optima = np.array(
            [
                not (set(json.loads(s) if str(s).startswith("[") else str(s).split(";")) & INTERVENTIONS)
                for s in bank.sources
            ]
        )
        masks = {
            "all": np.ones(len(bank), bool),
            "optima": optima,
            "cap6": max_epochs <= 6 + 1e-8,
            "cap8": max_epochs <= 8 + 1e-8,
        }
        fresh_path = output / "inputs/fresh_runs.csv"
        fresh = pd.read_csv(fresh_path) if fresh_path.exists() else None
        for variant in VARIANTS:
            fit = fitted_objective(output, objective.name, variant)
            write_json(output / "fits" / f"{objective.name}_{variant}.json", fit.to_json())
            prediction = fit.predict(weights)
            bank[f"prediction_{variant}"] = prediction
            for population, mask in masks.items():
                selected = np.flatnonzero(mask)[np.argmin(prediction[mask])]
                bank_rows.append(
                    {
                        "objective": objective.name,
                        "variant": variant,
                        "population": population,
                        "selected_id": bank.coordinate_id.iloc[selected],
                        **metric_values(prediction[mask], measured[mask]),
                    }
                )
            if variant == "frozen":
                reference = pd.read_csv(output / "inputs" / f"reference_bank_predictions_{objective.name}.csv")
                reference = reference.set_index("coordinate_id").loc[bank.coordinate_id]
                reference_prediction = reference[list(objective.components)].to_numpy(float) @ objective.weights
                error = float(np.max(np.abs(prediction - reference_prediction)))
                if error > PARITY_TOLERANCE:
                    raise ValueError(f"bank parity {objective.name}: {error}")
            if fresh is not None:
                fp = fit.predict(fresh[list(swarm.buckets)].to_numpy(float))
                fy = fresh[f"measured_{objective.name}"].to_numpy(float)
                for j, row in fresh.iterrows():
                    fresh_rows.append(
                        {
                            "objective": objective.name,
                            "variant": variant,
                            "candidate_id": row.candidate_id,
                            "launch": row.launch,
                            "target": row.target,
                            "predicted": float(fp[j]),
                            "measured": float(fy[j]),
                        }
                    )
            oof = np.full((len(swarm.runs), len(objective.components)), np.nan)
            for index, component in enumerate(objective.components):
                final = json.loads((output / "tasks" / objective.name / f"r0_f-1_c{index}.json").read_text())
                head = final["variants"][variant]
                component_rows.append(
                    {
                        "objective": objective.name,
                        "component": component,
                        "variant": variant,
                        "gamma": head["gamma"],
                        "floor": head["floor"],
                        "ridge": final["ridge"],
                        "train_rmse": head["train_rmse"],
                    }
                )
                for fold in [-1, 0, 1, 2, 3, 4]:
                    path = output / "tasks" / objective.name / f"r0_f{fold}_c{index}.json"
                    if not path.exists():
                        continue
                    task = json.loads(path.read_text())
                    h = task["variants"][variant]
                    if "solver" in h:
                        s = h["solver"]
                        solver_rows.append(
                            {
                                "objective": objective.name,
                                "component": component,
                                "fold": fold,
                                "variant": variant,
                                "gamma": h["gamma"],
                                "success": s["success"],
                                "cost": s["cost"],
                                "optimality": s["optimality"],
                                "nfev": s["nfev"],
                                "total_nfev": sum(x["nfev"] for x in s["starts"]),
                                "starts_success": sum(x["success"] for x in s["starts"]),
                                "starts": len(s["starts"]),
                                "elapsed_task": task["elapsed_seconds"],
                            }
                        )
                    if fold >= 0:
                        oof[np.asarray(task["test"]), index] = h["test_prediction"]
            valid = np.all(np.isfinite(oof), axis=1)
            if np.any(valid):
                predicted = oof[valid] @ objective.weights
                observed = swarm.outcomes[list(objective.components)].to_numpy(float)[valid] @ objective.weights
                oof_rows.append({"objective": objective.name, "variant": variant, **metric_values(predicted, observed)})
                pd.DataFrame(
                    {"run": np.asarray(swarm.runs)[valid], "predicted": predicted, "measured": observed}
                ).to_csv(output / f"oof_predictions_{objective.name}_{variant}.csv", index=False)
        bank.to_csv(output / f"bank_predictions_{objective.name}.csv", index=False)
    for name, rows in [
        ("bank_metrics", bank_rows),
        ("oof_metrics", oof_rows),
        ("component_fits", component_rows),
        ("solver_diagnostics", solver_rows),
        ("fresh_predictions", fresh_rows),
    ]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    if fresh_rows:
        records = []
        table = pd.DataFrame(fresh_rows)
        for (objective, variant), group in table.groupby(["objective", "variant"]):
            for population in ["all", "targeted"]:
                g = group if population == "all" else group[group.target.eq(objective)]
                error = g.predicted - g.measured
                records.append(
                    {
                        "objective": objective,
                        "variant": variant,
                        "population": population,
                        "n": len(g),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "bias_pred_minus_measured": float(error.mean()),
                        "spearman": float(spearmanr(g.predicted, g.measured).statistic),
                    }
                )
        pd.DataFrame(records).to_csv(output / "fresh_metrics.csv", index=False)
    print(pd.DataFrame(bank_rows).query("population == 'optima'").to_string(index=False), flush=True)


def optimize(output: Path) -> None:
    baseline, _, swarm, objectives = inputs(str(output))
    records, policies = [], []
    for objective in objectives:
        fits = {v: fitted_objective(output, objective, v) for v in VARIANTS}
        for cap in ([6.0, 1e9] if objective == "uncheatable" else [6.0, 8.0, 1e9]):
            for variant, fit in fits.items():
                name = f"{objective}_{variant}_cap{cap:g}"
                path = output / "policies" / f"{name}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                else:
                    continuous, starts = baseline.continuous_optimum(fit, swarm.weights, cap, 0.0)
                    counts, diagnostics = baseline.runtime_policy(fit, continuous, cap, 0.0)
                    row = {
                        "name": name,
                        "objective": objective,
                        "variant": variant,
                        "cap": cap,
                        "counts": counts.tolist(),
                        "continuous": continuous.tolist(),
                        "starts": starts,
                        **diagnostics,
                        "policy_sha256": hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
                    }
                    write_json(path, row)
                weights = np.asarray(row["counts"], float) / baseline.MIXTURE_BLOCK_SIZE
                reference_path = output / "policies" / f"{objective}_frozen_cap{cap:g}.json"
                reference_counts = np.asarray(json.loads(reference_path.read_text())["counts"])
                records.append(
                    {k: v for k, v in row.items() if k not in {"counts", "continuous", "starts"}}
                    | {f"predicted_by_{v}": float(f.predict(weights[None])[0]) for v, f in fits.items()}
                    | {"tv_from_frozen": float(np.abs(np.asarray(row["counts"]) - reference_counts).sum() / 4096)}
                )
                policies.append({"name": name, **dict(zip(swarm.buckets, weights.tolist(), strict=True))})
                pd.DataFrame(records).to_csv(output / "policy_summary.csv", index=False)
                pd.DataFrame(policies).to_csv(output / "policy_weights.csv", index=False)
                print(json.dumps(records[-1]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "fit", "evaluate", "optimize"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "fit":
        fit_all(args)
    elif args.phase == "evaluate":
        evaluate(args.output)
    else:
        optimize(args.output)


if __name__ == "__main__":
    main()
