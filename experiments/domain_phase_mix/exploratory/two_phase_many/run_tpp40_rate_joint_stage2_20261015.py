# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Separate, resumable learned-J2 synthetic stage for the qualified rate law.

Stage-one sources and outputs are immutable. This runner reads their seal,
uses policy covariates and synthetic truth only, and requires a separate exact
execution approval. The four noiseless worlds yield sixteen fitted heads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_tangent_round5_20261015 as scoring
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as joint
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_joint_models_20261015 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_reopening_20261015 as stage1
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_optimization_20261015 as optimization

ROOT = Path(__file__).resolve().parent
OUTPUT = stage1.OUTPUT / "stage2"
STAGE1_SEAL = "d101b3c8eef79f61b1591e851f7a369a3ce8af5c67aa6a60c80939bcf4c3ce4c"
FAMILIES = stage1.FAMILIES
WAVES = (FAMILIES[:1], FAMILIES[1:3], FAMILIES[3:])


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    parent = stage1.registration()
    if value["candidate_id"] != "TP40-026" or value["stage"] != 2 or value["status"] != "registered":
        raise ValueError("Stage two requires its separately registered protocol")
    expected = {
        "families": parent["families"],
        "worlds": parent["worlds"],
        "kappa_grid": parent["kappa_grid"],
        "nu_grid": parent["nu_grid"],
        "gamma_grid": list(joint.GAMMA_GRID),
        "noise_sd": 0.0,
        "stage1_result_manifest_sha256": STAGE1_SEAL,
        "stage2_gates": parent["whole_estimator_plan"]["stage2_gates"],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Stage-two protocol differs from the qualified four-world comparison")
    if value["environment"] != benchmark.environment_identity():
        raise ValueError("Numerical environment differs from registration")
    benchmark.verify_hashes(value["implementation_hashes"])
    return value


def source_identity() -> dict[str, Any]:
    paths = set(stage1.source_identity()["sources"])
    paths.update(
        str(path.resolve())
        for path in (
            Path(__file__),
            Path(models.__file__),
            Path(scoring.__file__),
            Path(optimization.__file__),
            OUTPUT / "registration.json",
            stage1.OUTPUT / "result_manifest.json",
            stage1.OUTPUT / "oracle_summary.json",
            stage1.OUTPUT / "preflight.json",
            stage1.OUTPUT / "execution_identity.json",
            stage1.OUTPUT / "audit/final_report_v1.json",
            scoring.OUTPUT / "registration.json",
        )
    )
    return {
        "sources": {path: benchmark.sha256(Path(path)) for path in sorted(paths)},
        "environment": benchmark.environment_identity(),
    }


def freeze() -> None:
    """Validate adoption and seal the source set without generating synthetic data."""
    registration()
    seal_path = stage1.OUTPUT / "result_manifest.json"
    if benchmark.sha256(seal_path) != STAGE1_SEAL:
        raise ValueError("Stage-one audited seal changed")
    seal = json.loads(seal_path.read_text())
    benchmark.verify_hashes({name: item["sha256"] for name, item in seal["files"].items()}, stage1.OUTPUT)
    oracle = json.loads((stage1.OUTPUT / "oracle_summary.json").read_text())
    if not oracle["stage2_prerequisites_satisfied"]:
        raise ValueError("Stage-one oracle and advancement checks must pass")
    old = json.loads((stage1.OUTPUT / "execution_identity.json").read_text())
    if stage1.source_identity() != old:
        raise ValueError("Stage-one scientific sources or data no longer match its frozen identity")
    round5 = json.loads((scoring.OUTPUT / "execution_identity.json").read_text())
    for module in (scoring, optimization):
        path = Path(module.__file__).resolve()
        if benchmark.sha256(path) != round5[str(path)]:
            raise ValueError(f"Frozen round-five evaluation dependency changed: {path}")
    # The reused scorer's only protocol lookup is the registered zero-gain tolerance.
    if scoring.registration()["metrics"]["gain_sign_zero_tolerance"] != 1e-10:
        raise ValueError("Reused scoring contract changed")
    current = source_identity()
    destination = OUTPUT / "execution_identity.json"
    if destination.exists() and json.loads(destination.read_text()) != current:
        raise ValueError("Preserve the prior stage-two identity; register a separate revision")
    benchmark.write_json(destination, current)


def execution_identity() -> dict[str, Any]:
    expected = json.loads((OUTPUT / "execution_identity.json").read_text())
    if source_identity() != expected:
        raise ValueError("Stage-two frozen source, data or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    required = {
        "scope": "rate_reopening_stage2_noiseless",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in required.items()):
        raise ValueError("A separate exact stage-two execution approval is required")
    return expected


def completed(folder: Path, spec: dict[str, Any], identity: dict[str, Any]) -> bool:
    receipt = folder / "complete.json"
    if not receipt.exists():
        return False
    value = json.loads(receipt.read_text())
    if value["identity"] != identity or value["spec"] != spec:
        raise ValueError(f"Mismatched completed cell: {folder}")
    benchmark.verify_hashes(value["outputs"], folder)
    benchmark.verify_hashes(value["dependencies"])
    return True


def finish(
    folder: Path,
    spec: dict[str, Any],
    identity: dict[str, Any],
    files: tuple[str, ...],
    dependencies: dict[str, str],
    **extra: Any,
) -> None:
    if execution_identity() != identity:
        raise ValueError("Identity changed during execution")
    benchmark.verify_hashes(dependencies)
    benchmark.write_json(
        folder / "complete.json",
        {
            "identity": identity,
            "spec": spec,
            "outputs": {name: benchmark.sha256(folder / name) for name in files},
            "dependencies": dependencies,
            **extra,
        },
    )


def prepare() -> None:
    identity = execution_identity()
    arrays, folds, truth, head = stage1.inputs()
    for world in registration()["worlds"]:
        folder = OUTPUT / "contexts" / world["name"]
        if completed(folder, world, identity):
            continue
        values = stage1.world_inputs(arrays, folds, truth, head, world["parameters"])
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "synthetic.npz",
            train=values["train"],
            asymmetric=values["held"],
            tied=values["tied"],
            inner_labels=values["labels"],
            clean=values["clean"],
            observed=values["clean"],
            clean_tied=values["clean_tied"],
        )
        benchmark.write_json(
            folder / "truth.json",
            {"shape": truth["shape"], "head": truth["head"], "parameters": world["parameters"], "world": world},
        )
        finish(folder, world, identity, ("synthetic.npz", "truth.json"), {})


def load_context(world: dict[str, Any], identity: dict[str, Any]) -> scoring.Context:
    folder = OUTPUT / "contexts" / world["name"]
    if not completed(folder, world, identity):
        raise ValueError(f"Context has not been prepared: {folder}")
    with np.load(folder / "synthetic.npz", allow_pickle=False) as data:
        return scoring.Context(
            world,
            scoring.load_geometry(),
            data["train"],
            data["asymmetric"],
            data["tied"],
            data["inner_labels"],
            data["clean"],
            data["observed"],
            data["clean_tied"],
            json.loads((folder / "truth.json").read_text()),
        )


def cell_spec(world: dict[str, Any], family: str) -> dict[str, Any]:
    return {"world": world, "family": family}


def load_cell(context: scoring.Context, family: str, identity: dict[str, Any]) -> models.JointFit:
    folder = OUTPUT / "contexts" / context.name / "cells" / family
    if not completed(folder, cell_spec(context.spec, family), identity):
        raise ValueError(f"Missing fit: {folder}")
    return models.JointFit.from_json(json.loads((folder / "model.json").read_text()))


def run_cell(world: dict[str, Any], family: str) -> dict[str, Any]:
    identity = execution_identity()
    context = load_context(world, identity)
    folder = OUTPUT / "contexts" / context.name / "cells" / family
    spec = cell_spec(world, family)
    if completed(folder, spec, identity):
        return {**spec, "cached": True}
    controls_names = () if family == "aggregate_mariner" else (FAMILIES[:3] if family == FAMILIES[3] else FAMILIES[:1])
    controls = tuple(load_cell(context, name, identity) for name in controls_names)
    dependencies = {
        str(OUTPUT / "contexts" / context.name / "complete.json"): benchmark.sha256(
            OUTPUT / "contexts" / context.name / "complete.json"
        )
    }
    dependencies.update(
        {
            str(folder.parent / name / "complete.json"): benchmark.sha256(folder.parent / name / "complete.json")
            for name in controls_names
        }
    )
    geometry, train = context.geometry, context.train
    print(f"Fitting {context.name} {family}", flush=True)
    started = time.monotonic()
    execution_identity()
    fit = models.fit_joint(
        geometry.early[train],
        geometry.late[train],
        context.observed[train],
        context.inner_labels,
        geometry.alpha,
        family,
        controls=controls,
    )
    query0, query1 = scoring.query_coordinates(context)
    predictions = fit.predict(query0, query1)
    scores, frame = scoring.prediction_scores(context, predictions)
    matrix, offset = models.design_and_offset(
        geometry.early[train], geometry.late[train], geometry.alpha, fit.shape, family, fit.parameters
    )
    with threadpool_limits(limits=1):
        inner = joint.candidate_oof(matrix, offset, context.observed[train], context.inner_labels, fit.ridge, fit.kappa)
    replay_score = float(np.sqrt(np.mean((inner - context.observed[train]) ** 2)))
    if abs(replay_score - fit.inner_rmse) > 1e-12:
        raise ValueError("Selected inner score does not replay")
    if controls and fit.inner_rmse > min(control.inner_rmse for control in controls) + 1e-12:
        raise ValueError("Expanded search lost a complete nested control")
    folder.mkdir(parents=True, exist_ok=True)
    benchmark.write_json(folder / "model.json", fit.to_json())
    restored = models.JointFit.from_json(json.loads((folder / "model.json").read_text()))
    if not np.array_equal(restored.predict(query0, query1), predictions):
        raise ValueError("Serialized fitted prediction differs")
    frame.to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(
        {"row_index": train, "inner_fold": context.inner_labels, "observed": context.observed[train], "predicted": inner}
    ).to_csv(folder / "inner_predictions.csv", index=False)
    profile = pd.DataFrame(fit.selection_profile)
    profile.sort_values(
        ["inner_rmse", "parameter_index", "shape_index", "ridge_index", "gamma"], kind="stable"
    ).drop_duplicates("parameter_index").to_csv(folder / "kinetic_profiles.csv", index=False)
    benchmark.write_json(
        folder / "scores.json",
        {
            "world": context.name,
            "family": family,
            **scores,
            "inner_rmse": fit.inner_rmse,
            "shape": fit.shape,
            "parameters": fit.parameters,
            "ridge": fit.ridge,
            "gamma": fit.kappa,
            "floor": fit.head.floor,
            "elapsed_seconds": time.monotonic() - started,
            "profile_scope": (
                "Best searched candidate per kinetic cell; floor refinement was not separately repeated for every cell"
            ),
        },
    )
    finish(
        folder,
        spec,
        identity,
        ("model.json", "predictions.csv", "inner_predictions.csv", "kinetic_profiles.csv", "scores.json"),
        dependencies,
    )
    return {**spec, "inner_rmse": fit.inner_rmse}


def summarize() -> None:
    identity = execution_identity()
    rows, checks, receipts = [], [], {}
    for world in registration()["worlds"]:
        context = load_context(world, identity)
        baseline_path = OUTPUT / "contexts" / context.name / "cells/aggregate_mariner/scores.json"
        baseline = json.loads(baseline_path.read_text())
        for family in FAMILIES:
            fit = load_cell(context, family, identity)
            folder = OUTPUT / "contexts" / context.name / "cells" / family
            scores = json.loads((folder / "scores.json").read_text())
            query0, query1 = scoring.query_coordinates(context)
            recomputed, _ = scoring.prediction_scores(context, fit.predict(query0, query1))
            for name, value in recomputed.items():
                if value != scores[name]:
                    raise ValueError(f"Saved prediction score changed: {context.name}/{family}/{name}")
            rows.append(scores)
            is_correct = context.name == "null" or family in (world["generating_family"], "bounded_benefit_rate")
            gates = stage1.registration()["whole_estimator_plan"]["stage2_gates"]
            checks.append(
                {
                    "world": context.name,
                    "family": family,
                    "correct_family_or_combined": is_correct,
                    "null_gain_pass": (
                        context.name != "null"
                        or scores["max_abs_predicted_gain"] <= gates["noiseless_null_max_abs_matched_gain_bpb"]
                    ),
                    "gain_recovery_pass": (
                        not is_correct
                        or scores["clean_gain_rmse"] <= gates["correct_family_and_combined_gain_rmse_max_bpb"]
                    ),
                    "tied_calibration_pass": (
                        not is_correct or scores["clean_actual_tied_rmse"] <= baseline["clean_actual_tied_rmse"] + 0.003
                    ),
                    "kinetic_boundary_pass": (
                        not is_correct or (fit.parameters["kappa"] != 128 and fit.parameters["nu"] != 2)
                    ),
                }
            )
            receipts[str(folder / "complete.json")] = benchmark.sha256(folder / "complete.json")
    pd.json_normalize(rows).to_csv(OUTPUT / "scores.csv", index=False)
    passed = all(
        all(
            item[key]
            for key in ("null_gain_pass", "gain_recovery_pass", "tied_calibration_pass", "kinetic_boundary_pass")
        )
        for item in checks
    )
    benchmark.write_json(
        OUTPUT / "summary.json",
        {
            "identity": identity,
            "cells": len(rows),
            "passed": passed,
            "checks": checks,
            "cell_receipts": receipts,
            "scores_sha256": benchmark.sha256(OUTPUT / "scores.csv"),
            "scope": (
                "Learned-estimator noiseless synthetic gate only; "
                "does not authorize noisy stage, empirical fits or promotion."
            ),
        },
    )


@dataclass(frozen=True)
class SyntheticTruth:
    specification: dict[str, Any]
    alpha: float

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        raw = self.specification["head"]
        matrix = stage1.design(early, late, self.alpha, self.specification["shape"], self.specification["parameters"])
        return raw["floor"] + np.exp(raw["intercept"] + matrix @ np.asarray(raw["coefficients"]))


def optimize_world(world: dict[str, Any]) -> dict[str, Any]:
    identity = execution_identity()
    context = load_context(world, identity)
    geometry = context.geometry.policy_geometry()
    seed = int(registration()["optimization_seed"])
    truth = SyntheticTruth(context.truth, geometry.alpha)
    truth_folder = OUTPUT / "truth_optima" / context.name
    truth_spec = {"world": world, "stage": "truth_optimization", "seed": seed}
    if not completed(truth_folder, truth_spec, identity):
        result = optimization.optimize_fit(truth, geometry, seed=seed, phase_blind=not any(world["parameters"].values()))
        benchmark.write_json(truth_folder / "optimization.json", result)
        finish(truth_folder, truth_spec, identity, ("optimization.json",), {})
    truth_optimum = json.loads((truth_folder / "optimization.json").read_text())
    for family in FAMILIES:
        fit = load_cell(context, family, identity)
        folder = OUTPUT / "contexts" / context.name / "optimization" / family
        spec = {**cell_spec(world, family), "stage": "raw_optimization"}
        if completed(folder, spec, identity):
            continue
        execution_identity()
        result = optimization.optimize_fit(fit, geometry, seed=seed, phase_blind=not any(fit.parameters.values()))
        scores = optimization.evaluate_selected_truth(result, truth, geometry, truth_optimum)
        benchmark.write_json(folder / "optimization.json", result)
        benchmark.write_json(folder / "truth_scores.json", scores)
        sources = (
            truth_folder / "complete.json",
            OUTPUT / "contexts" / context.name / "cells" / family / "complete.json",
        )
        finish(
            folder,
            spec,
            identity,
            ("optimization.json", "truth_scores.json"),
            {str(path): benchmark.sha256(path) for path in sources},
        )
    return {"world": context.name, "raw_cells": len(FAMILIES)}


def summarize_optimization() -> None:
    identity = execution_identity()
    rows, receipts = [], {}
    for world in registration()["worlds"]:
        for family in FAMILIES:
            folder = OUTPUT / "contexts" / world["name"] / "optimization" / family
            spec = {**cell_spec(world, family), "stage": "raw_optimization"}
            if not completed(folder, spec, identity):
                raise ValueError(f"Missing optimization: {folder}")
            result = json.loads((folder / "optimization.json").read_text())
            scores = json.loads((folder / "truth_scores.json").read_text())
            rows.append(
                {
                    "world": world["name"],
                    "family": family,
                    **scores,
                    **{
                        key: result[key]
                        for key in (
                            "predicted_gain_same_aggregate",
                            "predicted_gain_vs_static",
                            "phase_tv",
                            "max_total_epochs",
                            "nearest_policy_sqrt_distance",
                        )
                    },
                    "selected_solver_success": result["best"]["success"],
                    "static_solver_success": result["best_static"]["success"],
                }
            )
            receipts[str(folder / "complete.json")] = benchmark.sha256(folder / "complete.json")
    pd.DataFrame(rows).to_csv(OUTPUT / "optimization_scores.csv", index=False)
    benchmark.write_json(
        OUTPUT / "optimization_manifest.json",
        {
            "identity": identity,
            "cell_receipts": receipts,
            "outputs": {"optimization_scores.csv": benchmark.sha256(OUTPUT / "optimization_scores.csv")},
            "scope": "Raw numerical-incumbent diagnostics; no global guarantee and no retroactive extra stage-two gate.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "prepare", "fit", "summarize", "optimize", "summarize-optimization")
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "prepare":
        prepare()
    elif args.command == "fit":
        prepare()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for wave in WAVES:
                futures = [
                    executor.submit(run_cell, world, family) for world in registration()["worlds"] for family in wave
                ]
                for future in as_completed(futures):
                    print(json.dumps(future.result(), sort_keys=True), flush=True)
        summarize()
    elif args.command == "summarize":
        summarize()
    elif args.command == "optimize":
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(optimize_world, world) for world in registration()["worlds"]]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
        summarize_optimization()
    else:
        summarize_optimization()


if __name__ == "__main__":
    main()
