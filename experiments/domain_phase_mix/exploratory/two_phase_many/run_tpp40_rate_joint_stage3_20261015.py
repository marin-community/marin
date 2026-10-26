# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Paired noisy qualification using an isolated copy of the frozen stage-two runner.

The native J2 estimator, control union, cell fitter, scorer and optimizer remain
unchanged. This module supplies six noisy contexts, source receipts and the
previously registered stage-three acceptance rules. Scientific execution needs
an exact separate approval after independent stage-two replay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_rate_joint_stage2_20261015 as stage2
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_reopening_20261015 as stage1

ROOT = Path(__file__).resolve().parent
OUTPUT = stage1.OUTPUT / "stage3"
SEEDS = (20261015, 20261016, 20261017)
NOISE_SD = 0.003
FAMILIES = stage2.FAMILIES


def context_specs() -> list[dict[str, Any]]:
    return [
        {
            **world,
            "truth_world": world["name"],
            "name": f"{world['name']}_noise003_seed{seed}",
            "noise_seed": seed,
            "noise_sd": NOISE_SD,
        }
        for world in stage1.registration()["worlds"]
        if world["name"] in ("null", "combined")
        for seed in SEEDS
    ]


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    parent = stage1.registration()
    expected = {
        "candidate_id": "TP40-026",
        "stage": 3,
        "status": "registered",
        "contexts": context_specs(),
        "families": list(FAMILIES),
        "noise_sd": NOISE_SD,
        "noise_seeds": list(SEEDS),
        "stage3_gates": parent["whole_estimator_plan"]["stage3_gates"],
        "kappa_grid": parent["kappa_grid"],
        "nu_grid": parent["nu_grid"],
        "gamma_grid": list(stage2.joint.GAMMA_GRID),
        "optimization_seed": 20261015,
        "environment": benchmark.environment_identity(),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Stage-three registration differs from the paired noisy protocol")
    benchmark.verify_hashes(value["implementation_hashes"])
    return value


def source_identity() -> dict[str, Any]:
    paths = set(stage2.source_identity()["sources"])
    paths.update(
        str(path.resolve())
        for path in (
            Path(__file__),
            OUTPUT / "registration.json",
            stage2.OUTPUT / "execution_identity.json",
            stage2.OUTPUT / "summary.json",
            stage2.OUTPUT / "result_manifest.json",
        )
    )
    return {
        "sources": {path: benchmark.sha256(Path(path)) for path in sorted(paths)},
        "environment": benchmark.environment_identity(),
    }


def freeze() -> None:
    value = registration()
    stage2_identity = json.loads((stage2.OUTPUT / "execution_identity.json").read_text())
    if stage2.source_identity() != stage2_identity:
        raise ValueError("Frozen stage-two source or input changed")
    seal_path = stage2.OUTPUT / "result_manifest.json"
    if benchmark.sha256(seal_path) != value["stage2_result_manifest_sha256"]:
        raise ValueError("Independent stage-two result seal differs from registration")
    seal = json.loads(seal_path.read_text())
    benchmark.verify_hashes({name: item["sha256"] for name, item in seal["files"].items()}, stage2.OUTPUT)
    summary = json.loads((stage2.OUTPUT / "summary.json").read_text())
    if summary["identity"] != stage2_identity or summary["cells"] != 16 or not summary["passed"]:
        raise ValueError("All sixteen stage-two heads must pass before stage three")
    audit_path = Path(value["stage2_audit_report"])
    if benchmark.sha256(audit_path) != value["stage2_audit_report_sha256"]:
        raise ValueError("Stage-two independent replay report changed")
    audit = json.loads(audit_path.read_text())
    if audit[value["stage2_audit_pass_key"]] is not True:
        raise ValueError("Independent stage-two replay did not pass")
    identity = source_identity()
    identity["sources"][str(audit_path)] = benchmark.sha256(audit_path)
    destination = OUTPUT / "execution_identity.json"
    if destination.exists() and json.loads(destination.read_text()) != identity:
        raise ValueError("Preserve the previous stage-three identity and register a new revision")
    benchmark.write_json(destination, identity)


def execution_identity() -> dict[str, Any]:
    value = registration()
    actual = source_identity()
    actual["sources"][value["stage2_audit_report"]] = benchmark.sha256(Path(value["stage2_audit_report"]))
    expected = json.loads((OUTPUT / "execution_identity.json").read_text())
    if expected != actual:
        raise ValueError("Stage-three source, data or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    required = {
        "scope": "rate_reopening_stage3_paired_noise_and_raw_qualification",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != item for key, item in required.items()):
        raise ValueError("A separate exact stage-three approval is required")
    return expected


def noisy_training_labels(clean: np.ndarray, train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw the paired full-geometry noise vector and apply it only to training rows."""
    noise = np.random.default_rng(seed).normal(0, NOISE_SD, len(clean))
    observed = clean.copy()
    observed[train] += noise[train]
    return observed, noise


def prepare() -> None:
    identity = execution_identity()
    arrays, folds, truth, head = stage1.inputs()
    for spec in registration()["contexts"]:
        folder = OUTPUT / "contexts" / spec["name"]
        if _BACKEND.completed(folder, spec, identity):
            continue
        values = stage1.world_inputs(arrays, folds, truth, head, spec["parameters"])
        observed, noise = noisy_training_labels(values["clean"], values["train"], spec["noise_seed"])
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "synthetic.npz",
            train=values["train"],
            asymmetric=values["held"],
            tied=values["tied"],
            inner_labels=values["labels"],
            clean=values["clean"],
            observed=observed,
            clean_tied=values["clean_tied"],
            full_noise_vector=noise,
        )
        benchmark.write_json(
            folder / "truth.json",
            {"shape": truth["shape"], "head": truth["head"], "parameters": spec["parameters"], "world": spec},
        )
        _BACKEND.finish(folder, spec, identity, ("synthetic.npz", "truth.json"), {})


def isolated_backend() -> ModuleType:
    """Rebind stage-specific I/O locally without modifying the imported frozen runner."""
    name = f"{__name__}._stage2"
    specification = importlib.util.spec_from_file_location(name, stage2.__file__)
    assert specification is not None and specification.loader is not None
    backend = importlib.util.module_from_spec(specification)
    sys.modules[name] = backend
    specification.loader.exec_module(backend)
    backend.OUTPUT = OUTPUT
    backend.registration = registration
    backend.source_identity = source_identity
    backend.execution_identity = execution_identity
    return backend


_BACKEND = isolated_backend()


def run_cell(spec: dict[str, Any], family: str) -> dict[str, Any]:
    return _BACKEND.run_cell(spec, family)


def prepare_truth_optimum(spec: dict[str, Any]) -> None:
    identity = execution_identity()
    context = _BACKEND.load_context(spec, identity)
    geometry = context.geometry.policy_geometry()
    seed = registration()["optimization_seed"]
    truth = stage2.SyntheticTruth(context.truth, geometry.alpha)
    folder = OUTPUT / "truth_optima" / spec["truth_world"]
    truth_spec = {"truth_world": spec["truth_world"], "parameters": spec["parameters"], "seed": seed}
    if not _BACKEND.completed(folder, truth_spec, identity):
        result = stage2.optimization.optimize_fit(
            truth, geometry, seed=seed, phase_blind=not any(spec["parameters"].values())
        )
        benchmark.write_json(folder / "optimization.json", result)
        _BACKEND.finish(folder, truth_spec, identity, ("optimization.json",), {})


def optimize_context(spec: dict[str, Any]) -> dict[str, Any]:
    identity = execution_identity()
    context = _BACKEND.load_context(spec, identity)
    geometry = context.geometry.policy_geometry()
    seed = registration()["optimization_seed"]
    truth = stage2.SyntheticTruth(context.truth, geometry.alpha)
    folder = OUTPUT / "truth_optima" / spec["truth_world"]
    truth_spec = {"truth_world": spec["truth_world"], "parameters": spec["parameters"], "seed": seed}
    if not _BACKEND.completed(folder, truth_spec, identity):
        raise ValueError("Prepare the unique clean-truth optima before parallel raw-policy diagnostics")
    truth_optimum = json.loads((folder / "optimization.json").read_text())
    for family in FAMILIES:
        fit = _BACKEND.load_cell(context, family, identity)
        destination = OUTPUT / "contexts" / context.name / "optimization" / family
        cell = {**_BACKEND.cell_spec(spec, family), "stage": "raw_optimization"}
        if _BACKEND.completed(destination, cell, identity):
            continue
        execution_identity()
        result = stage2.optimization.optimize_fit(fit, geometry, seed=seed, phase_blind=not any(fit.parameters.values()))
        scores = stage2.optimization.evaluate_selected_truth(result, truth, geometry, truth_optimum)
        benchmark.write_json(destination / "optimization.json", result)
        benchmark.write_json(destination / "truth_scores.json", scores)
        dependencies = (
            folder / "complete.json",
            OUTPUT / "contexts" / context.name / "cells" / family / "complete.json",
        )
        _BACKEND.finish(
            destination,
            cell,
            identity,
            ("optimization.json", "truth_scores.json"),
            {str(path): benchmark.sha256(path) for path in dependencies},
        )
    return {"context": context.name, "raw_cells": len(FAMILIES)}


def summarize() -> None:
    """Replay saved predictions; leave qualification pending until every raw result exists."""
    identity = execution_identity()
    rows, receipts = [], {}
    for spec in registration()["contexts"]:
        context = _BACKEND.load_context(spec, identity)
        query0, query1 = stage2.scoring.query_coordinates(context)
        for family in FAMILIES:
            fit = _BACKEND.load_cell(context, family, identity)
            folder = OUTPUT / "contexts" / context.name / "cells" / family
            scores = json.loads((folder / "scores.json").read_text())
            replay, _ = stage2.scoring.prediction_scores(context, fit.predict(query0, query1))
            if any(scores[key] != value for key, value in replay.items()):
                raise ValueError("A saved prediction score does not replay")
            rows.append({**scores, "truth_world": spec["truth_world"], "noise_seed": spec["noise_seed"]})
            receipts[str(folder / "complete.json")] = benchmark.sha256(folder / "complete.json")
    pd.json_normalize(rows).to_csv(OUTPUT / "scores.csv", index=False)
    benchmark.write_json(
        OUTPUT / "fit_summary.json",
        {
            "identity": identity,
            "cells": len(rows),
            "cell_receipts": receipts,
            "outputs": {"scores.csv": benchmark.sha256(OUTPUT / "scores.csv")},
            "qualification": "Pending raw-policy diagnostics and all registered stage-three gates",
        },
    )


def qualify() -> None:
    identity = execution_identity()
    rows, checks, receipts = [], [], {}
    gates = registration()["stage3_gates"]
    for spec in registration()["contexts"]:
        context = _BACKEND.load_context(spec, identity)
        scores_by_family, raw_by_family, models_by_family = {}, {}, {}
        for family in FAMILIES:
            fit = _BACKEND.load_cell(context, family, identity)
            models_by_family[family] = fit
            cell_folder = OUTPUT / "contexts" / context.name / "cells" / family
            scores_by_family[family] = json.loads((cell_folder / "scores.json").read_text())
            folder = OUTPUT / "contexts" / context.name / "optimization" / family
            cell = {**_BACKEND.cell_spec(spec, family), "stage": "raw_optimization"}
            if not _BACKEND.completed(folder, cell, identity):
                raise ValueError(f"Raw qualification result is incomplete: {folder}")
            raw = json.loads((folder / "truth_scores.json").read_text())
            result = json.loads((folder / "optimization.json").read_text())
            raw_by_family[family] = {**raw, "predicted_gain_same_aggregate": result["predicted_gain_same_aggregate"]}
            rows.append(
                {
                    "context": context.name,
                    "truth_world": spec["truth_world"],
                    "noise_seed": spec["noise_seed"],
                    "family": family,
                    **raw,
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
            receipts[str(cell_folder / "complete.json")] = benchmark.sha256(cell_folder / "complete.json")
        checks.extend(qualification_checks(spec, scores_by_family, raw_by_family, models_by_family, gates))
    pd.DataFrame(rows).to_csv(OUTPUT / "optimization_scores.csv", index=False)
    benchmark.write_json(
        OUTPUT / "qualification.json",
        {
            "identity": identity,
            "contexts": 6,
            "cells": len(rows),
            "passed": all(item["passed"] for item in checks),
            "checks": checks,
            "cell_receipts": receipts,
            "outputs": {"optimization_scores.csv": benchmark.sha256(OUTPUT / "optimization_scores.csv")},
            "scope": (
                "Paired-noise synthetic qualification only; "
                "no empirical fit, physical-mechanism claim or promotion follows."
            ),
        },
    )


def qualification_checks(
    spec: dict[str, Any], scores: dict[str, Any], raw: dict[str, Any], fits: dict[str, Any], gates: dict[str, Any]
) -> list[dict[str, Any]]:
    checks = []
    correct = FAMILIES if spec["truth_world"] == "null" else FAMILIES[3:]
    for family in correct:
        checks.append(
            {
                "context": spec["name"],
                "family": family,
                "gate": "kinetic_boundary",
                "passed": fits[family].parameters["kappa"] != 128 and fits[family].parameters["nu"] != 2,
            }
        )
        bound = scores[FAMILIES[0]]["clean_actual_tied_rmse"] + 0.003
        checks.append(
            {
                "context": spec["name"],
                "family": family,
                "gate": "static_calibration",
                "value": scores[family]["clean_actual_tied_rmse"],
                "limit": bound,
                "passed": scores[family]["clean_actual_tied_rmse"] <= bound,
            }
        )
        if spec["truth_world"] == "null":
            maximum = max(scores[family]["max_positive_predicted_gain"], raw[family]["predicted_gain_same_aggregate"])
            checks.append(
                {
                    "context": spec["name"],
                    "family": family,
                    "gate": "false_positive_gain",
                    "value": maximum,
                    "limit": gates["null_max_positive_raw_or_menu_sameaggregate_gain_bpb"],
                    "passed": maximum <= gates["null_max_positive_raw_or_menu_sameaggregate_gain_bpb"],
                }
            )
        error = abs(raw[family]["phase_gain_overprediction"])
        checks.append(
            {
                "context": spec["name"],
                "family": family,
                "gate": "raw_gain_error",
                "value": error,
                "limit": gates["raw_gain_absolute_error_max_bpb"],
                "passed": error <= gates["raw_gain_absolute_error_max_bpb"],
            }
        )
    if spec["truth_world"] == "combined":
        combined = scores[FAMILIES[3]]["clean_gain_rmse"]
        for control, value in [("zero_gain", scores[FAMILIES[3]]["clean_gain_rms"])] + [
            (family, scores[family]["clean_gain_rmse"]) for family in FAMILIES[1:3]
        ]:
            checks.append(
                {
                    "context": spec["name"],
                    "family": FAMILIES[3],
                    "gate": f"positive_gain_vs_{control}",
                    "value": combined,
                    "limit": value,
                    "passed": combined < value,
                }
            )
    control = min(raw[family]["regret_vs_truth_numerical_incumbent"] for family in FAMILIES[:3])
    regret = raw[FAMILIES[3]]["regret_vs_truth_numerical_incumbent"]
    checks.append(
        {
            "context": spec["name"],
            "family": FAMILIES[3],
            "gate": "raw_regret_vs_best_control",
            "value": regret,
            "limit": control + gates["raw_signed_regret_vs_best_control_extra_max_bpb"],
            "passed": regret <= control + gates["raw_signed_regret_vs_best_control_extra_max_bpb"],
        }
    )
    for check in checks:
        check["applicability"] = (
            "Combined versus best of three controls in every context"
            if check["gate"] == "raw_regret_vs_best_control"
            else (
                "All four families on null truth"
                if spec["truth_world"] == "null"
                else "Correctly specified combined family on positive truth; misspecified controls are diagnostics"
            )
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "fit", "summarize", "optimize", "qualify"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "prepare":
        prepare()
    elif args.command == "fit":
        prepare()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for wave in stage2.WAVES:
                futures = [
                    executor.submit(run_cell, spec, family) for spec in registration()["contexts"] for family in wave
                ]
                for future in as_completed(futures):
                    print(json.dumps(future.result(), sort_keys=True), flush=True)
        summarize()
    elif args.command == "summarize":
        summarize()
    elif args.command == "qualify":
        qualify()
    else:
        # Compute each clean-truth world once before parallel seed optimizations.
        for spec in registration()["contexts"]:
            if spec["noise_seed"] == SEEDS[0]:
                prepare_truth_optimum(spec)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(optimize_context, spec) for spec in registration()["contexts"]]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
        qualify()


if __name__ == "__main__":
    main()
