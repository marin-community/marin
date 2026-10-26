# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Four noiseless late-refresh J2 fits and descriptive numerical raw optima.

The sealed oracle is a prerequisite. Every scientific command requires a
separate approval for this source identity; noisy and empirical stages are not
implemented. Receipt/context/optimizer flow is reused from frozen TP40-026.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_oracle_20261015 as oracle
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_rate_joint_stage2_20261015 as reused
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_tangent_round5_20261015 as scoring
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as joint
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_late_refresh_joint_models_20261015 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_optimization_20261015 as optimization

ROOT = Path(__file__).resolve().parent
OUTPUT = oracle.OUTPUT / "stage2"
AUDIT = oracle.OUTPUT / "audit/oracle_replay_report_v1.json"
STAGE1_SEAL = "eda1b26c39ed08a8e1615d09f7cab482380329521ee6378ac477d1fe640d726e"
STAGE1_AUDIT = "3417eef5648c98cf031bf0e8f565f7579533b979c4bfdbb945ad4de6f3f4ce2c"
REUSED_SEAL = "8a4e40d188dcfd85c1be85bc1fce2ded78417f76e81c1ea93407063b0d201dff"
FAMILIES = models.MODEL_NAMES
WORLDS: tuple[dict[str, Any], ...] = (
    {"name": "null", "generating_family": "aggregate_mariner", "parameters": {"lambda": 0.0}},
    {"name": "positive", "generating_family": "late_refresh_hazard", "parameters": {"lambda": 0.5}},
)
GATES = {
    "correct_family_clean_gain_rmse_max_bpb": 0.001,
    "null_max_abs_menu_sameaggregate_gain_bpb": 1e-8,
    "correct_family_actual_tied_rmse_extra_vs_aggregate_bpb": 0.003,
    "temporal_upper_boundary": 1.0,
}
CONTRACT = {
    "families": list(FAMILIES),
    "worlds": list(WORLDS),
    "parameter_grids": {family: list(models.parameter_grid(family)) for family in FAMILIES},
    "gamma_grid": list(joint.GAMMA_GRID),
    "noise_sd": 0.0,
    "optimization_seed": 20261015,
    "stage2_gates": GATES,
    "learned_heads": 4,
    "truth_optimization_cells": 2,
    "fitted_optimization_cells": 4,
    "raw_optimization_is_acceptance_gate": False,
}


def source_paths() -> set[Path]:
    return oracle.source_paths() | {
        Path(__file__),
        Path(models.__file__),
        Path(models.control_union.__file__),
        Path(reused.__file__),
        Path(scoring.__file__),
        Path(optimization.__file__),
        ROOT / "test_tpp40_late_refresh_joint_models_20261015.py",
        ROOT / "test_tpp40_late_refresh_stage2_20261015.py",
        OUTPUT / "registration.json",
        oracle.OUTPUT / "result_manifest.json",
        oracle.OUTPUT / "execution_identity.json",
        oracle.OUTPUT / "oracle_summary.json",
        oracle.OUTPUT / "oracle_summary_receipt.json",
        oracle.OUTPUT / "preflight.json",
        AUDIT,
        scoring.OUTPUT / "registration.json",
        scoring.OUTPUT / "execution_identity.json",
        reused.OUTPUT / "registration.json",
        reused.OUTPUT / "result_manifest.json",
    }


def registration() -> dict[str, Any]:
    config = json.loads((OUTPUT / "registration.json").read_text())
    if (config["candidate_id"], config["stage"], config["status"]) != ("TP40-028", 2, "registered"):
        raise ValueError("A separately registered stage-two protocol is required")
    if config["execution_contract"] != CONTRACT or config["environment"] != benchmark.environment_identity():
        raise ValueError("Registered four-fit contract or numerical runtime differs")
    if any(config[key] != CONTRACT[key] for key in ("worlds", "families", "optimization_seed")):
        raise ValueError("Reused runner bindings differ from the registered execution contract")
    if config["stage1_result_manifest_sha256"] != STAGE1_SEAL or config["stage1_audit_sha256"] != STAGE1_AUDIT:
        raise ValueError("Registered oracle seal or independent audit differs")
    paths = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(config["frozen_reference_inputs"]) != paths:
        raise ValueError("Registration must pin the complete executable source/input set")
    benchmark.verify_hashes(config["frozen_reference_inputs"])
    return config


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def freeze() -> None:
    """Adopt the audited oracle and frozen evaluation machinery without labels."""
    registration()
    oracle.registration()
    if oracle.source_identity() != json.loads((oracle.OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Oracle scientific sources changed after execution")
    if (
        benchmark.sha256(oracle.OUTPUT / "result_manifest.json") != STAGE1_SEAL
        or benchmark.sha256(AUDIT) != STAGE1_AUDIT
    ):
        raise ValueError("Required oracle seal or independent audit changed")
    seal = json.loads((oracle.OUTPUT / "result_manifest.json").read_text())
    sealed_files = {item["path"]: item["sha256"] for item in seal["files"]}
    benchmark.verify_hashes(sealed_files, oracle.OUTPUT)
    for path in (AUDIT, oracle.OUTPUT / "oracle_summary.json", oracle.OUTPUT / "execution_identity.json"):
        if benchmark.sha256(path) != sealed_files[str(path.relative_to(oracle.OUTPUT))]:
            raise ValueError("Required stage-one evidence is absent from its seal")
    report = json.loads(AUDIT.read_text())
    summary = json.loads((oracle.OUTPUT / "oracle_summary.json").read_text())
    if not report["passed"] or not summary["eligible_for_separate_learned_procedure_review"]:
        raise ValueError("Independent oracle recovery and learned-floor prerequisites must pass")
    if benchmark.sha256(reused.OUTPUT / "result_manifest.json") != REUSED_SEAL:
        raise ValueError("Frozen control-union runner adoption seal changed")
    previous_seal = json.loads((reused.OUTPUT / "result_manifest.json").read_text())
    previous_registration = reused.OUTPUT / "registration.json"
    if benchmark.sha256(previous_registration) != previous_seal["files"]["registration.json"]["sha256"]:
        raise ValueError("Frozen runner registration differs from its adoption seal")
    previous_pins = json.loads(previous_registration.read_text())["implementation_hashes"]
    for module in (reused, models.control_union):
        path = Path(module.__file__).resolve()
        if benchmark.sha256(path) != previous_pins[str(path)]:
            raise ValueError(f"Frozen control-union implementation changed: {path}")
    round5 = json.loads((scoring.OUTPUT / "execution_identity.json").read_text())
    for module in (scoring, optimization):
        path = Path(module.__file__).resolve()
        if benchmark.sha256(path) != round5[str(path)]:
            raise ValueError(f"Adopted round-five evaluator changed: {path}")
    if scoring.registration()["metrics"]["gain_sign_zero_tolerance"] != 1e-10:
        raise ValueError("Adopted gain-scoring tolerance changed")
    current = source_identity()
    destination = OUTPUT / "execution_identity.json"
    if destination.exists() and json.loads(destination.read_text()) != current:
        raise ValueError("Preserve the earlier stage-two identity and register a separate revision")
    benchmark.write_json(destination, current)


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != current:
        raise ValueError("Frozen stage-two sources, inputs or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    required = {
        "scope": "late_refresh_stage2_noiseless",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in required.items()):
        raise ValueError("Separate exact stage-two execution approval is required")
    return current


@dataclass(frozen=True)
class SyntheticTruth:
    specification: dict[str, Any]
    alpha: float

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        raw = self.specification["head"]
        matrix, _ = models.design_and_offset(
            early, late, self.alpha, self.specification["shape"], "late_refresh_hazard", self.specification["parameters"]
        )
        return raw["floor"] + np.exp(raw["intercept"] + matrix @ np.asarray(raw["coefficients"]))


def isolated_runner() -> ModuleType:
    """Reuse receipt/context/raw-optimizer flow with private family/output bindings."""
    name = f"{__name__}._frozen_runner"
    spec = importlib.util.spec_from_file_location(name, reused.__file__)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[name] = runner
    spec.loader.exec_module(runner)
    runner.OUTPUT = OUTPUT
    runner.FAMILIES = FAMILIES
    runner.models = models
    runner.registration = registration
    runner.execution_identity = execution_identity
    runner.SyntheticTruth = SyntheticTruth
    return runner


_RUNNER = isolated_runner()


def prepare() -> None:
    identity = execution_identity()
    arrays, folds, head = oracle.inputs()
    for world in WORLDS:
        folder = OUTPUT / "contexts" / world["name"]
        if _RUNNER.completed(folder, world, identity):
            continue
        values = oracle.world_inputs(arrays, folds, head, world["parameters"]["lambda"])
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
            {"shape": oracle.SHAPE, "head": oracle.head_record(head), "parameters": world["parameters"], "world": world},
        )
        dependencies = {
            str(path): benchmark.sha256(path)
            for path in (oracle.OUTPUT / "oracle_summary_receipt.json", oracle.OUTPUT / "result_manifest.json")
        }
        _RUNNER.finish(folder, world, identity, ("synthetic.npz", "truth.json"), dependencies)


def run_cell(world: dict[str, Any], family: str) -> dict[str, Any]:
    identity = execution_identity()
    context = _RUNNER.load_context(world, identity)
    folder = OUTPUT / "contexts" / context.name / "cells" / family
    spec = _RUNNER.cell_spec(world, family)
    if _RUNNER.completed(folder, spec, identity):
        return {**spec, "cached": True}
    controls_names = () if family == "aggregate_mariner" else ("aggregate_mariner",)
    controls = tuple(_RUNNER.load_cell(context, name, identity) for name in controls_names)
    paths = [folder.parent.parent / "complete.json"] + [
        folder.parent / name / "complete.json" for name in controls_names
    ]
    dependencies = {str(path): benchmark.sha256(path) for path in paths}
    geometry, train = context.geometry, context.train
    print(f"Fitting {context.name} {family}", flush=True)
    started = time.monotonic()
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
    if abs(float(np.sqrt(np.mean((inner - context.observed[train]) ** 2))) - fit.inner_rmse) > 1e-12:
        raise ValueError("Selected inner score does not replay")
    if controls and fit.inner_rmse > controls[0].inner_rmse + 1e-12:
        raise ValueError("Temporal search lost its complete native aggregate")
    folder.mkdir(parents=True, exist_ok=True)
    benchmark.write_json(folder / "model.json", fit.to_json())
    restored = models.JointFit.from_json(json.loads((folder / "model.json").read_text()))
    if not np.array_equal(restored.predict(query0, query1), predictions):
        raise ValueError("Serialized prediction differs")
    frame.to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(
        {"row_index": train, "inner_fold": context.inner_labels, "observed": context.observed[train], "predicted": inner}
    ).to_csv(folder / "inner_predictions.csv", index=False)
    pd.DataFrame(fit.selection_profile).sort_values(
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
                "Best already-searched trial per lambda; all trials/failures remain in model.json. "
                "No per-lambda extra gamma optimization."
            ),
        },
    )
    _RUNNER.finish(
        folder,
        spec,
        identity,
        ("model.json", "predictions.csv", "inner_predictions.csv", "kinetic_profiles.csv", "scores.json"),
        dependencies,
    )
    return {**spec, "inner_rmse": fit.inner_rmse}


def qualification_checks(
    world: dict[str, Any], family: str, parameters: dict[str, float], scores: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    correct = world["name"] == "null" or family == world["generating_family"]
    boundary = family == "late_refresh_hazard" and parameters["lambda"] == 1.0
    return {
        "world": world["name"],
        "family": family,
        "correct_family": correct,
        "null_gain_pass": (
            world["name"] != "null"
            or scores["max_abs_predicted_gain"] <= GATES["null_max_abs_menu_sameaggregate_gain_bpb"]
        ),
        "gain_recovery_pass": (
            not correct or scores["clean_gain_rmse"] <= GATES["correct_family_clean_gain_rmse_max_bpb"]
        ),
        "tied_calibration_pass": (
            not correct
            or scores["clean_actual_tied_rmse"]
            <= baseline["clean_actual_tied_rmse"] + GATES["correct_family_actual_tied_rmse_extra_vs_aggregate_bpb"]
        ),
        "selected_physical_upper_boundary": boundary,
        "kinetic_boundary_pass": not correct or not boundary,
    }


def summarize() -> None:
    identity = execution_identity()
    rows, checks, receipts = [], [], {}
    for world in WORLDS:
        context = _RUNNER.load_context(world, identity)
        baseline = json.loads((OUTPUT / "contexts" / context.name / "cells/aggregate_mariner/scores.json").read_text())
        for family in FAMILIES:
            fit = _RUNNER.load_cell(context, family, identity)
            folder = OUTPUT / "contexts" / context.name / "cells" / family
            scores = json.loads((folder / "scores.json").read_text())
            query0, query1 = scoring.query_coordinates(context)
            replay, _ = scoring.prediction_scores(context, fit.predict(query0, query1))
            if any(scores[name] != value for name, value in replay.items()):
                raise ValueError("Saved prediction scores changed")
            rows.append(scores)
            checks.append(qualification_checks(world, family, fit.parameters, scores, baseline))
            receipts[str(folder / "complete.json")] = benchmark.sha256(folder / "complete.json")
    pd.json_normalize(rows).to_csv(OUTPUT / "scores.csv", index=False)
    keys = ("null_gain_pass", "gain_recovery_pass", "tied_calibration_pass", "kinetic_boundary_pass")
    benchmark.write_json(
        OUTPUT / "summary.json",
        {
            "identity": identity,
            "cells": len(rows),
            "passed": all(all(row[key] for key in keys) for row in checks),
            "checks": checks,
            "cell_receipts": receipts,
            "scores_sha256": benchmark.sha256(OUTPUT / "scores.csv"),
            "scope": (
                "Noiseless finite-menu recovery gate only. Raw diagnostics and independent audit are required "
                "before acceptance; no noisy/empirical execution or optimum qualification follows."
            ),
        },
    )


def optimize_world(world: dict[str, Any]) -> dict[str, Any]:
    return _RUNNER.optimize_world(world)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "prepare", "fit", "summarize", "optimize", "summarize-optimization")
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "prepare":
        prepare()
    elif args.command == "fit":
        prepare()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for family in FAMILIES:
                futures = [executor.submit(run_cell, world, family) for world in WORLDS]
                for future in as_completed(futures):
                    print(json.dumps(future.result(), sort_keys=True), flush=True)
        summarize()
    elif args.command == "summarize":
        summarize()
    elif args.command == "optimize":
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(optimize_world, world) for world in WORLDS]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
        _RUNNER.summarize_optimization()
    else:
        _RUNNER.summarize_optimization()


if __name__ == "__main__":
    main()
