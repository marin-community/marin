# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Eight noiseless matching-law J2 heads and descriptive raw optima.

Two private TP028 runners retain its complete estimator, scoring and optimizer.
Each owns one mean family and its native aggregate, with disjoint null/positive
contexts. No cross-law, noisy or empirical procedure is implemented.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_stage2_20261015 as physical
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_mean_coordinate_oracle_20261015 as oracle
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mean_coordinate_joint_models_20261015 as models

ROOT = Path(__file__).resolve().parent
OUTPUT = oracle.OUTPUT / "stage2"
AUDIT = oracle.OUTPUT / "audit/oracle_replay_report_v1.json"
STAGE1_SEAL = "85937003912bf8b6213684de4c86722135ebfd043dacf3dffcb41485148b4a33"
STAGE1_AUDIT = "16d1412bda283743ab758292e7d6ce99f1756544ef82f97e882d2d295f8f9b9b"
APPROVAL_SCOPE = "mean_coordinate_stage2_matching_noiseless"
benchmark = physical.benchmark
FAMILIES = oracle.FAMILIES
CANDIDATES = oracle.CANDIDATES
WORLDS = tuple(
    {
        "matching_family": family,
        "name": name,
        "generating_family": "aggregate_mariner" if strength == 0 else family,
        "parameters": {"lambda": strength},
    }
    for family in FAMILIES
    for name, strength in (("null", 0.0), ("positive", 0.5))
)
GATES = dict(physical.GATES)
CONTRACT = {
    **physical.CONTRACT,
    "families": list(models.MODEL_NAMES),
    "matching_families": list(FAMILIES),
    "worlds": list(WORLDS),
    "families_per_context": 2,
    "cross_law_fits": False,
    "parameter_grids": {family: list(models.parameter_grid(family)) for family in models.MODEL_NAMES},
    "learned_heads": 8,
    "truth_optimization_cells": 4,
    "fitted_optimization_cells": 8,
}


def source_paths() -> set[Path]:
    return (
        oracle.source_paths()
        | physical.source_paths()
        | {
            Path(__file__),
            Path(models.__file__),
            ROOT / "test_tpp40_mean_coordinate_joint_models_20261015.py",
            ROOT / "test_tpp40_mean_coordinate_stage2_20261015.py",
            OUTPUT / "registration.json",
            oracle.OUTPUT / "result_manifest.json",
            oracle.OUTPUT / "execution_identity.json",
            oracle.OUTPUT / "oracle_summary.json",
            oracle.OUTPUT / "oracle_summary_receipt.json",
            oracle.OUTPUT / "preflight.json",
            AUDIT,
            physical.OUTPUT / "registration.json",
            physical.OUTPUT / "execution_identity.json",
        }
    )


def verify_adoption() -> None:
    """Verify sealed matching-law recovery and the exact frozen TP028 procedure."""
    oracle.registration()
    if oracle.source_identity() != json.loads((oracle.OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Matching-law oracle scientific identity changed")
    manifest = oracle.OUTPUT / "result_manifest.json"
    if benchmark.sha256(manifest) != STAGE1_SEAL or benchmark.sha256(AUDIT) != STAGE1_AUDIT:
        raise ValueError("Required stage-one seal or independent audit changed")
    sealed = {item["path"]: item["sha256"] for item in json.loads(manifest.read_text())["files"]}
    benchmark.verify_hashes(sealed, oracle.OUTPUT)
    for path in (AUDIT, oracle.OUTPUT / "oracle_summary.json", oracle.OUTPUT / "execution_identity.json"):
        if benchmark.sha256(path) != sealed[str(path.relative_to(oracle.OUTPUT))]:
            raise ValueError("Required oracle evidence is absent from its seal")
    audit = json.loads(AUDIT.read_text())
    summary = json.loads((oracle.OUTPUT / "oracle_summary.json").read_text())
    if not audit["passed"] or not summary["eligible_for_separate_learned_procedure_review"]:
        raise ValueError("Oracle recovery and learned-floor prerequisites must pass")
    physical.registration()
    if physical.source_identity() != json.loads((physical.OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Adopted TP028 learned procedure changed")


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    expected = {
        "candidate_ids": CANDIDATES,
        "stage": 2,
        "status": "registered",
        "execution_contract": CONTRACT,
        "worlds": list(WORLDS),
        "families": list(models.MODEL_NAMES),
        "optimization_seed": CONTRACT["optimization_seed"],
        "environment": benchmark.environment_identity(),
        "stage1_result_manifest_sha256": STAGE1_SEAL,
        "stage1_audit_sha256": STAGE1_AUDIT,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Registered eight-head matching-law stage-two contract differs")
    paths = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(value["frozen_reference_inputs"]) != paths:
        raise ValueError("Registration must pin the complete source/input set")
    benchmark.verify_hashes(value["frozen_reference_inputs"])
    verify_adoption()
    return value


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def freeze() -> None:
    """Freeze the reviewed source without generating labels or fitting heads."""
    registration()
    current = source_identity()
    target = OUTPUT / "execution_identity.json"
    if target.exists() and json.loads(target.read_text()) != current:
        raise ValueError("Preserve the previous identity and register a separate revision")
    benchmark.write_json(target, current)


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != current:
        raise ValueError("Frozen mean-coordinate stage-two sources, inputs or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": APPROVAL_SCOPE,
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != item for key, item in expected.items()):
        raise ValueError("Separate exact mean-coordinate stage-two approval is required")
    return current


def family_registration(family: str) -> dict[str, Any]:
    value = registration()
    return {
        **value,
        "worlds": [world for world in WORLDS if world["matching_family"] == family],
        "families": ["aggregate_mariner", family],
    }


@dataclass(frozen=True)
class SyntheticTruth:
    specification: dict[str, Any]
    alpha: float

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        raw = self.specification["head"]
        matrix, _ = models.design_and_offset(
            early,
            late,
            self.alpha,
            self.specification["shape"],
            self.specification["world"]["matching_family"],
            self.specification["parameters"],
        )
        return raw["floor"] + np.exp(raw["intercept"] + matrix @ np.asarray(raw["coefficients"]))


def qualification_checks(
    world: dict[str, Any], family: str, parameters: dict[str, float], scores: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Apply unchanged TP028 gates to each separately matched temporal family."""
    matching = world["matching_family"]
    if family not in ("aggregate_mariner", matching):
        raise ValueError("Cross-law qualification is outside this protocol")
    old_world = {
        **world,
        "generating_family": physical.FAMILIES[1] if world["name"] == "positive" else "aggregate_mariner",
    }
    old_family = physical.FAMILIES[1] if family == matching else family
    result = physical.qualification_checks(old_world, old_family, parameters, scores, baseline)
    return {**result, "matching_family": matching, "family": family}


def isolated_runner(family: str) -> ModuleType:
    """Bind one matching family, then rebuild the nested context/raw runner."""
    name = f"{__name__}._{family}"
    specification = importlib.util.spec_from_file_location(name, physical.__file__)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    # A narrow facade shares canonical serializable classes/functions, not module state.
    family_models = ModuleType(f"{name}._models")
    family_models.MODEL_NAMES = ("aggregate_mariner", family)
    family_models.JointFit = models.JointFit
    family_models.fit_joint = models.fit_joint
    family_models.design_and_offset = models.design_and_offset
    family_oracle = oracle.ORACLES[family]
    oracle_view = ModuleType(f"{name}._oracle")
    oracle_view.OUTPUT = oracle.OUTPUT
    oracle_view.SHAPE = family_oracle.SHAPE
    oracle_view.inputs = family_oracle.inputs
    oracle_view.world_inputs = family_oracle.world_inputs
    oracle_view.head_record = family_oracle.head_record
    module.OUTPUT = OUTPUT / family
    module.FAMILIES = family_models.MODEL_NAMES
    module.WORLDS = tuple(world for world in WORLDS if world["matching_family"] == family)
    module.oracle = oracle_view
    module.models = family_models
    module.registration = partial(family_registration, family)
    module.execution_identity = execution_identity
    module.SyntheticTruth = SyntheticTruth
    module.qualification_checks = qualification_checks
    module._RUNNER = module.isolated_runner()
    return module


RUNNERS = {family: isolated_runner(family) for family in FAMILIES}


def world_runner(world: dict[str, Any]) -> ModuleType:
    if world not in WORLDS:
        raise ValueError("Unregistered matching-law world")
    return RUNNERS[world["matching_family"]]


def prepare() -> None:
    execution_identity()
    for module in RUNNERS.values():
        module.prepare()


def run_cell(world: dict[str, Any], family: str) -> dict[str, Any]:
    module = world_runner(world)
    if family not in module.FAMILIES:
        raise ValueError("Cross-law fitting is outside this protocol")
    return module.run_cell(world, family)


def optimize_world(world: dict[str, Any]) -> dict[str, Any]:
    return world_runner(world).optimize_world(world)


def summarize_stage(stage: str) -> None:
    """Join freshly verified family summaries without recomputing their metrics."""
    identity = execution_identity()
    if stage not in ("fit", "optimization"):
        raise ValueError(f"Unknown summary stage: {stage}")
    name = "summary.json" if stage == "fit" else "optimization_manifest.json"
    csv_name = "scores.csv" if stage == "fit" else "optimization_scores.csv"
    records, frames, dependencies = [], [], {}
    for family, module in RUNNERS.items():
        if stage == "fit":
            module.summarize()
        else:
            module._RUNNER.summarize_optimization()
        record = json.loads((module.OUTPUT / name).read_text())
        if record["identity"] != identity:
            raise ValueError("Family summary identity differs")
        benchmark.verify_hashes(record["cell_receipts"])
        digest = record["scores_sha256"] if stage == "fit" else record["outputs"][csv_name]
        if benchmark.sha256(module.OUTPUT / csv_name) != digest:
            raise ValueError("Family summary score table differs")
        frame = pd.read_csv(module.OUTPUT / csv_name, keep_default_na=False)
        expected = {(world["name"], fitted) for world in module.WORLDS for fitted in module.FAMILIES}
        if len(frame) != 4 or set(zip(frame["world"], frame["family"], strict=True)) != expected:
            raise ValueError("Family summary is missing or duplicating a registered fit")
        frame.insert(0, "matching_family", family)
        frames.append(frame)
        records.append(record)
        dependencies[str(module.OUTPUT / name)] = benchmark.sha256(module.OUTPUT / name)
        dependencies[str(module.OUTPUT / csv_name)] = digest
    pd.concat(frames, ignore_index=True).to_csv(OUTPUT / csv_name, index=False)
    result: dict[str, Any] = {
        "identity": identity,
        "cells": 8,
        "family_artifacts": dependencies,
        "outputs": {csv_name: benchmark.sha256(OUTPUT / csv_name)},
        "scope": "Matching-law noiseless qualification only; raw optima are descriptive numerical incumbents.",
    }
    if stage == "fit":
        result["passed"] = all(record["passed"] for record in records)
        result["checks"] = [check for record in records for check in record["checks"]]
    benchmark.write_json(OUTPUT / name, result)


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
            for temporal_wave in (False, True):
                futures = [
                    executor.submit(run_cell, world, world["matching_family"] if temporal_wave else "aggregate_mariner")
                    for world in WORLDS
                ]
                for future in as_completed(futures):
                    print(json.dumps(future.result(), sort_keys=True), flush=True)
        summarize_stage("fit")
    elif args.command == "summarize":
        summarize_stage("fit")
    elif args.command == "optimize":
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(optimize_world, world) for world in WORLDS]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
        summarize_stage("optimization")
    else:
        summarize_stage("optimization")


if __name__ == "__main__":
    main()
