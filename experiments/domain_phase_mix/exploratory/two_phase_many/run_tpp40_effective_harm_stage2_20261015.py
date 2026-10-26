# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Four noiseless effective-harm J2 heads using the frozen TP028 procedure.

Only law/family, artifact and provenance bindings change. Seven strengths,
including the lambda-one all-early sink, remain visible. Raw optimization is
descriptive. No noisy or empirical procedure is implemented or authorized.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_effective_harm_oracle_20261015 as oracle
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_stage2_20261015 as physical
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_effective_harm_joint_models_20261015 as models

ROOT = Path(__file__).resolve().parent
OUTPUT = oracle.OUTPUT / "stage2"
AUDIT = oracle.OUTPUT / "audit/oracle_replay_report_v1.json"
STAGE1_SEAL = "5837df387f27736f1f7dab6f03c4028848614084a0c8096bd88237889dab4a03"
STAGE1_AUDIT = "4c07fcbe1d9af34e983b89eddf85ad14add1c53da2b51a8e0f81e796fda18993"
benchmark = physical.benchmark
FAMILIES = models.MODEL_NAMES
WORLDS: tuple[dict[str, Any], ...] = (
    {"name": "null", "generating_family": "aggregate_mariner", "parameters": {"lambda": 0.0}},
    {"name": "positive", "generating_family": "late_refresh_effective_harm", "parameters": {"lambda": 0.5}},
)
GATES = dict(physical.GATES)
CONTRACT = {
    **physical.CONTRACT,
    "families": list(FAMILIES),
    "worlds": list(WORLDS),
    "parameter_grids": {family: list(models.parameter_grid(family)) for family in FAMILIES},
}


def source_paths() -> set[Path]:
    return (
        oracle.source_paths()
        | physical.source_paths()
        | {
            Path(__file__),
            Path(models.__file__),
            ROOT / "test_tpp40_effective_harm_joint_models_20261015.py",
            ROOT / "test_tpp40_effective_harm_stage2_20261015.py",
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


def verify_procedure_adoption() -> None:
    """Require the exact frozen physical-harm estimator/evaluator bindings."""
    physical.registration()
    old = json.loads((physical.OUTPUT / "execution_identity.json").read_text())
    if physical.source_identity() != old:
        raise ValueError("Adopted TP028 learned procedure changed")
    pins = json.loads((physical.OUTPUT / "registration.json").read_text())["frozen_reference_inputs"]
    for module in (physical, physical.models):
        path = Path(module.__file__).resolve()
        if benchmark.sha256(path) != pins[str(path)]:
            raise ValueError("Adopted TP028 runner or model source changed")


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    expected = {
        "candidate_id": "TP40-029",
        "stage": 2,
        "status": "registered",
        "execution_contract": CONTRACT,
        "worlds": list(WORLDS),
        "families": list(FAMILIES),
        "optimization_seed": CONTRACT["optimization_seed"],
        "environment": benchmark.environment_identity(),
        "stage1_result_manifest_sha256": STAGE1_SEAL,
        "stage1_audit_sha256": STAGE1_AUDIT,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Registered effective-harm stage-two contract differs")
    paths = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(value["frozen_reference_inputs"]) != paths:
        raise ValueError("Registration must pin complete source/input set")
    benchmark.verify_hashes(value["frozen_reference_inputs"])
    verify_procedure_adoption()
    return value


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != current:
        raise ValueError("Frozen effective-harm stage-two sources, inputs or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": "effective_harm_stage2_noiseless",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != item for key, item in expected.items()):
        raise ValueError("Separate exact effective-harm stage-two approval is required")
    return current


@dataclass(frozen=True)
class SyntheticTruth:
    specification: dict[str, Any]
    alpha: float

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        raw = self.specification["head"]
        matrix, _ = models.design_and_offset(
            early, late, self.alpha, self.specification["shape"], FAMILIES[1], self.specification["parameters"]
        )
        return raw["floor"] + np.exp(raw["intercept"] + matrix @ np.asarray(raw["coefficients"]))


def qualification_checks(
    world: dict[str, Any], family: str, parameters: dict[str, float], scores: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen gates, translating only the temporal family identifier."""
    old_world = {**world, "generating_family": physical.FAMILIES[1] if world["name"] == "positive" else FAMILIES[0]}
    old_family = physical.FAMILIES[1] if family == FAMILIES[1] else family
    result = physical.qualification_checks(old_world, old_family, parameters, scores, baseline)
    return {**result, "family": family}


def isolated_runner() -> ModuleType:
    """Rebind the unchanged four-head procedure and its nested raw runner privately."""
    name = f"{__name__}._physical_runner"
    specification = importlib.util.spec_from_file_location(name, physical.__file__)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.OUTPUT = OUTPUT
    module.AUDIT = AUDIT
    module.STAGE1_SEAL = STAGE1_SEAL
    module.STAGE1_AUDIT = STAGE1_AUDIT
    module.FAMILIES = FAMILIES
    module.WORLDS = WORLDS
    module.GATES = GATES
    module.CONTRACT = CONTRACT
    module.oracle = oracle._ORACLE
    module.models = models
    module.registration = registration
    module.source_identity = source_identity
    module.execution_identity = execution_identity
    module.SyntheticTruth = SyntheticTruth
    module.qualification_checks = qualification_checks
    module._RUNNER = module.isolated_runner()
    return module


_RUNNER = isolated_runner()


def freeze() -> None:
    _RUNNER.freeze()


def prepare() -> None:
    _RUNNER.prepare()


def run_cell(world: dict[str, Any], family: str) -> dict[str, Any]:
    return _RUNNER.run_cell(world, family)


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
        _RUNNER.summarize()
    elif args.command == "summarize":
        _RUNNER.summarize()
    elif args.command == "optimize":
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(optimize_world, world) for world in WORLDS]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
        _RUNNER._RUNNER.summarize_optimization()
    else:
        _RUNNER._RUNNER.summarize_optimization()


if __name__ == "__main__":
    main()
