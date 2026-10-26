# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Matched paired-noise effective-harm qualification, pending separate approval.

Privately adopts the frozen TP028 stage-three preparation, scoring, gates and
optimizer flow. The sealed TP029 learned estimator and truth replace only the
readout/family bindings. No empirical or cross-law procedure is implemented.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_effective_harm_stage2_20261015 as stage2
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_stage3_20261015 as physical

ROOT = Path(__file__).resolve().parent
OUTPUT = stage2.OUTPUT.parent / "stage3"
benchmark = stage2.benchmark
SEEDS, NOISE_SD, FAMILIES = physical.SEEDS, physical.NOISE_SD, stage2.FAMILIES
GATES = dict(physical.GATES)
STAGE2_SEAL = "db0c067bbf8152c2fa66fc7dc26e84a0e82ba9df74e9781d269edac0ca44f099"
STAGE2_AUDITS = {
    "selected_replay_report_v1.json": "8503d700548be627d0e5d6534f84c271bbdc08ffd340ab37349481c6ac7d1e8d",
    "raw_replay_report_v1.json": "51d260638c353cec97e214b0ca8e5ec5d298ff4b17d702ea1e3f6d4afe6af5c9",
}
PHYSICAL_SEAL = "29553496204caa2cc7a075daea8132d468c3572acd19b087bb71fe934e8363b3"
CAVEATS = (
    "The positive-world native control has noiseless actual-tied RMSE about 0.384616 BPB; the matching "
    "effective-harm fit still has about 0.0144493 BPB. Relative calibration and raw-regret gates can be weak. "
    "Absolute errors and same-world/noiseless and same-seed/null comparisons remain visible and descriptive.",
    "The positive truth's numerical best-static versus best-two-phase gap is about 0.00178905 BPB, whereas "
    "its held-menu same-aggregate gain RMS is about 0.0395639 BPB. These quantities differ; numerical "
    "incumbents have no global certificate. Noisy qualification cannot establish a large empirical gain.",
    "Three paired Gaussian SD0.003 seeds on the fixed strong truth do not establish weak-effect sensitivity, "
    "empirical noise, a false-positive rate, transport or promotion. Lambda1 retains the all-early E_eff0 "
    "sink, with only native h(0) harm. No cap or endpoint exclusion is added.",
)
noisy_training_labels = physical.noisy_training_labels
comparison_values = physical.comparison_values


def context_specs() -> list[dict[str, Any]]:
    return [
        {
            **world,
            "truth_world": world["name"],
            "name": f"{world['name']}_noise003_seed{seed}",
            "noise_seed": seed,
            "noise_sd": NOISE_SD,
        }
        for world in stage2.WORLDS
        for seed in SEEDS
    ]


def source_paths() -> set[Path]:
    return (
        stage2.source_paths()
        | physical.source_paths()
        | {
            Path(__file__),
            ROOT / "test_tpp40_effective_harm_stage3_20261015.py",
            OUTPUT / "registration.json",
            stage2.OUTPUT / "execution_identity.json",
            stage2.OUTPUT / "result_manifest.json",
            stage2.OUTPUT / "summary.json",
            stage2.OUTPUT / "scores.csv",
            stage2.OUTPUT / "optimization_manifest.json",
            stage2.OUTPUT / "optimization_scores.csv",
            *(stage2.OUTPUT / "audit" / name for name in STAGE2_AUDITS),
            physical.OUTPUT / "execution_identity.json",
            physical.OUTPUT / "result_manifest.json",
        }
    )


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def verify_procedure_adoption() -> None:
    physical.registration()
    if physical.source_identity() != json.loads((physical.OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("The frozen physical-harm stage-three procedure changed")
    if benchmark.sha256(physical.OUTPUT / "result_manifest.json") != PHYSICAL_SEAL:
        raise ValueError("The adopted stage-three seal changed")
    seal = json.loads((physical.OUTPUT / "result_manifest.json").read_text())
    for name in ("registration.json", "execution_identity.json"):
        if benchmark.sha256(physical.OUTPUT / name) != seal["files"][name]["sha256"]:
            raise ValueError("Adopted stage-three source evidence differs from its seal")


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    expected = {
        "candidate_id": "TP40-029",
        "stage": 3,
        "status": "registered",
        "contexts": context_specs(),
        "families": list(FAMILIES),
        "noise_sd": NOISE_SD,
        "noise_seeds": list(SEEDS),
        "stage3_gates": GATES,
        "parameter_grids": {family: list(stage2.models.parameter_grid(family)) for family in FAMILIES},
        "gamma_grid": list(stage2.models.frozen.GAMMA_GRID),
        "optimization_seed": 20261015,
        "environment": benchmark.environment_identity(),
        "stage2_result_manifest_sha256": STAGE2_SEAL,
        "stage2_audits": STAGE2_AUDITS,
        "adopted_physical_stage3_seal_sha256": PHYSICAL_SEAL,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Registered effective-harm paired-noise protocol differs")
    paths = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(value["frozen_reference_inputs"]) != paths:
        raise ValueError("Registration must pin every executable source and input")
    benchmark.verify_hashes(value["frozen_reference_inputs"])
    verify_procedure_adoption()
    return value


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if current != json.loads((OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen effective-harm stage-three source, data or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": "effective_harm_stage3_paired_noise_and_raw_qualification",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Separate exact effective-harm stage-three approval is required")
    return current


def load_private(path: str, suffix: str) -> ModuleType:
    name = f"{__name__}._{suffix}"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def isolated_runner() -> ModuleType:
    """Reuse the reviewed flow while preserving both nested effective-harm bindings."""
    runner = load_private(physical.__file__, "stage3")
    # The effective stage2 wrapper embeds a physical-shaped implementation and
    # its raw I/O backend. Keep a separate unmodified prior for comparisons.
    prior = load_private(stage2.__file__, "prior")._RUNNER
    fitter = load_private(stage2.__file__, "fitter")._RUNNER
    for module in (fitter, fitter._RUNNER):
        module.OUTPUT = OUTPUT
        module.registration, module.execution_identity = registration, execution_identity
        module.source_identity = source_identity
    runner.OUTPUT, runner.FAMILIES, runner.stage2 = OUTPUT, FAMILIES, prior
    runner.STAGE2_SEAL, runner.STAGE2_AUDITS = STAGE2_SEAL, STAGE2_AUDITS
    runner.GATES, runner.CAVEATS = GATES, CAVEATS
    runner.registration, runner.execution_identity, runner.source_identity = (
        registration,
        execution_identity,
        source_identity,
    )
    runner._FITTER = fitter
    flow = runner._FLOW
    flow.OUTPUT, flow.FAMILIES, flow.stage2, flow._BACKEND = OUTPUT, FAMILIES, fitter, fitter._RUNNER
    flow.registration, flow.execution_identity = registration, execution_identity
    flow.qualification_checks = runner.qualification_checks
    return runner


_RUNNER = isolated_runner()
qualification_checks = _RUNNER.qualification_checks


def freeze() -> None:
    _RUNNER.freeze()


def prepare() -> None:
    _RUNNER.prepare()


def run_cell(spec: dict[str, Any], family: str) -> dict[str, Any]:
    return _RUNNER.run_cell(spec, family)


def optimize_context(spec: dict[str, Any]) -> dict[str, Any]:
    return _RUNNER.optimize_context(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "fit", "summarize", "optimize", "qualify"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "prepare":
        prepare()
    elif args.command == "fit":
        prepare()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for family in FAMILIES:
                futures = [executor.submit(run_cell, spec, family) for spec in registration()["contexts"]]
                for future in as_completed(futures):
                    print(json.dumps(future.result(), sort_keys=True), flush=True)
        _RUNNER._FLOW.summarize()
    elif args.command == "summarize":
        _RUNNER._FLOW.summarize()
    elif args.command == "qualify":
        _RUNNER.qualify()
    else:
        for spec in registration()["contexts"]:
            if spec["noise_seed"] == SEEDS[0]:
                _RUNNER.prepare_truth_optimum(spec)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(optimize_context, spec) for spec in registration()["contexts"]]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
        _RUNNER.qualify()


if __name__ == "__main__":
    main()
