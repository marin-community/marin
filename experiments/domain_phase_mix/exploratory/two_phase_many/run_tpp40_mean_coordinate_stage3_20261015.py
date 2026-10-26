# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Matching-law paired-noise qualification, conditional on both stage-two passes.

Privately reuses frozen TP028 stage-three scoring, gates and raw optimization.
The future exact stage-two seal and independent audits must be registered before
any labels or fits. No partial-family, cross-law or empirical execution exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_stage3_20261015 as physical
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_mean_coordinate_stage2_20261015 as stage2

ROOT = Path(__file__).resolve().parent
OUTPUT = stage2.OUTPUT.parent / "stage3"
benchmark = stage2.benchmark
FAMILIES, SEEDS, NOISE_SD = stage2.FAMILIES, physical.SEEDS, physical.NOISE_SD
GATES = dict(physical.GATES)
PHYSICAL_SEAL = "29553496204caa2cc7a075daea8132d468c3572acd19b087bb71fe934e8363b3"
APPROVAL_SCOPE = "mean_coordinate_stage3_matching_paired_noise_and_raw"
STAGE2_AUDIT_NAMES = ("selected_replay_report_v1.json", "raw_replay_report_v1.json")
CAVEATS = (
    "Relative tied-calibration and signed-regret gates may pass while absolute calibration remains poor. "
    "Report all absolute errors and same-world noiseless/same-seed null comparisons without new thresholds.",
    "Held-menu same-aggregate gain RMS and the numerical truth static-to-free gap differ. Neither a menu "
    "pass nor a numerical incumbent certifies a continuous global optimum or token savings.",
    "Three paired Gaussian SD0.003 seeds on unchanged lambda .5 truths do not establish empirical noise, "
    "weak-effect sensitivity, a population false-positive rate, transport or promotion. Both mean families "
    "retain physical harm and the lambda1 all-early benefit-zero endpoint.",
)
noisy_training_labels = physical.noisy_training_labels


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
            ROOT / "test_tpp40_mean_coordinate_stage3_20261015.py",
            OUTPUT / "registration.json",
            stage2.OUTPUT / "execution_identity.json",
            stage2.OUTPUT / "result_manifest.json",
            stage2.OUTPUT / "summary.json",
            stage2.OUTPUT / "scores.csv",
            stage2.OUTPUT / "optimization_manifest.json",
            stage2.OUTPUT / "optimization_scores.csv",
            *(stage2.OUTPUT / "audit" / name for name in STAGE2_AUDIT_NAMES),
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
        raise ValueError("Frozen TP028 noisy procedure changed")
    seal_path = physical.OUTPUT / "result_manifest.json"
    if benchmark.sha256(seal_path) != PHYSICAL_SEAL:
        raise ValueError("Frozen TP028 noisy seal changed")
    sealed = json.loads(seal_path.read_text())["files"]
    for name in ("registration.json", "execution_identity.json"):
        if benchmark.sha256(physical.OUTPUT / name) != sealed[name]["sha256"]:
            raise ValueError("Adopted noisy procedure evidence differs from its seal")


def verify_stage2_adoption(value: dict[str, Any]) -> None:
    """Require the entire future eight-head stage; neither family is optional."""
    stage2.registration()
    identity = json.loads((stage2.OUTPUT / "execution_identity.json").read_text())
    if stage2.source_identity() != identity:
        raise ValueError("Stage-two source identity changed")
    digest = value["stage2_result_manifest_sha256"]
    audit_pins = value["stage2_audits"]
    if not isinstance(digest, str) or len(digest) != 64 or set(audit_pins) != set(STAGE2_AUDIT_NAMES):
        raise ValueError("Register exact future stage-two seal and both independent audits")
    manifest_path = stage2.OUTPUT / "result_manifest.json"
    if benchmark.sha256(manifest_path) != digest:
        raise ValueError("Required stage-two seal changed")
    sealed = json.loads(manifest_path.read_text())["files"]
    benchmark.verify_hashes({name: item["sha256"] for name, item in sealed.items()}, stage2.OUTPUT)
    for name in ("summary.json", "execution_identity.json", "optimization_manifest.json"):
        if benchmark.sha256(stage2.OUTPUT / name) != sealed[name]["sha256"]:
            raise ValueError("Stage-two prerequisite is absent from the parent seal")
    summary = json.loads((stage2.OUTPUT / "summary.json").read_text())
    expected = {
        (world["matching_family"], world["name"], family)
        for world in stage2.WORLDS
        for family in ("aggregate_mariner", world["matching_family"])
    }
    actual = {(item["matching_family"], item["world"], item["family"]) for item in summary["checks"]}
    if summary["identity"] != identity or summary["cells"] != 8 or len(summary["checks"]) != 8 or actual != expected:
        raise ValueError("Stage-two summary must cover all eight registered heads")
    keys = ("null_gain_pass", "gain_recovery_pass", "tied_calibration_pass", "kinetic_boundary_pass")
    if not summary["passed"] or not all(all(item[key] for key in keys) for item in summary["checks"]):
        raise ValueError("Both mean families must pass stage two before any noisy execution")
    raw = json.loads((stage2.OUTPUT / "optimization_manifest.json").read_text())
    if raw["identity"] != identity or raw["cells"] != 8:
        raise ValueError("All eight stage-two raw diagnostics are required")
    benchmark.verify_hashes(raw["outputs"], stage2.OUTPUT)
    benchmark.verify_hashes(raw["family_artifacts"])
    for name, audit_digest in audit_pins.items():
        path = stage2.OUTPUT / "audit" / name
        if benchmark.sha256(path) != audit_digest or audit_digest != sealed[f"audit/{name}"]["sha256"]:
            raise ValueError("Independent stage-two audit differs from the registered parent seal")
        if not json.loads(path.read_text())["passed"]:
            raise ValueError("Independent selected and raw stage-two replay must both pass")


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    expected = {
        "candidate_ids": stage2.CANDIDATES,
        "stage": 3,
        "status": "registered",
        "contexts": context_specs(),
        "matching_families": list(FAMILIES),
        "families": list(stage2.models.MODEL_NAMES),
        "noise_sd": NOISE_SD,
        "noise_seeds": list(SEEDS),
        "stage3_gates": GATES,
        "parameter_grids": {family: list(stage2.models.parameter_grid(family)) for family in stage2.models.MODEL_NAMES},
        "gamma_grid": list(stage2.models.frozen.GAMMA_GRID),
        "optimization_seed": 20261015,
        "environment": benchmark.environment_identity(),
        "adopted_physical_stage3_seal_sha256": PHYSICAL_SEAL,
        "execution_counts": {"contexts": 12, "learned_heads": 24, "truth_raw_cells": 4, "fitted_raw_cells": 24},
        "cross_law_fits": False,
        "require_both_stage2_families_pass": True,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Registered complete matching-law noisy protocol differs")
    paths = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(value["frozen_reference_inputs"]) != paths or value.get("pending_reference_inputs"):
        raise ValueError("Complete future prerequisite pins are required before registration")
    benchmark.verify_hashes(value["frozen_reference_inputs"])
    verify_procedure_adoption()
    verify_stage2_adoption(value)
    return value


def freeze() -> None:
    registration()
    current = source_identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and json.loads(path.read_text()) != current:
        raise ValueError("Preserve the earlier identity and register a separate revision")
    benchmark.write_json(path, current)


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if current != json.loads((OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen matching-law noisy source, data or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": APPROVAL_SCOPE,
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != item for key, item in expected.items()):
        raise ValueError("Separate exact matching-law noisy execution approval is required")
    return current


def family_registration(family: str) -> dict[str, Any]:
    value = registration()
    return {
        **value,
        "contexts": [spec for spec in context_specs() if spec["matching_family"] == family],
        "families": ["aggregate_mariner", family],
    }


def load_private(path: str, suffix: str) -> ModuleType:
    name = f"{__name__}._{suffix}"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def isolated_runner(family: str) -> ModuleType:
    runner = load_private(physical.__file__, f"{family}_stage3")
    fitter = load_private(stage2.__file__, f"{family}_stage2").RUNNERS[family]
    family_output = OUTPUT / family
    family_config = partial(family_registration, family)
    for module in (fitter, fitter._RUNNER):
        module.OUTPUT = family_output
        module.registration, module.execution_identity = family_config, execution_identity
        module.source_identity = source_identity
    runner.OUTPUT, runner.FAMILIES = family_output, ("aggregate_mariner", family)
    runner.stage2 = stage2.RUNNERS[family]
    runner.GATES, runner.CAVEATS = GATES, CAVEATS
    runner.registration, runner.execution_identity = family_config, execution_identity
    runner._FITTER = fitter
    flow = runner._FLOW
    flow.OUTPUT, flow.FAMILIES, flow.stage2, flow._BACKEND = family_output, runner.FAMILIES, fitter, fitter._RUNNER
    flow.registration, flow.execution_identity = family_config, execution_identity
    flow.qualification_checks = runner.qualification_checks
    return runner


RUNNERS = {family: isolated_runner(family) for family in FAMILIES}


def context_runner(spec: dict[str, Any]) -> ModuleType:
    if spec not in context_specs():
        raise ValueError("Context is outside the complete registered paired-noise design")
    return RUNNERS[spec["matching_family"]]


def prepare() -> None:
    identity = execution_identity()
    arrays, folds, head = stage2.oracle.ORACLES[FAMILIES[0]].inputs()
    dependency = stage2.OUTPUT / "result_manifest.json"
    for spec in context_specs():
        runner = context_runner(spec)
        backend = runner._FITTER._RUNNER
        folder = runner.OUTPUT / "contexts" / spec["name"]
        if backend.completed(folder, spec, identity):
            continue
        prior = stage2.RUNNERS[spec["matching_family"]]
        values = prior.oracle.world_inputs(arrays, folds, head, spec["parameters"]["lambda"])
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
            {
                "shape": prior.oracle.SHAPE,
                "head": prior.oracle.head_record(head),
                "parameters": spec["parameters"],
                "world": spec,
            },
        )
        backend.finish(
            folder, spec, identity, ("synthetic.npz", "truth.json"), {str(dependency): benchmark.sha256(dependency)}
        )


def run_cell(spec: dict[str, Any], family: str) -> dict[str, Any]:
    runner = context_runner(spec)
    if family not in runner.FAMILIES:
        raise ValueError("Cross-law fitting is outside this protocol")
    return runner.run_cell(spec, family)


def optimize_context(spec: dict[str, Any]) -> dict[str, Any]:
    return context_runner(spec).optimize_context(spec)


def summarize(stage: str) -> None:
    identity = execution_identity()
    if stage not in ("fit", "qualification"):
        raise ValueError(f"Unknown stage: {stage}")
    name = "fit_summary.json" if stage == "fit" else "qualification.json"
    csv_name = "scores.csv" if stage == "fit" else "optimization_scores.csv"
    frames, records, dependencies = [], [], {}
    for family, runner in RUNNERS.items():
        if stage == "fit":
            runner._FLOW.summarize()
        else:
            runner._FLOW.qualify()
        record = json.loads((runner.OUTPUT / name).read_text())
        if record["identity"] != identity or record["cells"] != 12:
            raise ValueError("Family noisy summary identity or completeness differs")
        benchmark.verify_hashes(record["cell_receipts"])
        benchmark.verify_hashes(record["outputs"], runner.OUTPUT)
        frame = pd.read_csv(runner.OUTPUT / csv_name, keep_default_na=False)
        key = "world" if stage == "fit" else "context"
        expected = {
            (spec["name"], model)
            for spec in context_specs()
            if spec["matching_family"] == family
            for model in runner.FAMILIES
        }
        if len(frame) != 12 or set(zip(frame[key], frame["family"], strict=True)) != expected:
            raise ValueError("A registered noisy cell is missing or duplicated")
        frame.insert(0, "matching_family", family)
        frames.append(frame)
        records.append(record)
        for filename in (name, csv_name):
            dependencies[str(runner.OUTPUT / filename)] = benchmark.sha256(runner.OUTPUT / filename)
    pd.concat(frames, ignore_index=True).to_csv(OUTPUT / csv_name, index=False)
    result: dict[str, Any] = {
        "identity": identity,
        "contexts": 12,
        "cells": 24,
        "family_artifacts": dependencies,
        "outputs": {csv_name: benchmark.sha256(OUTPUT / csv_name)},
        "scope": "Matching-law paired-noise qualification; no partial-family execution, empirical fit or promotion.",
    }
    if stage == "qualification":
        result["passed"] = all(record["passed"] for record in records)
        result["checks"] = [
            {**check, "matching_family": family}
            for family, record in zip(FAMILIES, records, strict=True)
            for check in record["checks"]
        ]
    benchmark.write_json(OUTPUT / name, result)


def diagnostic_comparisons() -> None:
    """Preserve frozen absolute/signed comparisons using the shared parent seal."""
    identity = execution_identity()
    config = registration()
    sealed = json.loads((stage2.OUTPUT / "result_manifest.json").read_text())["files"]
    rows = []
    for spec in context_specs():
        matching = spec["matching_family"]
        output = OUTPUT / matching
        optimum = json.loads((output / "truth_optima" / spec["truth_world"] / "optimization.json").read_text())
        for family in ("aggregate_mariner", matching):
            current = output / "contexts" / spec["name"]
            noiseless = stage2.OUTPUT / matching / "contexts" / spec["truth_world"]
            matched = output / "contexts" / f"null_noise003_seed{spec['noise_seed']}"
            row = {
                "matching_family": matching,
                "context": spec["name"],
                "truth_world": spec["truth_world"],
                "noise_seed": spec["noise_seed"],
                "family": family,
                "numerical_truth_static_to_free_gap": optimum["best_static"]["predicted"] - optimum["best"]["predicted"],
            }
            for subdir, filename, metrics in (
                ("cells", "scores.json", physical.SCORE_METRICS),
                ("optimization", "truth_scores.json", physical.RAW_METRICS),
            ):
                records = [
                    json.loads((folder / subdir / family / filename).read_text())
                    for folder in (current, noiseless, matched)
                ]
                prior = noiseless / subdir / family / filename
                if benchmark.sha256(prior) != sealed[str(prior.relative_to(stage2.OUTPUT))]["sha256"]:
                    raise ValueError("Sealed same-family noiseless diagnostic input changed")
                row.update(physical.comparison_values(records[0], records[1], records[2], metrics))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / "diagnostic_comparisons.csv", index=False)
    benchmark.write_json(
        OUTPUT / "diagnostic_comparisons.json",
        {
            "identity": identity,
            "rows": len(rows),
            "caveats": CAVEATS,
            "scope": "Descriptive absolute/signed error differences; no additional gates or causal decomposition.",
            "outputs": {"diagnostic_comparisons.csv": benchmark.sha256(OUTPUT / "diagnostic_comparisons.csv")},
            "dependencies": {
                str(OUTPUT / "qualification.json"): benchmark.sha256(OUTPUT / "qualification.json"),
                str(stage2.OUTPUT / "result_manifest.json"): config["stage2_result_manifest_sha256"],
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "fit", "summarize", "optimize", "qualify"))
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
                    executor.submit(run_cell, spec, spec["matching_family"] if temporal_wave else "aggregate_mariner")
                    for spec in context_specs()
                ]
                for future in as_completed(futures):
                    print(json.dumps(future.result(), sort_keys=True), flush=True)
        summarize("fit")
    elif args.command == "summarize":
        summarize("fit")
    elif args.command == "qualify":
        summarize("qualification")
        diagnostic_comparisons()
    else:
        for spec in context_specs():
            if spec["noise_seed"] == SEEDS[0]:
                context_runner(spec).prepare_truth_optimum(spec)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(optimize_context, spec) for spec in context_specs()]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
        summarize("qualification")
        diagnostic_comparisons()


if __name__ == "__main__":
    main()
