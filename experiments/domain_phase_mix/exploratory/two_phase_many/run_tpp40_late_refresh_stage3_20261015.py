# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Separate paired-noise late-refresh qualification with unchanged fitted procedure.

Privately adopts the sealed stage-two fitter and previously audited stage-three
orchestration. No model equations, fitting search, optimization starts or score
rules change. Scientific execution requires a separately frozen registration
and explicit approval. Absolute-error comparisons remain descriptive.
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
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_stage2_20261015 as stage2
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_rate_joint_stage3_20261015 as reused

ROOT = Path(__file__).resolve().parent
OUTPUT = stage2.OUTPUT.parent / "stage3"
SEEDS = (20261015, 20261016, 20261017)
NOISE_SD = 0.003
FAMILIES = stage2.FAMILIES
STAGE2_SEAL = "1a1b7fa05d98649554caa8124a4f602b312af7ff1d7d4714baf59c7172b2bf7f"
STAGE2_AUDITS = {
    "selected_replay_report_v1.json": "d2f49890e6d23860be452cf788ec51413780f5b66f3f944927ca2e87a26c99bb",
    "raw_replay_report_v1.json": "953a73523a1b169198c7613adebbe34c92fabde8817cf486d2c5f9fa45d75259",
}
GATES = {
    "actual_tied_rmse_extra_vs_aggregate_bpb": 0.003,
    "null_max_positive_menu_or_raw_sameaggregate_gain_bpb": 0.003,
    "raw_gain_absolute_error_max_bpb": 0.003,
    "signed_raw_regret_extra_vs_aggregate_bpb": 0.001,
    "temporal_upper_boundary": 1.0,
}
CAVEATS = (
    "The positive-world aggregate control is severely misspecified: its noiseless actual-tied RMSE is about "
    "0.388 BPB. Relative tied-calibration and raw-regret gates alone are therefore weak. Absolute errors, "
    "noiseless same-world/same-family comparisons and same-seed null-fit comparisons remain visible and descriptive.",
    "The fixed positive truth has a numerical best-two-phase versus best-static gap of about 0.000745 BPB, "
    "despite a held-menu same-aggregate gain RMS near 0.038 BPB. These are different quantities; menu recovery "
    "does not imply a large globally optimized phase advantage; numerical incumbents lack global certification.",
    "Three paired Gaussian SD0.003 seeds on a strong fixed truth do not estimate empirical noise, weak-effect "
    "sensitivity, a population false-positive rate, or real TPP40 gain. No empirical execution or promotion follows.",
)
SCORE_METRICS = (
    "clean_asymmetric_rmse",
    "clean_actual_tied_rmse",
    "clean_matched_tied_rmse",
    "clean_gain_rmse",
    "clean_gain_rms",
    "predicted_frontier_rmse",
    "predicted_frontier_mean_optimism",
    "predicted_frontier_max_optimism",
    "clean_menu_regret_at1",
    "clean_menu_regret_at3",
    "clean_menu_regret_at5",
)
RAW_METRICS = (
    "true_gain_same_aggregate",
    "phase_gain_overprediction",
    "selected_optimism",
    "regret_vs_truth_numerical_incumbent",
    "regret_vs_truth_static_incumbent",
)
noisy_training_labels = reused.noisy_training_labels


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
    return stage2.source_paths() | {
        Path(__file__),
        Path(reused.__file__),
        ROOT / "test_tpp40_late_refresh_stage3_20261015.py",
        OUTPUT / "registration.json",
        stage2.OUTPUT / "execution_identity.json",
        stage2.OUTPUT / "result_manifest.json",
        stage2.OUTPUT / "summary.json",
        stage2.OUTPUT / "scores.csv",
        stage2.OUTPUT / "optimization_manifest.json",
        stage2.OUTPUT / "optimization_scores.csv",
        *(stage2.OUTPUT / "audit" / name for name in STAGE2_AUDITS),
    }


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    expected = {
        "candidate_id": "TP40-028",
        "stage": 3,
        "status": "registered",
        "contexts": context_specs(),
        "families": list(FAMILIES),
        "noise_sd": NOISE_SD,
        "noise_seeds": list(SEEDS),
        "stage3_gates": GATES,
        "parameter_grids": {family: list(stage2.models.parameter_grid(family)) for family in FAMILIES},
        "gamma_grid": list(stage2.joint.GAMMA_GRID),
        "optimization_seed": 20261015,
        "environment": benchmark.environment_identity(),
        "stage2_result_manifest_sha256": STAGE2_SEAL,
        "stage2_audits": STAGE2_AUDITS,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Registration differs from the separate paired-noise late-refresh protocol")
    expected_sources = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(value["frozen_reference_inputs"]) != expected_sources:
        raise ValueError("Registration must pin every executable source and input")
    benchmark.verify_hashes(value["frozen_reference_inputs"])
    return value


def freeze() -> None:
    registration()
    if stage2.source_identity() != json.loads((stage2.OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Stage-two scientific sources or inputs changed")
    if benchmark.sha256(stage2.OUTPUT / "result_manifest.json") != STAGE2_SEAL:
        raise ValueError("Required independently audited stage-two seal changed")
    manifest = json.loads((stage2.OUTPUT / "result_manifest.json").read_text())
    benchmark.verify_hashes({name: item["sha256"] for name, item in manifest["files"].items()}, stage2.OUTPUT)
    summary = json.loads((stage2.OUTPUT / "summary.json").read_text())
    if summary["cells"] != 4 or not summary["passed"]:
        raise ValueError("All four stage-two heads must pass before the separate noisy stage")
    for name, digest in STAGE2_AUDITS.items():
        path = stage2.OUTPUT / "audit" / name
        if benchmark.sha256(path) != digest or not json.loads(path.read_text())["passed"]:
            raise ValueError("Both independent stage-two audits must pass unchanged")
    identity = source_identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Preserve the earlier identity and register a separate revision")
    benchmark.write_json(path, identity)


def execution_identity() -> dict[str, Any]:
    registration()
    identity = source_identity()
    if identity != json.loads((OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen stage-three source, data or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    required = {
        "scope": "late_refresh_stage3_paired_noise_and_raw_qualification",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in required.items()):
        raise ValueError("Exact separate stage-three execution approval is required")
    return identity


def load_private(path: str, suffix: str) -> ModuleType:
    name = f"{__name__}._{suffix}"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def isolated_backends() -> tuple[ModuleType, ModuleType]:
    fitter = load_private(stage2.__file__, "fitter")
    fitter.OUTPUT, fitter.registration, fitter.execution_identity = OUTPUT, registration, execution_identity
    io = fitter._RUNNER
    io.OUTPUT, io.registration, io.execution_identity = OUTPUT, registration, execution_identity
    orchestrator = load_private(reused.__file__, "orchestrator")
    orchestrator.OUTPUT, orchestrator.FAMILIES = OUTPUT, FAMILIES
    orchestrator.stage2, orchestrator._BACKEND = fitter, io
    orchestrator.registration, orchestrator.execution_identity = registration, execution_identity
    return fitter, orchestrator


_FITTER, _FLOW = isolated_backends()


def prepare() -> None:
    identity = execution_identity()
    arrays, folds, head = stage2.oracle.inputs()
    for spec in registration()["contexts"]:
        folder = OUTPUT / "contexts" / spec["name"]
        if _FITTER._RUNNER.completed(folder, spec, identity):
            continue
        values = stage2.oracle.world_inputs(arrays, folds, head, spec["parameters"]["lambda"])
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
                "shape": stage2.oracle.SHAPE,
                "head": stage2.oracle.head_record(head),
                "parameters": spec["parameters"],
                "world": spec,
            },
        )
        dependency = stage2.OUTPUT / "result_manifest.json"
        _FITTER._RUNNER.finish(
            folder, spec, identity, ("synthetic.npz", "truth.json"), {str(dependency): benchmark.sha256(dependency)}
        )


def run_cell(spec: dict[str, Any], family: str) -> dict[str, Any]:
    return _FITTER.run_cell(spec, family)


def prepare_truth_optimum(spec: dict[str, Any]) -> None:
    _FLOW.prepare_truth_optimum(spec)


def optimize_context(spec: dict[str, Any]) -> dict[str, Any]:
    return _FLOW.optimize_context(spec)


def qualification_checks(
    spec: dict[str, Any], scores: dict[str, Any], raw: dict[str, Any], fits: dict[str, Any], gates: dict[str, Any]
) -> list[dict[str, Any]]:
    native, temporal = FAMILIES
    checks = []
    correct = FAMILIES if spec["truth_world"] == "null" else (temporal,)
    for family in correct:
        upper = family == temporal and fits[family].parameters["lambda"] == gates["temporal_upper_boundary"]
        checks.append({"context": spec["name"], "family": family, "gate": "kinetic_boundary", "passed": not upper})
        limit = scores[native]["clean_actual_tied_rmse"] + gates["actual_tied_rmse_extra_vs_aggregate_bpb"]
        value = scores[family]["clean_actual_tied_rmse"]
        checks.append(
            {
                "context": spec["name"],
                "family": family,
                "gate": "static_calibration",
                "value": value,
                "limit": limit,
                "passed": value <= limit,
            }
        )
        if spec["truth_world"] == "null":
            value = max(scores[family]["max_positive_predicted_gain"], raw[family]["predicted_gain_same_aggregate"])
            limit = gates["null_max_positive_menu_or_raw_sameaggregate_gain_bpb"]
            checks.append(
                {
                    "context": spec["name"],
                    "family": family,
                    "gate": "false_positive_gain",
                    "value": value,
                    "limit": limit,
                    "passed": value <= limit,
                }
            )
        value, limit = abs(raw[family]["phase_gain_overprediction"]), gates["raw_gain_absolute_error_max_bpb"]
        checks.append(
            {
                "context": spec["name"],
                "family": family,
                "gate": "raw_gain_error",
                "value": value,
                "limit": limit,
                "passed": value <= limit,
            }
        )
    if spec["truth_world"] == "positive":
        value = scores[temporal]["clean_gain_rmse"]
        for control, limit in (
            ("zero_gain", scores[temporal]["clean_gain_rms"]),
            (native, scores[native]["clean_gain_rmse"]),
        ):
            checks.append(
                {
                    "context": spec["name"],
                    "family": temporal,
                    "gate": f"positive_gain_vs_{control}",
                    "value": value,
                    "limit": limit,
                    "passed": value < limit,
                }
            )
    value = raw[temporal]["regret_vs_truth_numerical_incumbent"]
    limit = raw[native]["regret_vs_truth_numerical_incumbent"] + gates["signed_raw_regret_extra_vs_aggregate_bpb"]
    checks.append(
        {
            "context": spec["name"],
            "family": temporal,
            "gate": "raw_regret_vs_aggregate",
            "value": value,
            "limit": limit,
            "passed": value <= limit,
        }
    )
    for check in checks:
        check["applicability"] = (
            "Temporal versus aggregate signed regret in every context"
            if check["gate"] == "raw_regret_vs_aggregate"
            else (
                "Both families on null truth"
                if spec["truth_world"] == "null"
                else "Correctly specified temporal family; positive-world aggregate is a misspecified diagnostic control"
            )
        )
    return checks


_FLOW.qualification_checks = qualification_checks


def comparison_values(
    value: dict[str, Any], noiseless: dict[str, Any], matched_null: dict[str, Any], metrics: tuple[str, ...]
) -> dict[str, float]:
    result = {}
    for metric in metrics:
        result[metric] = value[metric]
        result[f"noiseless_same_family_{metric}"] = noiseless[metric]
        result[f"delta_vs_noiseless_{metric}"] = value[metric] - noiseless[metric]
        result[f"paired_null_same_family_{metric}"] = matched_null[metric]
        result[f"delta_vs_paired_null_{metric}"] = value[metric] - matched_null[metric]
    return result


def diagnostic_comparisons() -> None:
    identity = execution_identity()
    rows = []
    seal = json.loads((stage2.OUTPUT / "result_manifest.json").read_text())
    for spec in registration()["contexts"]:
        null_name = f"null_noise003_seed{spec['noise_seed']}"
        optimum = json.loads((OUTPUT / "truth_optima" / spec["truth_world"] / "optimization.json").read_text())
        for family in FAMILIES:
            current = OUTPUT / "contexts" / spec["name"]
            noiseless = stage2.OUTPUT / "contexts" / spec["truth_world"]
            matched = OUTPUT / "contexts" / null_name
            row = {
                "context": spec["name"],
                "truth_world": spec["truth_world"],
                "noise_seed": spec["noise_seed"],
                "family": family,
                "numerical_truth_static_to_free_gap": optimum["best_static"]["predicted"] - optimum["best"]["predicted"],
            }
            for subdir, filename, metrics in (
                ("cells", "scores.json", SCORE_METRICS),
                ("optimization", "truth_scores.json", RAW_METRICS),
            ):
                records = [
                    json.loads((folder / subdir / family / filename).read_text())
                    for folder in (current, noiseless, matched)
                ]
                prior_path = noiseless / subdir / family / filename
                prior_name = str(prior_path.relative_to(stage2.OUTPUT))
                if benchmark.sha256(prior_path) != seal["files"][prior_name]["sha256"]:
                    raise ValueError("A sealed noiseless comparison input changed")
                row.update(comparison_values(records[0], records[1], records[2], metrics))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / "diagnostic_comparisons.csv", index=False)
    qualification = OUTPUT / "qualification.json"
    benchmark.write_json(
        OUTPUT / "diagnostic_comparisons.json",
        {
            "identity": identity,
            "rows": len(rows),
            "caveats": CAVEATS,
            "scope": (
                "Descriptive error comparisons across different truths and fit noise; "
                "no new thresholds or causal decomposition."
            ),
            "outputs": {"diagnostic_comparisons.csv": benchmark.sha256(OUTPUT / "diagnostic_comparisons.csv")},
            "dependencies": {
                str(qualification): benchmark.sha256(qualification),
                str(stage2.OUTPUT / "result_manifest.json"): STAGE2_SEAL,
            },
        },
    )


def qualify() -> None:
    _FLOW.qualify()
    diagnostic_comparisons()


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
        _FLOW.summarize()
    elif args.command == "summarize":
        _FLOW.summarize()
    elif args.command == "qualify":
        qualify()
    else:
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
