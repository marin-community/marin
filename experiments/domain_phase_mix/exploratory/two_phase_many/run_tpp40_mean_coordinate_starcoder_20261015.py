# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Conditional shared-aggregate StarCoder screen for both mean controls.

Exactly75 heads reuse the frozen TP028 screen fitting/scoring flow. Both noisy
families must pass before preparation; separate exact approval precedes target
access. Each temporal family retains its own unmodified two-reference verdicts.
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

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_starcoder_20261015 as physical
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_mean_coordinate_stage3_20261015 as noisy
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mean_coordinate_joint_models_20261015 as models

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_mean_coordinate_starcoder_20261015"
benchmark = physical.benchmark
FROZEN, PREPARED, PANEL = physical.FROZEN, physical.PREPARED, physical.PANEL
REFERENCES, EVIDENCE, DATA_FILES = physical.REFERENCES, physical.EVIDENCE, physical.DATA_FILES
EXPECTED_HEADS, EXPECTED_PREDICTIONS = 75, 5190
APPROVAL_SCOPE = "mean_coordinate_starcoder_frozen_v3_shared_aggregate_screen"
selected_targets = physical.selected_targets


def registration_value() -> dict[str, Any]:
    return json.loads((OUTPUT / "registration.json").read_text())


def source_paths(registration: dict[str, Any]) -> set[Path]:
    return _ORIGINAL_SOURCE_PATHS(registration) | {
        Path(__file__),
        Path(models.__file__),
        ROOT / "test_tpp40_mean_coordinate_starcoder_20261015.py",
        physical.OUTPUT / "registration.json",
        physical.OUTPUT / "execution_identity.json",
        physical.OUTPUT / "adoption.json",
    }


def source_hashes() -> dict[str, str]:
    paths = source_paths(registration_value()) | {OUTPUT / "adoption.json"}
    return {str(path.resolve()): benchmark.sha256(path) for path in sorted(paths)}


def verify_procedure_adoption() -> None:
    physical.verify_registration(physical.registration_value())
    prior = {"sources": physical.source_hashes(), "environment": benchmark.environment_identity()}
    if prior != json.loads((physical.OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen TP028 empirical screen implementation or inputs changed")


def verify_noisy_prerequisite(registration: dict[str, Any]) -> None:
    """Require sealed complete qualification of both mean families and all seeds."""
    prerequisite = registration["noisy_prerequisite"]
    expected = {
        "registration": noisy.OUTPUT / "registration.json",
        "identity": noisy.OUTPUT / "execution_identity.json",
        "qualification": noisy.OUTPUT / "qualification.json",
        "manifest": noisy.OUTPUT / "result_manifest.json",
    }
    if set(prerequisite["receipts"]) != set(expected) or not prerequisite["independent_audits"]:
        raise ValueError("Both noisy families need a complete sealed qualification and independent audit")
    for key, path in expected.items():
        if prerequisite["receipts"][key] != benchmark.receipt(path):
            raise ValueError(f"Noisy prerequisite receipt changed: {key}")
    noisy.registration()
    frozen = json.loads(expected["identity"].read_text())
    if noisy.source_identity() != frozen:
        raise ValueError("The qualified noisy scientific identity changed")
    manifest = json.loads(expected["manifest"].read_text())
    benchmark.verify_hashes({name: item["sha256"] for name, item in manifest["files"].items()}, noisy.OUTPUT)
    for key in ("registration", "identity", "qualification"):
        path = expected[key]
        if benchmark.sha256(path) != manifest["files"][str(path.relative_to(noisy.OUTPUT))]["sha256"]:
            raise ValueError("Noisy prerequisite evidence is absent from its seal")
    qualification = json.loads(expected["qualification"].read_text())
    required = set()
    for spec in noisy.context_specs():
        temporal = spec["matching_family"]
        correct = ("aggregate_mariner", temporal) if spec["truth_world"] == "null" else (temporal,)
        for family in correct:
            gates = ["kinetic_boundary", "static_calibration", "raw_gain_error"]
            if spec["truth_world"] == "null":
                gates.append("false_positive_gain")
            required.update((temporal, spec["name"], family, gate) for gate in gates)
        if spec["truth_world"] == "positive":
            required.update(
                (temporal, spec["name"], temporal, gate)
                for gate in ("positive_gain_vs_zero_gain", "positive_gain_vs_aggregate_mariner")
            )
        required.add((temporal, spec["name"], temporal, "raw_regret_vs_aggregate"))
    checks = qualification["checks"]
    actual = {(item["matching_family"], item["context"], item["family"], item["gate"]) for item in checks}
    if (
        qualification["identity"] != frozen
        or qualification["contexts"] != 12
        or qualification["cells"] != 24
        or not qualification["passed"]
        or not all(item["passed"] for item in checks)
        or len(checks) != len(required)
        or actual != required
    ):
        raise ValueError("Every registered check for both noisy families must pass before the screen")
    for record in prerequisite["independent_audits"]:
        path = Path(record["path"])
        name = str(path.relative_to(noisy.OUTPUT))
        if record != benchmark.receipt(path) or manifest["files"][name]["sha256"] != record["sha256"]:
            raise ValueError("The independent noisy audit differs from its sealed receipt")
        if not json.loads(path.read_text())["passed"]:
            raise ValueError("The independent noisy audit must pass")


def verify_registration(registration: dict[str, Any]) -> None:
    expected = {
        "candidate_ids": noisy.stage2.CANDIDATES,
        "stage": "starcoder_empirical_screen",
        "status": "registered",
        "families": list(models.MODEL_NAMES),
        "models": list(models.MODEL_NAMES),
        "panel": PANEL,
        "targets": list(selected_targets()),
        "outer_folds": list(range(5)),
        "repeat": 0,
        "expected_contexts": 25,
        "expected_heads": EXPECTED_HEADS,
        "expected_prediction_rows_per_model": 1730,
        "expected_prediction_rows": EXPECTED_PREDICTIONS,
        "fit_strengths": list(models.STRENGTHS),
        "parameter_grids": {family: list(models.parameter_grid(family)) for family in models.MODEL_NAMES},
        "structural_tolerance_bpb": 1e-8,
        "environment": benchmark.environment_identity(),
        "shared_native_aggregate": True,
        "require_both_noisy_families_pass": True,
    }
    if any(registration.get(key) != value for key, value in expected.items()):
        raise ValueError("Registered complete shared-aggregate mean screen differs")
    for key, path in {
        "models_source": Path(models.__file__),
        "estimator_source": Path(models.frozen.__file__),
        "prepared_manifest": PREPARED / "manifest.json",
        "data_contract": benchmark.PROTOCOL_DIR / "data_contract.json",
        "protocol": benchmark.PROTOCOL_DIR / "protocol.json",
    }.items():
        if registration[key] != benchmark.receipt(path):
            raise ValueError(f"Registered source receipt differs: {key}")
    if set(registration["data_files"]) != set(DATA_FILES) or set(registration["references"]) != set(REFERENCES):
        raise ValueError("All frozen data inputs and both reference policies must be pinned")
    for filename, record in registration["data_files"].items():
        if record != benchmark.receipt(PREPARED / PANEL / filename):
            raise ValueError(f"Registered data receipt differs: {filename}")
    for reference in REFERENCES:
        record = registration["references"][reference]
        if record["policy"] != benchmark.receipt(FROZEN / "screen_policies" / f"{reference}.json") or record[
            "reference_predictions"
        ] != benchmark.receipt(FROZEN / "screen_calibration/predictions.csv"):
            raise ValueError(f"Registered reference differs: {reference}")
    paths = {str(path.resolve()) for path in source_paths(registration) - {OUTPUT / "registration.json"}}
    if set(registration["frozen_reference_inputs"]) != paths or registration.get("pending_reference_inputs"):
        raise ValueError("Every future prerequisite, source and input must be pinned before preparation")
    benchmark.verify_hashes(registration["frozen_reference_inputs"])
    verify_procedure_adoption()


def identity() -> dict[str, Any]:
    registration = registration_value()
    verify_registration(registration)
    current = {"sources": source_hashes(), "environment": benchmark.environment_identity()}
    if current != json.loads((OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen screen source, data or runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": APPROVAL_SCOPE,
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Separate exact empirical-screen approval is required before opening targets")
    return current


def isolated_screen() -> ModuleType:
    name = f"{__name__}._physical_screen"
    specification = importlib.util.spec_from_file_location(name, physical.__file__)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.OUTPUT, module.models, module.noisy = OUTPUT, models, noisy
    module.EXPECTED_HEADS, module.EXPECTED_PREDICTIONS = EXPECTED_HEADS, EXPECTED_PREDICTIONS
    module.APPROVAL_SCOPE = APPROVAL_SCOPE
    module.registration_value, module.verify_registration = registration_value, verify_registration
    module.verify_noisy_prerequisite, module.source_hashes, module.identity = (
        verify_noisy_prerequisite,
        source_hashes,
        identity,
    )
    module._SCREEN = module.isolated_screen()
    return module


_BASE = isolated_screen()
_ORIGINAL_SOURCE_PATHS = _BASE.source_paths
fit_partition = _BASE.fit_partition


def prepare() -> None:
    _BASE.prepare()


def fit_context(task: tuple[int, int]) -> dict[str, Any]:
    return _BASE.fit_context(task)


def summarize() -> None:
    """Replay all 75 heads and frozen held-out memberships before native scoring."""
    identity()
    panel = benchmark.read_panel(PREPARED, PANEL)
    frames, parameters = [], []
    receipts = {family: [] for family in models.MODEL_NAMES}
    mapping_errors = {family: [] for family in models.MODEL_NAMES}
    probe = np.column_stack([np.linspace(0, 1, 257), np.linspace(1, 0, 257)])
    e0, e1 = probe * panel.c0, probe * panel.c1
    for outer in range(5):
        for task_index, target in enumerate(selected_targets()):
            folder = OUTPUT / f"fits/fold{outer}_task{task_index}"
            if not _BASE._SCREEN.completed(folder, {"stage": "predictions", "fold": outer, "target": target}):
                raise ValueError(f"Incomplete screen context: {folder}")
            training, query, _train, held = _BASE._SCREEN.partition_rows(panel, target, outer)
            frame = pd.read_csv(folder / "predictions.csv", keep_default_na=False, float_precision="round_trip")
            for family in models.MODEL_NAMES:
                cell = folder / family
                if not _BASE._SCREEN.completed(cell, {"stage": "fit", "fold": outer, "target": target, "model": family}):
                    raise ValueError(f"Incomplete model: {cell}")
                payload = json.loads((cell / "fit.json").read_text())
                if (
                    payload["training_rows"] != training.row_id.tolist()
                    or payload["inner_folds"] != training.inner_fold.tolist()
                ):
                    raise ValueError("Recorded training membership differs from frozen folds")
                fitted = models.JointFit.from_json(payload)
                selected = frame[frame.model.eq(family)]
                if selected.row_id.tolist() != query.row_id.tolist() or selected.role.tolist() != query.role.tolist():
                    raise ValueError("Prediction membership differs from frozen held-out rows")
                np.testing.assert_array_equal(selected.observed.to_numpy(), panel.targets[target][held])
                np.testing.assert_array_equal(
                    selected.predicted.to_numpy(), fitted.predict(panel.phase0_epochs[held], panel.phase1_epochs[held])
                )
                expected = fitted.head.predict(models.mariner.design_matrix(e0 + e1, fitted.shape))
                error = float(np.max(abs(fitted.predict(e0, e1) - expected)))
                if error > 1e-8:
                    raise ValueError("Tied MARINER mapping failed")
                mapping_errors[family].append(error)
                receipts[family].append(benchmark.receipt(cell / "fit.json"))
                parameters.append(
                    {
                        "model": family,
                        "target": target,
                        "outer_fold": outer,
                        "lambda": fitted.parameters.get("lambda"),
                        "lambda_at_physical_upper_boundary": fitted.parameters.get("lambda") == 1.0,
                        "gamma": fitted.kappa,
                        "floor": fitted.head.floor,
                        "ridge": fitted.ridge,
                        "inner_rmse": fitted.inner_rmse,
                        **fitted.shape,
                    }
                )
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True).sort_values(["model", "target", "outer_fold", "row_id"])
    if len(predictions) != EXPECTED_PREDICTIONS or predictions.duplicated(["model", "target", "row_id"]).any():
        raise ValueError("The three-family screen requires 5,190 unique held-out predictions")
    if predictions.groupby("model").size().to_dict() != {family: 1730 for family in models.MODEL_NAMES}:
        raise ValueError("Each family requires 1,730 held-out predictions")
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    pd.DataFrame(parameters).to_csv(OUTPUT / "selected_parameters.csv", index=False)
    benchmark.scoring.score_predictions(predictions).to_csv(OUTPUT / "metrics.csv", index=False)
    benchmark.scoring.summarize_predictions(predictions).to_csv(OUTPUT / "summary.csv", index=False)
    policy = json.loads((FROZEN / "screen_policies/effective_exposure_dsp.json").read_text())
    outputs = ["predictions.csv", "selected_parameters.csv", "metrics.csv", "summary.csv"]
    for family in models.MODEL_NAMES:
        destination = OUTPUT / "candidates" / family
        destination.mkdir(parents=True, exist_ok=True)
        predictions[predictions.model.eq(family)].to_csv(destination / "predictions.csv", index=False)
        evidence_receipts = {}
        for name in EVIDENCE:
            record = {
                "evidence": name,
                "protocol_id": policy["benchmark_protocol_id"],
                "candidate_model": family,
                "data_contract_sha256": policy["data_contract_sha256"],
                "status": "pass",
                "registration": benchmark.receipt(OUTPUT / "registration.json"),
                "fitted_receipts": receipts[family],
                "adoption": benchmark.receipt(OUTPUT / "adoption.json"),
            }
            if name == "tied_class_mapping":
                record["proof"] = (
                    "At E1=(1-alpha)*E, the normalized late exposure equals E. "
                    "Both hazard-coordinate and benefit-coordinate means therefore equal native b(E); "
                    "harm remains native h(E). Zero dose is native, the head/link are unchanged, "
                    "and lambda0 plus the aggregate family delegate to the native design exactly."
                )
                record["maximum_probe_error_bpb"] = max(mapping_errors[family])
            if name in ("asymmetric_only_fitting", "nested_split_integrity"):
                record["verification"] = (
                    "All 25 training memberships and inner labels equal frozen folds; tied rows excluded. "
                    "Held-out predictions replay exactly. The fitting API receives only eligible training labels."
                )
            path = destination / f"{name}.json"
            benchmark.write_json(path, record)
            evidence_receipts[name] = [benchmark.receipt(path)]
            outputs.append(str(path.relative_to(OUTPUT)))
        benchmark.write_json(destination / "evidence.json", evidence_receipts)
        outputs.extend(str((destination / name).relative_to(OUTPUT)) for name in ("predictions.csv", "evidence.json"))
    _BASE._SCREEN.finish(OUTPUT, {"stage": "summary"}, tuple(outputs))
    print(json.dumps({"summary": True, "heads": EXPECTED_HEADS, "predictions": len(predictions)}))


def gate() -> None:
    _BASE.gate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "summarize", "gate"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "fit":
        identity()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = [(fold, task) for fold in range(5) for task in range(5)]
            for future in as_completed(pool.submit(fit_context, task) for task in tasks):
                print(json.dumps(future.result()), flush=True)
    elif args.stage == "summarize":
        summarize()
    else:
        gate()


if __name__ == "__main__":
    main()
