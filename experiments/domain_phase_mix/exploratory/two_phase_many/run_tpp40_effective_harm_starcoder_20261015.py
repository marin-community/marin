# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Effective-harm late-refresh screen on immutable StarCoder v3 folds and references.

Preparation hashes outcomes without parsing values. A passing, independently
audited and sealed noisy qualification is required before preparation; a
separate exact approval is required before empirical target values are opened.
This development screen cannot establish mechanism, promotion or an optimum.
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
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_effective_harm_stage3_20261015 as noisy
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_tangent_starcoder_20261015 as reused
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_effective_harm_joint_models_20261015 as models

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_effective_harm_starcoder_20261015"
FROZEN, PREPARED, PANEL = reused.FROZEN, reused.PREPARED, reused.PANEL
REFERENCES, EVIDENCE = reused.REFERENCES, reused.EVIDENCE
DATA_FILES = ("policies.npz", "rows.csv", "outcomes.csv", "folds.csv", "metadata.json", "split_audit.json")
EXPECTED_HEADS, EXPECTED_PREDICTIONS = 50, 3460
APPROVAL_SCOPE = "effective_harm_starcoder_frozen_v3_empirical_screen"
selected_targets = reused.selected_targets


def registration_value() -> dict[str, Any]:
    return json.loads((OUTPUT / "registration.json").read_text())


def source_paths(registration: dict[str, Any]) -> set[Path]:
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    metadata = json.loads((PREPARED / PANEL / "metadata.json").read_text())
    return noisy.source_paths() | {
        Path(__file__),
        ROOT / "test_tpp40_effective_harm_starcoder_20261015.py",
        Path(reused.__file__),
        OUTPUT / "registration.json",
        PREPARED / "manifest.json",
        benchmark.PROTOCOL_DIR / "protocol.json",
        benchmark.PROTOCOL_DIR / "data_contract.json",
        *(Path(path) for path in manifest["identity"]["code"]),
        *(Path(path) for path in metadata["source_hashes"]),
        *(PREPARED / PANEL / name for name in DATA_FILES),
        *(FROZEN / "screen_policies" / f"{reference}.json" for reference in REFERENCES),
        *(
            FROZEN / "screen_calibration" / name
            for name in ("run.json", "predictions.csv", "metrics.csv", "summary.csv")
        ),
        *(Path(record["path"]) for record in registration["noisy_prerequisite"]["receipts"].values()),
        *(Path(record["path"]) for record in registration["noisy_prerequisite"]["independent_audits"]),
    }


def source_hashes() -> dict[str, str]:
    paths = source_paths(registration_value()) | {OUTPUT / "adoption.json"}
    return {str(path.resolve()): benchmark.sha256(path) for path in sorted(paths)}


def verify_noisy_prerequisite(registration: dict[str, Any]) -> None:
    prerequisite = registration["noisy_prerequisite"]
    expected = {
        "registration": noisy.OUTPUT / "registration.json",
        "identity": noisy.OUTPUT / "execution_identity.json",
        "qualification": noisy.OUTPUT / "qualification.json",
        "manifest": noisy.OUTPUT / "result_manifest.json",
    }
    if set(prerequisite["receipts"]) != set(expected) or not prerequisite["independent_audits"]:
        raise ValueError("A passing sealed noisy qualification and independent audits must be pinned")
    for key, path in expected.items():
        if prerequisite["receipts"][key] != benchmark.receipt(path):
            raise ValueError(f"Noisy prerequisite receipt changed: {key}")
    frozen_identity = json.loads(expected["identity"].read_text())
    if noisy.source_identity() != frozen_identity:
        raise ValueError("The qualified noisy-stage sources or inputs changed")
    manifest = json.loads(expected["manifest"].read_text())
    benchmark.verify_hashes({name: item["sha256"] for name, item in manifest["files"].items()}, noisy.OUTPUT)
    qualification = json.loads(expected["qualification"].read_text())
    if (
        qualification["identity"] != frozen_identity
        or qualification["contexts"] != 6
        or qualification["cells"] != 12
        or not qualification["passed"]
        or not qualification["checks"]
        or not all(check["passed"] for check in qualification["checks"])
    ):
        raise ValueError("Every registered noisy-stage check must pass before the empirical screen")
    for record in prerequisite["independent_audits"]:
        path = Path(record["path"])
        name = str(path.relative_to(noisy.OUTPUT))
        if (
            record != benchmark.receipt(path)
            or manifest["files"][name]["sha256"] != record["sha256"]
            or not json.loads(path.read_text())["passed"]
        ):
            raise ValueError("The sealed independent noisy-stage audit must pass unchanged")


def verify_registration(registration: dict[str, Any]) -> None:
    expected = {
        "candidate_id": "TP40-029",
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
    }
    for key, value in expected.items():
        if registration.get(key) != value:
            raise ValueError(f"Registered screen setting differs: {key}")
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
        raise ValueError("Every frozen panel input and both reference policies must be pinned")
    for filename, receipt in registration["data_files"].items():
        if receipt != benchmark.receipt(PREPARED / PANEL / filename):
            raise ValueError(f"Registered data receipt differs: {filename}")
    for reference in REFERENCES:
        record = registration["references"][reference]
        if record["policy"] != benchmark.receipt(FROZEN / "screen_policies" / f"{reference}.json") or record[
            "reference_predictions"
        ] != benchmark.receipt(FROZEN / "screen_calibration/predictions.csv"):
            raise ValueError(f"Registered reference receipts differ: {reference}")
    expected_paths = {str(path.resolve()) for path in source_paths(registration) - {OUTPUT / "registration.json"}}
    if set(registration["frozen_reference_inputs"]) != expected_paths:
        raise ValueError("Every executable source and input must have an exact registration pin")
    benchmark.verify_hashes(registration["frozen_reference_inputs"])


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
        raise ValueError("Separate exact empirical-screen execution approval is required")
    return current


def prepare() -> None:
    """Adopt frozen covariates and hashes after noisy qualification, without parsing outcomes."""
    registration = registration_value()
    verify_registration(registration)
    verify_noisy_prerequisite(registration)
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    benchmark.verify_hashes(manifest["sources"])
    benchmark.verify_hashes(manifest["outputs"], PREPARED)
    scientific, packaging = {}, {}
    for filename, expected in manifest["identity"]["code"].items():
        path = Path(filename)
        actual = benchmark.sha256(path)
        if path.name in ("pyproject.toml", "uv.lock"):
            packaging[filename] = {"historical": expected, "current": actual}
        else:
            if actual != expected:
                raise ValueError(f"Frozen scientific implementation changed: {filename}")
            scientific[filename] = actual
    environment = benchmark.environment_identity()
    if environment != manifest["identity"]["environment"]:
        raise ValueError("The frozen v3 numerical environment is required")
    if benchmark.sha256(benchmark.PROTOCOL_DIR / "data_contract.json") != manifest["identity"]["data_contract_sha256"]:
        raise ValueError("The frozen data contract changed")
    for reference in REFERENCES:
        policy = benchmark.verified_policy(FROZEN / "screen_policies" / f"{reference}.json", None, "preparation")
        if not policy["tolerances_frozen"]:
            raise ValueError(f"Frozen reference/tolerance receipts failed: {reference}")
    targets = selected_targets()
    if len(targets) != 5:
        raise ValueError("Expected five frozen quick-screen targets")
    metadata = json.loads((PREPARED / PANEL / "metadata.json").read_text())
    with np.load(PREPARED / PANEL / "policies.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["phase0_epochs"], arrays["phase0"] * arrays["c0"])
        np.testing.assert_array_equal(arrays["phase1_epochs"], arrays["phase1"] * arrays["c1"])
        np.testing.assert_allclose(arrays["c0"] / (arrays["c0"] + arrays["c1"]), metadata["alpha"], rtol=1e-12, atol=0)
    adoption = {
        "scope": "Unchanged v3 scientific inputs and runtime; current packaging hashes explicitly recorded.",
        "prepared_manifest": benchmark.receipt(PREPARED / "manifest.json"),
        "noisy_prerequisite": registration["noisy_prerequisite"],
        "registration": registration,
        "scientific_sources": scientific,
        "packaging_hashes": packaging,
        "environment": environment,
        "panel": PANEL,
        "targets": list(targets),
        "models": list(models.MODEL_NAMES),
        "repeat": 0,
        "folds": list(range(5)),
        "heads": EXPECTED_HEADS,
        "outcome_values_parsed": False,
    }
    destination = OUTPUT / "adoption.json"
    if destination.exists() and json.loads(destination.read_text()) != adoption:
        raise ValueError("Preserve the existing adoption and register a separate revision")
    benchmark.write_json(destination, adoption)
    current = {"sources": source_hashes(), "environment": environment}
    destination = OUTPUT / "execution_identity.json"
    if destination.exists() and json.loads(destination.read_text()) != current:
        raise ValueError("Preserve the existing execution identity and register a separate revision")
    benchmark.write_json(destination, current)
    print(json.dumps({"prepared": True, "heads": EXPECTED_HEADS, "outcome_values_parsed": False}))


def fit_partition(
    panel: benchmark.data.BenchPanel,
    target: str,
    train: np.ndarray,
    inner: np.ndarray,
    family: str,
    matched_null_fit: models.JointFit | None,
    *,
    test_shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: models.TestGrid | None = None,
) -> models.JointFit:
    """Fit eligible labels with the complete native competitor freshly rescored by J2."""
    return models.fit_joint(
        panel.phase0_epochs[train],
        panel.phase1_epochs[train],
        panel.targets[target][train],
        inner,
        panel.alpha,
        family,
        controls=() if matched_null_fit is None else (matched_null_fit,),
        shapes=test_shapes,
        test_grid=test_grid,
    )


def isolated_screen() -> ModuleType:
    name = f"{__name__}._screen"
    specification = importlib.util.spec_from_file_location(name, reused.__file__)
    assert specification is not None and specification.loader is not None
    screen = importlib.util.module_from_spec(specification)
    sys.modules[name] = screen
    specification.loader.exec_module(screen)
    screen.OUTPUT, screen.models = OUTPUT, models
    screen.identity, screen.fit_partition = identity, fit_partition
    return screen


_SCREEN = isolated_screen()


def fit_context(task: tuple[int, int]) -> dict[str, Any]:
    return _SCREEN.fit_context(task)


def summarize() -> None:
    """Replay all 50 heads and frozen held-out memberships before native scoring."""
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
            if not _SCREEN.completed(folder, {"stage": "predictions", "fold": outer, "target": target}):
                raise ValueError(f"Incomplete screen context: {folder}")
            training, query, _train, held = _SCREEN.partition_rows(panel, target, outer)
            frame = pd.read_csv(folder / "predictions.csv", keep_default_na=False, float_precision="round_trip")
            for family in models.MODEL_NAMES:
                cell = folder / family
                if not _SCREEN.completed(cell, {"stage": "fit", "fold": outer, "target": target, "model": family}):
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
        raise ValueError("The two-family screen requires 3,460 unique held-out predictions")
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
                    "At E1=(1-alpha)*E the refresh ratio is 1, so H_eff=H and E_eff=E; "
                    "both benefit and harm are native. Zero dose is explicitly native. "
                    "The head and link are unchanged. "
                    "Lambda=0 and the aggregate family delegate to the native design exactly."
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
    _SCREEN.finish(OUTPUT, {"stage": "summary"}, tuple(outputs))
    print(json.dumps({"summary": True, "heads": EXPECTED_HEADS, "predictions": len(predictions)}))


def gate() -> None:
    _SCREEN.gate()


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
