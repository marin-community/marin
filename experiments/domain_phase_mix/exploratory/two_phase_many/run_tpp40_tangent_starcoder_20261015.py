# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Apply unchanged round-five J2 estimators to the frozen StarCoder quick screen.

Preparation hashes outcomes without parsing them. Only an explicitly invoked
fit stage opens target values. This screen cannot override synthetic failures
or establish full qualification, transfer, or prospective performance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_models_20261015 as models

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_tangent_starcoder_20261015"
FROZEN = benchmark.DEFAULT_OUTPUT / "v3"
PREPARED = FROZEN / "prepared"
PANEL = "starcoder_wsd80_dense"
REFERENCES = ("effective_exposure_dsp", "aggregate_mariner_train_median")
PREFLIGHT = ROOT / "reference_outputs/tpp40_tangent_round5_20261015/transfer_preflight.json"
ROUND5_IDENTITY = PREFLIGHT.parent / "execution_identity.json"
EVIDENCE = ("candidate_registration", "asymmetric_only_fitting", "nested_split_integrity", "tied_class_mapping")


def selected_targets() -> tuple[str, ...]:
    metadata = json.loads((PREPARED / PANEL / "metadata.json").read_text())
    return tuple(
        sorted(
            target
            for target in metadata["targets"]
            if any(benchmark.target_matches(target, selector) for selector in benchmark.QUICK_TASKS)
        )
    )


def source_hashes() -> dict[str, str]:
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    metadata = json.loads((PREPARED / PANEL / "metadata.json").read_text())
    paths = {
        Path(__file__).resolve(),
        Path(models.__file__),
        Path(models.frozen.__file__),
        Path(models.frozen.mechanisms.__file__),
        Path(models.frozen.temporal.__file__),
        OUTPUT / "registration.json",
        OUTPUT / "adoption.json",
        PREFLIGHT,
        ROUND5_IDENTITY,
        PREPARED / "manifest.json",
        benchmark.PROTOCOL_DIR / "protocol.json",
        benchmark.PROTOCOL_DIR / "data_contract.json",
        *(Path(path) for path in manifest["identity"]["code"]),
        *(Path(path) for path in metadata["source_hashes"]),
        *(
            PREPARED / PANEL / name
            for name in ("policies.npz", "rows.csv", "outcomes.csv", "folds.csv", "metadata.json", "split_audit.json")
        ),
        *(FROZEN / "screen_policies" / f"{reference}.json" for reference in REFERENCES),
        *(
            FROZEN / "screen_calibration" / name
            for name in ("run.json", "predictions.csv", "metrics.csv", "summary.csv")
        ),
    }
    return {str(path.resolve()): benchmark.sha256(path) for path in sorted(paths)}


def identity() -> dict[str, Any]:
    current = {"sources": source_hashes(), "environment": benchmark.environment_identity()}
    frozen = json.loads((OUTPUT / "execution_identity.json").read_text())
    if current != frozen:
        raise ValueError("Source/data/runtime drift; preserve outputs and register a new execution")
    return frozen


def verify_registration(registration: dict[str, Any]) -> None:
    """Reject adoption of sources or screen settings outside the registered procedure."""
    expected_fields = {
        "families": list(models.MODEL_NAMES),
        "models": list(models.MODEL_NAMES),
        "panel": PANEL,
        "targets": list(selected_targets()),
        "outer_folds": list(range(5)),
        "repeat": 0,
        "expected_contexts": 25,
        "expected_heads": 75,
        "expected_prediction_rows_per_model": 1730,
        "fit_strengths": list(models.STRENGTHS),
        "structural_tolerance_bpb": 1e-8,
    }
    for key, expected in expected_fields.items():
        if registration[key] != expected:
            raise ValueError(f"Registered screen setting differs: {key}")
    expected_receipts = {
        "models_source": Path(models.__file__),
        "estimator_source": Path(models.frozen.__file__),
        "prepared_manifest": PREPARED / "manifest.json",
        "data_contract": benchmark.PROTOCOL_DIR / "data_contract.json",
        "protocol": benchmark.PROTOCOL_DIR / "protocol.json",
    }
    for key, path in expected_receipts.items():
        if registration[key] != benchmark.receipt(path):
            raise ValueError(f"Registered source receipt differs: {key}")
    round5_sources = json.loads(ROUND5_IDENTITY.read_text())
    for module in (models, models.frozen, models.frozen.mechanisms, models.frozen.temporal, models.mariner):
        path = Path(module.__file__).resolve()
        if benchmark.sha256(path) != round5_sources[str(path)]:
            raise ValueError(f"Round-five estimator dependency changed: {path}")
    for filename, receipt in registration["data_files"].items():
        if receipt != benchmark.receipt(PREPARED / PANEL / filename):
            raise ValueError(f"Registered data receipt differs: {filename}")
    if set(registration["references"]) != set(REFERENCES):
        raise ValueError("Registered reference models differ")
    for reference in REFERENCES:
        record = registration["references"][reference]
        if record["policy"] != benchmark.receipt(FROZEN / "screen_policies" / f"{reference}.json"):
            raise ValueError(f"Registered reference policy differs: {reference}")
        if record["reference_predictions"] != benchmark.receipt(FROZEN / "screen_calibration" / "predictions.csv"):
            raise ValueError(f"Registered reference predictions differ: {reference}")


def prepare() -> None:
    """Adopt immutable v3 data with explicit, limited packaging-hash drift."""
    registration = json.loads((OUTPUT / "registration.json").read_text())
    verify_registration(registration)
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
        raise ValueError("Use the recorded v3 numerical environment; `uv run --no-sync python` currently matches")
    if benchmark.sha256(benchmark.PROTOCOL_DIR / "data_contract.json") != manifest["identity"]["data_contract_sha256"]:
        raise ValueError("The frozen data contract changed")
    for reference in REFERENCES:
        policy = benchmark.verified_policy(FROZEN / "screen_policies" / f"{reference}.json", None, "preparation")
        if not policy["tolerances_frozen"]:
            raise ValueError(f"Frozen reference/tolerance receipts failed: {reference}")
    targets = selected_targets()
    if len(targets) != 5:
        raise ValueError("Expected the five frozen quick-screen targets")
    metadata = json.loads((PREPARED / PANEL / "metadata.json").read_text())
    with np.load(PREPARED / PANEL / "policies.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["phase0_epochs"], arrays["phase0"] * arrays["c0"])
        np.testing.assert_array_equal(arrays["phase1_epochs"], arrays["phase1"] * arrays["c1"])
        np.testing.assert_allclose(arrays["c0"] / (arrays["c0"] + arrays["c1"]), metadata["alpha"], rtol=1e-12, atol=0)
    adoption = {
        "scope": "Reuse immutable v3 scientific inputs and exact runtime; explicitly adopt current packaging hashes.",
        "prepared_manifest": benchmark.receipt(PREPARED / "manifest.json"),
        "transfer_preflight": benchmark.receipt(PREFLIGHT),
        "registration": registration,
        "scientific_sources": scientific,
        "packaging_hashes": packaging,
        "environment": environment,
        "panel": PANEL,
        "targets": targets,
        "models": models.MODEL_NAMES,
        "repeat": 0,
        "folds": list(range(5)),
        "heads": 75,
        "outcome_values_parsed": False,
    }
    # Normalize tuples exactly as the saved JSON will represent them.
    adoption = json.loads(json.dumps(adoption))
    destination = OUTPUT / "adoption.json"
    if destination.exists() and json.loads(destination.read_text()) != adoption:
        raise ValueError("Existing adoption differs; preserve it and use a new registered artifact root")
    benchmark.write_json(destination, adoption)
    current = {"sources": source_hashes(), "environment": environment}
    frozen_path = OUTPUT / "execution_identity.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != current:
        raise ValueError("Existing execution identity differs; preserve it and use a new registered artifact root")
    benchmark.write_json(frozen_path, current)
    print(json.dumps({"prepared": True, "targets": targets, "heads": 75}))


def completed(folder: Path, context: dict[str, Any]) -> bool:
    path = folder / "complete.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record["identity"] != identity() or record["context"] != context:
        raise ValueError(f"Stale completion: {folder}")
    benchmark.verify_hashes(record["outputs"], folder)
    for family, digest in record.get("model_receipts", {}).items():
        if benchmark.sha256(folder / family / "complete.json") != digest:
            raise ValueError(f"Model receipt changed after prediction: {folder / family}")
    return True


def finish(folder: Path, context: dict[str, Any], files: tuple[str, ...], **extra: Any) -> None:
    benchmark.write_json(
        folder / "complete.json",
        {
            "identity": identity(),
            "context": context,
            "outputs": {name: benchmark.sha256(folder / name) for name in files},
            **extra,
        },
    )


def partition_rows(
    panel: benchmark.data.BenchPanel, target: str, outer: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    table = pd.read_csv(PREPARED / PANEL / "folds.csv", dtype={"row_id": str})
    table = table[(table.repeat == 0) & (table.outer_fold == outer)]
    index = pd.Series(np.arange(len(panel.rows)), index=panel.rows.row_id)
    table = table.loc[np.isfinite(panel.targets[target][index.loc[table.row_id].to_numpy(int)])]
    training, query = table[table.role.eq("train")], table[table.role.ne("train")]
    train, held = index.loc[training.row_id].to_numpy(int), index.loc[query.row_id].to_numpy(int)
    if panel.rows.iloc[train].tied.any() or np.any(training.inner_fold.to_numpy(int) < 0):
        raise ValueError("The eligible fitting population includes tied rows or invalid inner labels")
    return training, query, train, held


def fit_partition(
    panel: benchmark.data.BenchPanel,
    target: str,
    train: np.ndarray,
    inner: np.ndarray,
    family: str,
    matched_null_fit: models.JointFit | None,
    *,
    test_shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: models.frozen.TestGrid | None = None,
) -> models.JointFit:
    """Pass only eligible labels to unchanged J2; optional menus are behavior-test-only."""
    return models.fit_joint(
        panel.phase0_epochs[train],
        panel.phase1_epochs[train],
        panel.targets[target][train],
        inner,
        panel.alpha,
        family,
        matched_null_fit=matched_null_fit,
        shapes=test_shapes,
        test_grid=test_grid,
    )


def fit_context(task: tuple[int, int]) -> dict[str, Any]:
    outer, task_index = task
    target = selected_targets()[task_index]
    folder = OUTPUT / f"fits/fold{outer}_task{task_index}"
    context = {"stage": "predictions", "fold": outer, "target": target}
    if completed(folder, context):
        return {"fold": outer, "target": target, "cached": True}
    identity()
    panel = benchmark.read_panel(PREPARED, PANEL)
    training, query, train, held = partition_rows(panel, target, outer)
    frames, receipts, aggregate = [], {}, None
    for family in models.MODEL_NAMES:
        cell = folder / family
        cell_context = {"stage": "fit", "fold": outer, "target": target, "model": family}
        if completed(cell, cell_context):
            fitted = models.JointFit.from_json(json.loads((cell / "fit.json").read_text()))
        else:
            identity()
            started = time.monotonic()
            fitted = fit_partition(panel, target, train, training.inner_fold.to_numpy(int), family, aggregate)
            benchmark.write_json(
                cell / "fit.json",
                fitted.to_json()
                | {
                    "training_rows": training.row_id.tolist(),
                    "inner_folds": training.inner_fold.tolist(),
                    "runtime_seconds": time.monotonic() - started,
                },
            )
            finish(cell, cell_context, ("fit.json",))
        if family == "aggregate_mariner":
            aggregate = fitted
        predicted = fitted.predict(panel.phase0_epochs[held], panel.phase1_epochs[held])
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite held-out predictions")
        frame = panel.rows.iloc[held][["row_id", "policy_id", "group_id", "source", "region"]].copy()
        frame["model"], frame["panel"], frame["target"] = family, PANEL, target
        frame["repeat"], frame["outer_fold"], frame["role"] = 0, outer, query.role.to_numpy()
        frame["observed"], frame["predicted"] = panel.targets[target][held], predicted
        frames.append(frame)
        receipts[family] = benchmark.sha256(cell / "complete.json")
    pd.concat(frames, ignore_index=True).to_csv(folder / "predictions.csv", index=False)
    finish(folder, context, ("predictions.csv",), model_receipts=receipts)
    return {"fold": outer, "target": target, "cached": False}


def summarize() -> None:
    """Replay frozen splits and structural mappings before emitting screen evidence."""
    identity()
    panel = benchmark.read_panel(PREPARED, PANEL)
    frames, receipts, mapping_errors = (
        [],
        {family: [] for family in models.MODEL_NAMES},
        {family: [] for family in models.MODEL_NAMES},
    )
    probe = np.column_stack([np.linspace(0, 1, 257), np.linspace(1, 0, 257)])
    e0, e1 = probe * panel.c0, probe * panel.c1
    for outer in range(5):
        for task_index, target in enumerate(selected_targets()):
            folder = OUTPUT / f"fits/fold{outer}_task{task_index}"
            if not completed(folder, {"stage": "predictions", "fold": outer, "target": target}):
                raise ValueError(f"Incomplete screen context: {folder}")
            training, query, _train, held = partition_rows(panel, target, outer)
            frame = pd.read_csv(folder / "predictions.csv", keep_default_na=False, float_precision="round_trip")
            for family in models.MODEL_NAMES:
                cell = folder / family
                if not completed(cell, {"stage": "fit", "fold": outer, "target": target, "model": family}):
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
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True).sort_values(["model", "target", "outer_fold", "row_id"])
    if len(predictions) != 5190 or predictions.duplicated(["model", "target", "row_id"]).any():
        raise ValueError("The complete screen must have 1,730 unique predictions per model")
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    benchmark.scoring.score_predictions(predictions).to_csv(OUTPUT / "metrics.csv", index=False)
    benchmark.scoring.summarize_predictions(predictions).to_csv(OUTPUT / "summary.csv", index=False)
    policy = json.loads((FROZEN / "screen_policies" / "effective_exposure_dsp.json").read_text())
    outputs = ["predictions.csv", "metrics.csv", "summary.csv"]
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
                    "At E0=alpha*(E0+E1), both departures vanish and B=b; harm, link and coefficients are native."
                )
                record["maximum_probe_error_bpb"] = max(mapping_errors[family])
            if name in ("asymmetric_only_fitting", "nested_split_integrity"):
                record["verification"] = (
                    "All 25 recorded training memberships and inner labels equal frozen folds; tied rows excluded. "
                    "Serialized held-out predictions replay exactly. Training API receives only training labels."
                )
            path = destination / f"{name}.json"
            benchmark.write_json(path, record)
            evidence_receipts[name] = [benchmark.receipt(path)]
            outputs.append(str(path.relative_to(OUTPUT)))
        benchmark.write_json(destination / "evidence.json", evidence_receipts)
        outputs.extend(str((destination / name).relative_to(OUTPUT)) for name in ("predictions.csv", "evidence.json"))
    finish(OUTPUT, {"stage": "summary"}, tuple(outputs))
    print(json.dumps({"summary": True, "heads": 75, "predictions": len(predictions)}))


def gate() -> None:
    if not completed(OUTPUT, {"stage": "summary"}):
        raise ValueError("A verified complete summary is required")
    for family in models.MODEL_NAMES:
        folder = OUTPUT / "candidates" / family
        for reference in REFERENCES:
            policy_path = FROZEN / "screen_policies" / f"{reference}.json"
            evidence_path, candidate_path = folder / "evidence.json", folder / "predictions.csv"
            reference_path = FROZEN / "screen_calibration/predictions.csv"
            policy = benchmark.verified_policy(policy_path, evidence_path, family)
            paired = benchmark.promotion_inputs(candidate_path, reference_path, policy, reference)
            policy["candidates"] = [family]
            verdict = benchmark.scoring.compare_models(paired, reference, policy)
            verdict["receipts"] = {
                "policy": benchmark.receipt(policy_path),
                "predictions": benchmark.receipt(candidate_path),
                "reference_predictions": benchmark.receipt(reference_path),
                "evidence": policy["evidence_receipts"],
            }
            verdict["scope"] = (
                "Frozen empirical screen only; does not supersede synthetic failures or establish promotion."
            )
            benchmark.write_json(folder / f"gate_{reference}.json", verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "summarize", "gate"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "fit":
        identity()
        tasks = [(fold, task) for fold in range(5) for task in range(5)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed(pool.submit(fit_context, task) for task in tasks):
                print(json.dumps(future.result()), flush=True)
    elif args.stage == "summarize":
        summarize()
    else:
        gate()


if __name__ == "__main__":
    main()
