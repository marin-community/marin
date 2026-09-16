# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fsspec==2026.1.0", "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Fit registered fixed-checkpoint models on frozen training branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from fixed_checkpoint_hellinger_baselines_20260907 import fit_baseline, predict_baseline
from fixed_checkpoint_wspu_models_20260907 import Variant, fit_model, predict_model, prediction_audit

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "reference_outputs/fixed_checkpoint_branch_wspu_20260907"
LOCAL_PREFIXES = (
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)
VARIANTS = {
    "BRW-002": Variant.CUMULATIVE_LOG,
    "BRW-003": Variant.CUMULATIVE_BPB,
    "BRW-004": Variant.CONTINUATION_LOG,
    "BRW-005": Variant.CUMULATIVE_LOG,
}


@dataclass(frozen=True)
class Cohort:
    name: str
    actions: list[int]
    calibration: list[int]
    tests: dict[str, list[int]]
    test_calibration: list[int]


@dataclass(frozen=True)
class Job:
    cohort: str
    model: str
    component: int


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(output: Path) -> dict[str, str]:
    paths = [
        output / "data/manifest.json",
        output / "data/arrays.npz",
        output / "data/rows.csv",
        output / "data/audit_frame.csv",
        output / "PROTOCOL.md",
        output / "sources/mixture_selection.py",
        output / "sources/audit_delphi_phase1_branch_response_20260826.py",
        BASE / "fixed_checkpoint_wspu_models_20260907.py",
        BASE / "fixed_checkpoint_hellinger_baselines_20260907.py",
        BASE / "fit_two_phase_link_spines_20260907.py",
        Path(__file__),
    ]
    return {str(path): digest(path) for path in paths}


def selected_rows(frame: pd.DataFrame, mask: pd.Series) -> list[int]:
    return frame.loc[mask, "row"].astype(int).tolist()


def cohort_identity(cohort: Cohort, frame: pd.DataFrame) -> dict[str, object]:
    indices = cohort.actions + cohort.calibration + cohort.test_calibration
    indices += [index for test in cohort.tests.values() for index in test]
    cell = frame.iloc[indices]
    for field in ("state_id", "prefix", "prefix_repeat_seed", "trainer_seed"):
        assert cell[field].nunique(dropna=False) == 1, (cohort.name, field)
    uris = cell.prefix_checkpoint_uri.dropna().unique().tolist()
    assert len(uris) <= 1, (cohort.name, uris)
    return {
        "state_id": str(cell.state_id.iloc[0]),
        "prefix_repeat_seed": int(cell.prefix_repeat_seed.iloc[0]),
        "prefix_checkpoint_uri": uris[0] if uris else None,
        "provenance_sources": sorted(cell.checkpoint_identity_source.unique().tolist()),
        "exact_archived_prefix_uri_available": bool(uris),
    }


def cohorts(frame: pd.DataFrame) -> list[Cohort]:
    primary = frame.panel.eq("proportional")
    proportional_state = frame.loc[frame.benchmark_role.eq("primary_train"), "state_id"].unique()
    assert len(proportional_state) == 1
    primary &= frame.state_id.eq(proportional_state[0])
    broad = frame.panel.eq("cap10_kl0p05_broad")
    local = frame.panel.eq("cap10_kl0p05_local")
    broad_state = frame.loc[broad & frame.benchmark_role.eq("broad_train"), "state_id"].unique()
    assert len(broad_state) == 1
    broad &= frame.state_id.eq(broad_state[0])
    local &= frame.state_id.eq(broad_state[0])
    result = [
        Cohort(
            "proportional",
            selected_rows(frame, primary & frame.benchmark_role.eq("primary_train")),
            selected_rows(frame, primary & frame.is_tied_control),
            {
                "coverage": selected_rows(frame, primary & frame.benchmark_role.eq("primary_test")),
                "adaptive_descriptive": selected_rows(frame, primary & frame.benchmark_role.eq("adaptive_descriptive")),
            },
            [],
        ),
        Cohort(
            "cap10_local",
            selected_rows(frame, broad & frame.benchmark_role.eq("broad_train")),
            selected_rows(frame, broad & frame.is_tied_control),
            {"local80": selected_rows(frame, local & frame.benchmark_role.eq("local_test"))},
            selected_rows(frame, local & frame.is_tied_control),
        ),
    ]
    for state in LOCAL_PREFIXES:
        broad = frame.panel.eq("crossed_broad") & frame.state_id.eq(state)
        local = frame.panel.eq("crossed_local") & frame.state_id.eq(state)
        result.append(
            Cohort(
                state,
                selected_rows(frame, broad & frame.fit_budget),
                selected_rows(frame, broad & frame.is_tied_control),
                {"local10": selected_rows(frame, local & frame.fit_budget)},
                selected_rows(frame, local & frame.is_tied_control),
            )
        )
    assert len(result[0].actions) == 80 and len(result[0].tests["coverage"]) == 40
    assert len(result[1].actions) == 100 and len(result[1].tests["local80"]) == 80
    for cohort in result[2:]:
        assert len(cohort.actions) == 50 and len(cohort.tests["local10"]) == 10, cohort.name
    for cohort in result:
        assert cohort.calibration, cohort.name
        training = set(cohort.actions + cohort.calibration)
        assert len(training) == len(cohort.actions) + len(cohort.calibration)
        train_hashes = set(frame.iloc[cohort.actions].coordinate_hash)
        for label, test in cohort.tests.items():
            assert not training.intersection(test), (cohort.name, label)
            assert not train_hashes.intersection(frame.iloc[test].coordinate_hash), (cohort.name, label)
    return result


def cohort_folds(cohort: Cohort, legacy: pd.DataFrame, source: Path) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    historical = load_module(source, "branch_historical_folds")
    labels = historical.geometric_folds(legacy.iloc[cohort.actions])
    count = len(cohort.actions) + len(cohort.calibration)
    full_labels = np.full(count, -1, int)
    full_labels[: len(labels)] = labels
    return tuple((np.flatnonzero(full_labels != fold), np.flatnonzero(full_labels == fold)) for fold in range(5))


def fit_job(job: Job, output_path: str) -> dict[str, object]:
    output = Path(output_path)
    identity = inputs(output)
    metadata = pd.read_csv(output / "data/rows.csv")
    legacy = pd.read_csv(output / "data/audit_frame.csv")
    assert np.array_equal(metadata.row.to_numpy(), np.arange(len(metadata)))
    assert np.array_equal(legacy.row.to_numpy(), metadata.row.to_numpy())
    assert legacy.row_id.equals(metadata.row_id)
    arrays = np.load(output / "data/arrays.npz")
    cohort = next(cohort for cohort in cohorts(metadata) if cohort.name == job.cohort)
    prefix_identity = cohort_identity(cohort, metadata)
    training = np.asarray(cohort.actions + cohort.calibration, int)
    calibration = np.arange(len(training)) >= len(cohort.actions)
    test_rows = sorted({row for rows in cohort.tests.values() for row in rows} | set(cohort.test_calibration))
    prediction_rows = np.asarray(list(dict.fromkeys(training.tolist() + test_rows)), int)
    prefix = arrays["phase0_epochs"][training[0]]
    assert np.allclose(arrays["phase0_epochs"][training], prefix[None], rtol=1e-10, atol=1e-10)
    assert np.allclose(arrays["phase0_epochs"][prediction_rows], prefix[None], rtol=1e-10, atol=1e-10)
    source = output / "sources/mixture_selection.py"
    baseline_source = output / "sources/audit_delphi_phase1_branch_response_20260826.py"
    folds = cohort_folds(cohort, legacy, baseline_source)
    component_name = "macro" if job.component < 0 else str(arrays["component_names"][job.component])
    destination = output / "fits" / cohort.name / job.model / f"{component_name}.json"
    if destination.exists():
        stored = json.loads(destination.read_text())
        assert stored["input_hashes"] == identity and stored["job"] == asdict(job)
        return {"job": asdict(job), "status": "cached", "elapsed": stored["elapsed"]}
    started = time.monotonic()
    response = arrays["target"] if job.component < 0 else arrays["component_bpb"][:, job.component]
    y = response[training]
    assert np.isfinite(y).all()
    sd = float(np.std(y[calibration], ddof=1)) if calibration.sum() > 1 else 0.0
    if job.model in {"BRW-000", "BRW-001"}:
        center = arrays["phase0_weight"][training[0]]
        fitted = fit_baseline(
            arrays["phase1_weight"][training], center, y, calibration, folds, job.model, baseline_source
        )
        prediction = predict_baseline(fitted, arrays["phase1_weight"][prediction_rows], center, baseline_source)
        tied_prediction = float(predict_baseline(fitted, center[None], center, baseline_source)[0])
        audit = {"response_link": "additive", "finite": bool(np.isfinite(prediction).all())}
    else:
        fitted = fit_model(
            prefix,
            arrays["phase1_epochs"][training],
            y,
            calibration,
            folds,
            variant=VARIANTS[job.model],
            source_path=source,
            noise_sd=sd,
        )
        prediction = predict_model(fitted, arrays["phase1_epochs"][prediction_rows], source_path=source)
        tied_prediction = float(
            predict_model(fitted, arrays["tied_phase1_epochs"][[training[0]]], source_path=source)[0]
        )
        audit = prediction_audit(fitted, arrays["phase1_epochs"][prediction_rows], source_path=source)
    assert np.isfinite(prediction).all() and np.isfinite(tied_prediction)
    floor = fitted.get("floor")
    stored = {
        "job": asdict(job),
        "component_name": component_name,
        "prefix_identity": prefix_identity,
        "input_hashes": identity,
        "training_rows": training.tolist(),
        "calibration_rows": training[calibration].tolist(),
        "test_rows": cohort.tests,
        "test_calibration_rows": cohort.test_calibration,
        "inner_folds": [[training[a].tolist(), training[b].tolist()] for a, b in folds],
        "fit": fitted,
        "prediction_rows": prediction_rows.tolist(),
        "predicted": prediction.tolist(),
        "predicted_tied": tied_prediction,
        "prediction_audit": audit,
        "training_anchor": float(np.mean(y[calibration])),
        "noise_sd": sd,
        "noise_sd_available": bool(calibration.sum() > 1),
        "held_outcomes_below_floor": None if floor is None else int(np.sum(response[test_rows] < float(floor))),
        "elapsed": time.monotonic() - started,
    }
    write_json_atomic(destination, stored)
    return {"job": asdict(job), "status": "fit", "elapsed": stored["elapsed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=[f"BRW-{i:03d}" for i in range(6)])
    parser.add_argument("--cohorts", nargs="+")
    args = parser.parse_args()
    frame = pd.read_csv(args.output / "data/rows.csv")
    arrays = np.load(args.output / "data/arrays.npz")
    defined = cohorts(frame)
    for cohort in defined:
        cohort_identity(cohort, frame)
    write_json_atomic(args.output / "cohorts.json", [asdict(cohort) for cohort in defined])
    jobs = []
    for cohort in defined:
        if args.cohorts and cohort.name not in args.cohorts:
            continue
        for model in args.models:
            if model not in {"BRW-000", "BRW-001", *VARIANTS}:
                raise ValueError(model)
            if model == "BRW-005":
                complete = np.isfinite(arrays["component_bpb"][cohort.actions + cohort.calibration]).all()
                if complete:
                    jobs.extend(Job(cohort.name, model, component) for component in range(7))
            else:
                jobs.append(Job(cohort.name, model, -1))
    assert jobs
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fit_job, job, str(args.output.resolve())): job for job in jobs}
        for i, future in enumerate(as_completed(pending), start=1):
            try:
                result = future.result()
            except Exception:
                failed = pending[future]
                write_json_atomic(
                    args.output / "failures" / f"{failed.cohort}_{failed.model}_{failed.component}.json",
                    {"job": asdict(failed), "input_hashes": inputs(args.output), "traceback": traceback.format_exc()},
                )
                raise
            print(json.dumps({"completed": i, "total": len(jobs), **result}), flush=True)


if __name__ == "__main__":
    main()
