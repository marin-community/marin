# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP043 continuation: preserve fits and explicitly record overflowing CV losses."""

from __future__ import annotations

import argparse
import functools
import gzip
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_profile_20261015 as prior

BASE = Path(__file__).resolve().parent
ORIGINAL_SOURCE = BASE / "run_tpp40_whole_response_clean_20261015.py"
ORIGINAL_SHA = "e1f59a07bfe8bbf06a539024a8a53c261448c6b079132e06f3f2465a67ee7f8c"


def private_runner() -> Any:
    if prior.digest(ORIGINAL_SOURCE) != ORIGINAL_SHA:
        raise ValueError("The frozen v1 runner changed")
    spec = importlib.util.spec_from_file_location("_tp043_preserved_runner", ORIGINAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the preserved runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backend = private_runner()
ORIGINAL_OUTPUT = backend.OUTPUT
OUTPUT = prior.PARENT / "whole_response_clean_v2"
INTERRUPTED_SEAL = ORIGINAL_OUTPUT / "interrupted_manifest_v1.json"
INTERRUPTED_SHA = "361769f98d342050f9a0637fac846a33580a7abe80b45c0739032d271eb6dcca"
ADOPTION = OUTPUT / "adoption_manifest.json"
SCOPE = "tp043_ten_clean_outer0_whole_response_cv_ridge0_v2_numerical_continuation"
CONTRACT = dict(backend.CONTRACT) | {
    "scope": SCOPE,
    "continuation_of": "tp043_ten_clean_outer0_whole_response_cv_ridge0",
    "cv_numerical_repair": (
        "Keep the original finite pooled MSE exactly. After overflow only, use a scaled residual mean; "
        "unrepresentable MSE is ineligible with null score, finite predictions and reason retained."
    ),
    "adoption": (
        "Only sealed complete conditional/full fit payloads; verify original identity, exact spec/input "
        "and payload hash; write a new header and explicit adopted_from lineage, without changing fits."
    ),
    "interrupted_v1_sha256": INTERRUPTED_SHA,
}
_original_sources = backend.source_paths
_original_inputs = backend.input_paths
_original_identity = backend.identity
_original_conditional_fit = backend.conditional_fit


@functools.cache
def sealed_adoption_manifest() -> dict[str, Any]:
    """Inventory finalized records from the preserved, interrupted run."""
    if prior.digest(INTERRUPTED_SEAL) != INTERRUPTED_SHA:
        raise ValueError("The interrupted v1 seal changed")
    seal = prior.read(INTERRUPTED_SEAL)
    if seal["status"] != "interrupted_before_held_scoring" or seal["root"] != str(ORIGINAL_OUTPUT):
        raise ValueError("Wrong interrupted-run scope")
    prior.verify_hashes({str(ORIGINAL_OUTPUT / name): item["sha256"] for name, item in seal["files"].items()})
    old_identity = prior.digest(ORIGINAL_OUTPUT / "execution_identity.json")
    if old_identity != seal["execution_identity_sha256"]:
        raise ValueError("The interrupted run's execution identity changed")
    records = {}
    for name, item in sorted(seal["files"].items()):
        if not name.startswith("cells/") or not name.endswith(".json.gz"):
            continue
        path = ORIGINAL_OUTPUT / name
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record["header"]["identity_sha256"] != old_identity:
            raise ValueError("A sealed conditional fit has another identity")
        if record["fit_sha256"] != prior.canonical_hash(record["fit"]):
            raise ValueError("A sealed conditional fit has a corrupt payload")
        records[name] = {
            "path": str(path),
            "sha256": item["sha256"],
            "header": record["header"],
            "fit_sha256": record["fit_sha256"],
        }
    return {
        "scope": SCOPE,
        "interrupted_manifest_sha256": INTERRUPTED_SHA,
        "original_source_sha256": ORIGINAL_SHA,
        "original_execution_identity_sha256": old_identity,
        "records": records,
        "records_count": len(records),
        "family_metadata_adopted": False,
        "incomplete_files_adopted": False,
    }


@functools.cache
def adoption_manifest() -> dict[str, Any]:
    return prior.read(ADOPTION)


def source_paths() -> set[Path]:
    return _original_sources() | {
        Path(__file__),
        BASE / "test_tpp40_whole_response_clean_v2_20261015.py",
        ORIGINAL_OUTPUT / "proposal_v1.md",
    }


def input_paths() -> set[Path]:
    return _original_inputs() | {
        ADOPTION,
        INTERRUPTED_SEAL,
        ORIGINAL_OUTPUT / "registration.json",
        ORIGINAL_OUTPUT / "execution_identity.json",
        ORIGINAL_OUTPUT / "execution_approval.json",
    }


def identity() -> dict[str, Any]:
    frozen = _original_identity()
    if adoption_manifest() != sealed_adoption_manifest():
        raise ValueError("The explicit adoption inventory changed")
    return frozen


def adopted_record(
    entry: dict[str, Any], header: dict[str, Any], original_identity: str, adoption_sha: str
) -> dict[str, Any]:
    """Wrap an unchanged verified payload with the new identity and its lineage."""
    path = Path(entry["path"])
    if prior.digest(path) != entry["sha256"]:
        raise ValueError("The adopted source bytes changed")
    old = json.loads(gzip.decompress(path.read_bytes()))
    expected = dict(header) | {"identity_sha256": original_identity}
    if old["header"] != expected or old["header"] != entry["header"]:
        raise ValueError("Adoption requires identical original spec, inputs and identity")
    if old["fit_sha256"] != entry["fit_sha256"] or old["fit_sha256"] != prior.canonical_hash(old["fit"]):
        raise ValueError("The adopted fit payload changed")
    return {
        "header": header,
        "fit_sha256": old["fit_sha256"],
        "fit": old["fit"],
        "adopted_from": {
            "path": str(path),
            "sha256": entry["sha256"],
            "identity_sha256": original_identity,
            "fit_sha256": entry["fit_sha256"],
            "adoption_manifest_sha256": adoption_sha,
        },
    }


def conditional_fit(
    path: Path, matrix: np.ndarray, observed: np.ndarray, spec: dict[str, Any], identity_sha: str
) -> dict[str, Any]:
    header = {
        "spec": spec,
        "identity_sha256": identity_sha,
        "input_sha256": prior.canonical_hash({"matrix": matrix.tolist(), "observed": observed.tolist()}),
    }
    manifest = adoption_manifest()
    entry = manifest["records"].get(str(path.relative_to(OUTPUT)))
    if path.exists():
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record["header"] != header or record["fit_sha256"] != prior.canonical_hash(record["fit"]):
            raise ValueError("Conditional checkpoint identity or contents differ")
        if entry is not None:
            expected = adopted_record(
                entry, header, manifest["original_execution_identity_sha256"], prior.digest(ADOPTION)
            )
            if record != expected:
                raise ValueError("Adopted checkpoint lineage or payload differs")
        elif "adopted_from" in record:
            raise ValueError("Unregistered adoption lineage")
        return record["fit"]
    if path.with_name(path.name + ".writing").exists():
        raise ValueError(f"Preserve interrupted checkpoint before adoption or fitting {path}")
    if entry is None:
        return _original_conditional_fit(path, matrix, observed, spec, identity_sha)
    record = adopted_record(entry, header, manifest["original_execution_identity_sha256"], prior.digest(ADOPTION))
    backend.require_disk_space(path)
    backend.atomic_bytes(path, gzip.compress(json.dumps(record, sort_keys=True, allow_nan=False).encode(), mtime=0))
    return record["fit"]


def finite_cv_mse(predicted: np.ndarray, observed: np.ndarray) -> tuple[float | None, str]:
    """Preserve ordinary scoring; recover representable means after intermediate overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        residual = predicted - observed
        ordinary = float(np.mean(residual**2))
    if np.isfinite(ordinary):
        return ordinary, "original_finite_arithmetic"
    if not np.isfinite(residual).all():
        return None, "nonfinite_cv_residual"
    maximum = float(np.max(np.abs(residual)))
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_mean = float(np.mean((residual / maximum) ** 2))
        scaled = float((maximum * normalized_mean) * maximum)
    if np.isfinite(scaled):
        return scaled, "scaled_mean_after_intermediate_overflow"
    return None, "cv_mean_squared_error_exceeds_float64"


def profile_candidate(
    matrix: np.ndarray, observed: np.ndarray, labels: np.ndarray, folder: Path, spec: dict[str, Any], identity_sha: str
) -> dict[str, Any]:
    predictions = np.empty(len(observed))
    records, failures = [], []
    for label in np.unique(labels):
        train, held = np.flatnonzero(labels != label), np.flatnonzero(labels == label)
        path = folder / f"inner{label}.json.gz"
        fit = conditional_fit(path, matrix[train], observed[train], spec | {"inner_fold": int(label)}, identity_sha)
        record = {
            "label": int(label),
            "path": str(path),
            "sha256": prior.digest(path),
            "train_indices": train.tolist(),
            "held_indices": held.tolist(),
            "unresolved": fit["unresolved"],
            "diagnostics": prior.fit_diagnostics(fit),
        }
        try:
            if fit["selected"] is None:
                raise ArithmeticError("No finite conditional candidate")
            predictions[held] = backend.solver.prior.predict(matrix[held], fit["selected"])
        except ArithmeticError as error:
            record["prediction_failure"] = str(error)
            failures.append(int(label))
        records.append(record)
    score, status = (None, "conditional_prediction_failure") if failures else finite_cv_mse(predictions, observed)
    return {
        "spec": spec,
        "eligible_for_selection": score is not None,
        "inner_mse": score,
        "inner_predictions": None if failures else predictions.tolist(),
        "conditional_records": records,
        "prediction_failure_folds": failures,
        "cv_numerical_status": status,
        "cv_loss_failure": None if score is not None else status,
        "any_unresolved_inner": any(record["unresolved"] for record in records),
    }


backend.OUTPUT = OUTPUT
backend.SCOPE = SCOPE
backend.CONTRACT = CONTRACT
backend.source_paths = source_paths
backend.input_paths = input_paths
backend.identity = identity
backend.conditional_fit = conditional_fit
backend.profile_candidate = profile_candidate


def fit_world(name: str) -> None:
    backend.fit_world(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "freeze", "fit", "score"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "inventory":
        backend.save_json(ADOPTION, sealed_adoption_manifest())
    elif args.command == "freeze":
        backend.save_json(OUTPUT / "execution_identity.json", identity())
    elif args.command == "fit":
        backend.execution_identity()
        if args.workers < 1:
            raise ValueError("A positive local worker count is required")
        names = [world["name"] for world in backend.WORLDS]
        if args.workers == 1:
            for name in names:
                fit_world(name)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(fit_world, names))
    else:
        backend.score_all()


if __name__ == "__main__":
    main()
