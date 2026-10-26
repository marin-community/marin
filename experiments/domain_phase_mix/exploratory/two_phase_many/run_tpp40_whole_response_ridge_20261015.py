# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP049: train-only ridge/shape/kinetic selection on the frozen noisy stress fold."""

from __future__ import annotations

import argparse
import functools
import gzip
import importlib.util
import json
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_profile_20261015 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_response_ridge_20261015 as model

BASE = Path(__file__).resolve().parent
FROZEN_SOURCE = BASE / "run_tpp40_whole_response_fold3_noise_20261015.py"
FROZEN_SHA = "46e6846c7c392c8683ff9b80bab7ef80dd7233d9f87a18562f56aac9781fb552"
OLD_OUTPUT = prior.PARENT / "whole_response_fold3_noise"
OLD_SEAL = OLD_OUTPUT / "result_manifest_v1.json"
OLD_SEAL_SHA = "24466a9ea9d9e014279edd03801a163424f97dca9f06d8f1b201af53d53eb36e"
OUTPUT = prior.PARENT / "whole_response_ridge_stress"
SCOPE = "tp049_outer3_paired_noise_ridge_shape_kinetic_inner_cv"
RIDGES = (0.0, 1e-4, 0.01, 1.0, 100.0)
DEADLINE = "2026-10-15T18:00:00+00:00"


def private_runner() -> Any:
    if prior.digest(FROZEN_SOURCE) != FROZEN_SHA:
        raise ValueError("The frozen TP046 runner changed")
    spec = importlib.util.spec_from_file_location("_tp049_private_noisy_runner", FROZEN_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen noisy runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = private_runner()
backend, stage = old.backend, old.stage
CONTRACT = dict(old.CONTRACT) | {
    "candidate_id": "TP40-049",
    "scope": SCOPE,
    "ridge": list(RIDGES),
    "objective": "mean_squared_bpb_divided_by_0.003_squared_plus_lambda_times_canonical_amplitude_norm_squared",
    "selection": "pooled held-inner BPB MSE, unpenalized; ties by lambda index, strength index, native shape index",
    "native_reuse": "all five-lambda native inner profiles reused at mu0; selected complete native head reused",
    "maximum_conditional_fits": 302430,
    "maximum_new_conditional_fits": 241950,
    "maximum_nonlinear_starts": 1209750,
    "new_inner_fits": 241920,
    "maximum_new_full_fits": 30,
    "adopted_zero_inner_fits": 60480,
    "zero_ridge_adoption": "verified sealed TP046 exact spec/input/fit payload; no zero-ridge refitting",
    "checkpoint": (
        "selected head, all five solve records, each start-best trial, all numerical-failure trials; "
        "counts/hash of omitted finite trials; full final fits retain all trials"
    ),
    "compact_audit_scope": "retained records and omission commitments only; no claim of all-trial numerical replay",
    "profile_storage": "compact JSON; same OOF predictions and fold indices retained",
    "deadline_utc": DEADLINE,
    "resource_stop": (
        "before next fresh solve or adoption: 4GiB free and deadline; preserve fixed-schedule partial results"
    ),
    "clean_qualification": "not performed; separate future registration required",
    "held_ridge_curves_used_for_choice": False,
}
_old_sources, _old_inputs, _old_identity = old.source_paths, old.input_paths, old.identity
_save_json = backend.save_json


def source_paths() -> set[Path]:
    return _old_sources() | {
        FROZEN_SOURCE,
        Path(__file__),
        Path(model.__file__),
        BASE / "test_tpp40_whole_response_ridge_20261015.py",
        BASE / "test_tpp40_response_ridge_20261015.py",
        OLD_OUTPUT / "proposal_v1.md",
        prior.PARENT / "response_ridge_isolation/audit/independent_source_review_v1.json",
        OUTPUT / "proposal_v1.md",
        OUTPUT / "resource_estimate_v1.json",
    }


def input_paths() -> set[Path]:
    return _old_inputs() | {
        OLD_SEAL,
        OLD_OUTPUT / "execution_identity.json",
        OLD_OUTPUT / "registration.json",
        OLD_OUTPUT / "complete.json",
        OLD_OUTPUT / "audit/independent_result_review_v1.json",
        OUTPUT / "adoption_manifest.json",
    }


def identity() -> dict[str, Any]:
    frozen = _old_identity()
    if prior.digest(OLD_SEAL) != OLD_SEAL_SHA:
        raise ValueError("The sealed zero-ridge experiment changed")
    adopted = adoption_manifest()
    if adopted["old_seal_sha256"] != OLD_SEAL_SHA or adopted["scope"] != SCOPE:
        raise ValueError("Wrong adoption manifest")
    return frozen


@functools.cache
def adoption_manifest() -> dict[str, Any]:
    return prior.read(OUTPUT / "adoption_manifest.json")


def prepare_adoption() -> None:
    """Verify the complete prior seal without opening a training or query table."""
    if prior.digest(OLD_SEAL) != OLD_SEAL_SHA:
        raise ValueError("The sealed zero-ridge experiment changed")
    seal = prior.read(OLD_SEAL)
    prior.verify_hashes({str(OLD_OUTPUT / name): entry["sha256"] for name, entry in seal["files"].items()})
    records = {
        name: entry["sha256"]
        for name, entry in seal["files"].items()
        if name.startswith("cells/") and name.endswith(".json.gz")
    }
    _save_json(
        OUTPUT / "adoption_manifest.json",
        {
            "scope": SCOPE,
            "old_seal_sha256": OLD_SEAL_SHA,
            "old_identity_sha256": prior.digest(OLD_OUTPUT / "execution_identity.json"),
            "records": records,
            "full_sealed_files_verified": len(seal["files"]),
            "scientific_fits_or_queries": False,
        },
    )


def resource_check(path: Path) -> None:
    backend.require_disk_space(path)
    if datetime.now(UTC) >= datetime.fromisoformat(DEADLINE):
        _save_json(
            path.parent / f"deadline_stop_{datetime.now(UTC).timestamp():.6f}.json",
            {
                "next_profile": str(path),
                "deadline_utc": DEADLINE,
                "scientific_failure": False,
                "action": "Preserve partial schedule; no new solve or adoption",
            },
        )
        raise RuntimeError("Registered execution deadline reached")


def compact_fit(fit: dict[str, Any]) -> dict[str, Any]:
    """Retain meaningful trial records without changing their original identifiers."""
    trials = fit.get("trials", [])
    keep = {item["best_trial"] for item in fit.get("solves", []) if item.get("best_trial") is not None}
    if fit.get("selected_trial") is not None:
        keep.add(fit["selected_trial"])
    kept = [row for row in trials if row["status"] != "finite" or row["trial_id"] in keep]
    omitted = [row for row in trials if row["status"] == "finite" and row["trial_id"] not in keep]
    return dict(fit) | {
        "trials": kept,
        "trial_storage": {
            "schema": "selected_start_best_and_all_failure_trials_v1",
            "original_trial_count": len(trials),
            "retained_trial_count": len(kept),
            "omitted_finite_trial_count": len(omitted),
            "omitted_finite_trials_sha256": prior.canonical_hash(omitted),
            "original_full_fit_sha256": prior.canonical_hash(fit),
            "all_trial_replay_possible": False,
        },
    }


def adopted_fit(path: Path, header: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = adoption_manifest()
    if spec["ridge"] != 0:
        raise ValueError("Only lambda zero can be adopted")
    old_spec = {k: v for k, v in spec.items() if k not in {"ridge", "ridge_index"}}
    folder = OLD_OUTPUT / "cells/outer3" / spec["noise"] / spec["world"] / spec["candidate"]
    if spec["inner_fold"] is None:
        source = folder / "full.json.gz"
    else:
        source = folder / "conditional" / f"mu{spec['strength_index']}_shape{spec['shape_index']:03d}" / path.name
    expected_sha = manifest["records"][str(source.relative_to(OLD_OUTPUT))]
    if prior.digest(source) != expected_sha:
        raise ValueError("An adopted checkpoint changed")
    record = json.loads(gzip.decompress(source.read_bytes()))
    expected = header | {"spec": old_spec, "identity_sha256": manifest["old_identity_sha256"]}
    if record["header"] != expected or record["fit_sha256"] != prior.canonical_hash(record["fit"]):
        raise ValueError("Adoption requires identical spec, input values and payload")
    return record["fit"], {
        "path": str(source),
        "sha256": expected_sha,
        "original_fit_sha256": record["fit_sha256"],
        "old_identity_sha256": manifest["old_identity_sha256"],
        "adoption_manifest_sha256": prior.digest(OUTPUT / "adoption_manifest.json"),
    }


def conditional_fit(
    path: Path, matrix: np.ndarray, observed: np.ndarray, spec: dict[str, Any], identity_sha: str
) -> dict[str, Any]:
    header = {
        "spec": spec,
        "identity_sha256": identity_sha,
        "input_sha256": prior.canonical_hash({"matrix": matrix.tolist(), "observed": observed.tolist()}),
    }
    if path.exists():
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record["header"] != header or record["fit_sha256"] != prior.canonical_hash(record["fit"]):
            raise ValueError("Conditional checkpoint identity or payload differs")
        if spec["ridge"] == 0:
            original, lineage = adopted_fit(path, header, spec)
            expected = compact_fit(original) if spec["inner_fold"] is not None else original
            if record["adopted_from"] != lineage or record["fit"] != expected:
                raise ValueError("Resumed zero-ridge lineage or retained payload differs")
        elif record["adopted_from"] is not None:
            raise ValueError("Positive ridge must not adopt a zero-ridge fit")
        return record["fit"]
    if path.with_name(path.name + ".writing").exists():
        raise ValueError("Preserve interrupted compact checkpoint")
    resource_check(path)
    lineage = None
    if spec["ridge"] == 0:
        fit, lineage = adopted_fit(path, header, spec)
    else:
        try:
            starts = model.prior.gamma_starts(matrix, observed)
            fit = model.fit_response(matrix, observed, None, starts, ridge=spec["ridge"], evaluation_cap=5000)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            fit = {
                "status": "initialization_or_solver_failure",
                "selected": None,
                "unresolved": True,
                "solves": [],
                "trials": [],
                "exception": type(error).__name__,
                "message": str(error),
            }
    if spec["inner_fold"] is not None:
        fit = compact_fit(fit)
    record = {"header": header, "fit_sha256": prior.canonical_hash(fit), "fit": fit, "adopted_from": lineage}
    backend.atomic_bytes(
        path, gzip.compress(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
    )
    return fit


def choose(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in profiles if item["eligible_for_selection"]]
    if not eligible:
        raise ArithmeticError("No complete finite inner-CV candidate; all profiles remain saved")
    return min(
        eligible,
        key=lambda item: (
            item["inner_mse"],
            item["spec"]["ridge_index"],
            item["spec"]["strength_index"],
            item["spec"]["shape_index"],
        ),
    )


def save_json(path: Path, value: Any) -> None:
    """Keep frozen scorer arithmetic; replace only its obsolete ridge-zero metadata."""
    if path.name == "profiles.json":
        backend.atomic_bytes(
            path, (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        )
        return
    if path.name == "scores.json":
        value = dict(value) | {"ridge": prior.read(path.parent / "fit.json")["selected_candidate"]["ridge"]}
    if path == OUTPUT / "summary.json":
        value = dict(value) | {
            "rows": [
                dict(row)
                | {
                    "ridge": prior.read(old.cell(row["noise"], row["world"], row["candidate"]) / "fit.json")[
                        "selected_candidate"
                    ]["ridge"]
                }
                for row in value["rows"]
            ]
        }
    _save_json(path, value)


def fit_context(context: tuple[str, str]) -> None:
    noise, name = context
    if context not in old.contexts():
        raise ValueError("Unregistered noisy context")
    frozen = old.execution_identity()
    identity_sha = prior.digest(OUTPUT / "execution_identity.json")
    world = next(w for w in backend.WORLDS if w["name"] == name)
    data, folds = stage.oracle.inputs()
    split = stage.oracle.split_indices(data.rows, folds, old.OUTER)
    train, labels = split["train"], split["inner"]
    observed = old.training_only(world, noise, train)
    with closing(sqlite3.connect(f"file:{old.CACHE}?mode=ro&immutable=1", uri=True)) as connection:
        cache = backend.helper.Cache(
            connection, backend.helper.Checks(), frozen, {"scales": data.total_scales, "alpha": data.alpha}
        )
        native_profiles, native_fit = None, None
        for candidate in stage.oracle.candidates(world):
            if old.execution_identity() != frozen:
                raise ValueError("Sources changed before a family search")
            folder = old.cell(noise, name, candidate)
            if backend.completed_fit(folder, identity_sha):
                profiles, fitted = prior.read(folder / "profiles.json")["profiles"], prior.read(folder / "fit.json")
            else:
                clock = stage.oracle.CLOCKS[0] if candidate == "aggregate_mariner" else candidate
                profiles = []
                strengths = (0.0,) if candidate == "aggregate_mariner" else backend.STRENGTHS
                for ridge_index, ridge in enumerate(RIDGES):
                    for strength_index, strength in enumerate(strengths):
                        if strength == 0 and candidate != "aggregate_mariner":
                            if native_profiles is None:
                                raise ValueError("Native profiles required first")
                            profiles.extend(p for p in native_profiles if p["spec"]["ridge_index"] == ridge_index)
                            continue
                        for shape_index, shape in enumerate(backend.SHAPES):
                            spec = {
                                "world": name,
                                "noise": noise,
                                "outer_fold": old.OUTER,
                                "candidate": candidate,
                                "ridge_index": ridge_index,
                                "ridge": ridge,
                                "shape_index": shape_index,
                                "strength_index": strength_index,
                                "shape": shape,
                                "mu": strength,
                                "clock": clock,
                            }
                            matrix = cache.matrix(
                                data.early[train],
                                data.late[train],
                                backend.feature_spec(shape, strength, clock, data.alpha),
                            )
                            path = (
                                folder / "conditional" / f"ridge{ridge_index}_mu{strength_index}_shape{shape_index:03d}"
                            )
                            with threadpool_limits(limits=1):
                                profiles.append(
                                    old.template.profile_candidate(matrix, observed, labels, path, spec, identity_sha)
                                )
                save_json(folder / "profiles.json", {"identity_sha256": identity_sha, "profiles": profiles})
                selected = choose(profiles)
                spec = selected["spec"]
                if spec["mu"] == 0 and candidate != "aggregate_mariner":
                    if native_fit is None:
                        raise ValueError("Complete native head required")
                    fitted = backend.reuse_native_full(native_fit, selected, candidate)
                else:
                    matrix = cache.matrix(
                        data.early[train],
                        data.late[train],
                        backend.feature_spec(spec["shape"], spec["mu"], clock, data.alpha),
                    )
                    with threadpool_limits(limits=1):
                        full = conditional_fit(
                            folder / "full.json.gz", matrix, observed, spec | {"inner_fold": None}, identity_sha
                        )
                    fitted = {
                        "identity_sha256": identity_sha,
                        "world": name,
                        "noise": noise,
                        "outer_fold": old.OUTER,
                        "candidate": candidate,
                        "selected_candidate": spec,
                        "inner_mse": selected["inner_mse"],
                        "inner_predictions": selected["inner_predictions"],
                        "selected_inner_records": selected["conditional_records"],
                        "selected": full["selected"],
                        "full_diagnostics": prior.fit_diagnostics(full),
                        "full_unresolved": full["unresolved"],
                        "native_full_fit_reused": False,
                        "full_checkpoint": str(folder / "full.json.gz"),
                        "train_indices": train.tolist(),
                        "training_sha256": prior.canonical_hash(observed.tolist()),
                    }
                save_json(folder / "fit.json", fitted)
                conditional = {r["path"]: r["sha256"] for p in profiles for r in p["conditional_records"]}
                conditional[fitted["full_checkpoint"]] = prior.digest(Path(fitted["full_checkpoint"]))
                save_json(
                    folder / "fit_complete.json",
                    {
                        "identity_sha256": identity_sha,
                        "files": {f: prior.digest(folder / f) for f in ("profiles.json", "fit.json")},
                        "conditional_files": conditional,
                    },
                )
            if candidate == "aggregate_mariner":
                native_profiles, native_fit = profiles, fitted
            print(
                json.dumps(
                    {"noise": noise, "world": name, "candidate": candidate, "selected": fitted["selected_candidate"]}
                ),
                flush=True,
            )


old.OUTPUT, old.SCOPE, old.CONTRACT = OUTPUT, SCOPE, CONTRACT
old.source_paths, old.input_paths, old.identity = source_paths, input_paths, identity
old.template.conditional_fit = conditional_fit
backend.save_json = save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "fit", "score"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_adoption()
    elif args.command == "freeze":
        save_json(OUTPUT / "execution_identity.json", identity())
    elif args.command == "fit":
        old.execution_identity()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(fit_context, old.contexts()))
    else:
        old.score_all()


if __name__ == "__main__":
    main()
