# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Extend the sealed supplied-floor search to four remaining synthetic folds.

Only 160 new heads are fitted. The forty sealed outer-zero heads retain their
original identities. All 200 receive unchanged descriptive menu scoring.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
sys.path.insert(0, str(ROOT))


def private_module(filename: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prior = private_module("run_tpp40_acquisition_retention_known_floor_20261015.py", "_tp034_all_folds_prior")
menu = private_module("run_tpp40_acquisition_retention_oracle_extension_20261015.py", "_tp034_all_folds_menu")
oracle, models, benchmark = prior.oracle, prior.models, prior.benchmark
PARENT, STAGE1 = prior.PARENT, prior.STAGE1
PRIOR = prior.OUTPUT
OUTPUT = PARENT / "known_floor_all_folds"
PRIOR_SEAL = PRIOR / "result_manifest_v1.json"
PRIOR_AUDIT = PRIOR / "audit/author_replay_report_v1.json"
PRIOR_DISPOSITION = PRIOR / "audit/root_disposition_v1.json"
NOISES, STATUS = prior.NOISES, prior.STATUS
NEW_FOLDS = (1, 2, 3, 4)
SCOPE = "tp034_known_floor_all_folds_160_new_40_reused"
CONTRACT = {
    key: item for key, item in prior.CONTRACT.items() if key not in {"scope", "outer_fold", "heads", "contexts"}
} | {
    "scope": SCOPE,
    "new_outer_folds": list(NEW_FOLDS),
    "reused_outer_folds": [0],
    "new_contexts": 64,
    "reused_contexts": 16,
    "new_heads": 160,
    "reused_heads": 40,
    "reported_heads": 200,
    "new_native_heads": 64,
    "new_temporal_heads": 96,
    "new_grid_configurations": 618240,
    "new_profile_records_including_retained_native": 618336,
    "workers": 1,
    "numeric_threads": 1,
    "cache": "one verified local copy; serial mutable writes; sealed original remains read-only",
    "menu_scorer": "unchanged oracle_extension.scores",
    "frontier_fraction": menu.CONTRACT["frontier_fraction"],
    "gain_sign_zero_tolerance": menu.CONTRACT["gain_sign_zero_tolerance"],
    "regret_at_k": [1, 3, 5],
    "new_acceptance_gate": False,
}


def source_paths() -> set[Path]:
    return (
        prior.source_paths()
        | menu.source_paths()
        | {
            Path(__file__),
            ROOT / "test_tpp40_acquisition_retention_known_floor_all_folds_20261015.py",
            OUTPUT / "proposal_v1.md",
        }
    )


def input_paths() -> set[Path]:
    return prior.input_paths() | {
        PRIOR_SEAL,
        PRIOR_AUDIT,
        PRIOR_DISPOSITION,
        PRIOR / "registration.json",
        PRIOR / "execution_identity.json",
        PRIOR / "complete.json",
        PRIOR / "features.sqlite",
    }


def verify_prior(value: dict[str, Any]) -> None:
    pins = value["prior_prerequisite"]
    for key, path in (
        ("manifest_sha256", PRIOR_SEAL),
        ("audit_sha256", PRIOR_AUDIT),
        ("root_disposition_sha256", PRIOR_DISPOSITION),
    ):
        if benchmark.sha256(path) != pins[key]:
            raise ValueError("The sealed outer-zero prerequisite differs")
    seal = json.loads(PRIOR_SEAL.read_text())
    benchmark.verify_hashes({name: record["sha256"] for name, record in seal["files"].items()}, PRIOR)
    audit = json.loads(PRIOR_AUDIT.read_text())
    disposition = json.loads(PRIOR_DISPOSITION.read_text())
    if not audit["passed"] or not disposition["passed"] or not disposition["learned_floor_domain_failure_preserved"]:
        raise ValueError("Require the audited prerequisite while preserving failed learned-floor readiness")
    prior.registration()


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    if value.get("status") != "registered" or value.get("scope") != SCOPE:
        raise ValueError("Register this separate all-fold diagnostic before execution")
    if value.get("contract") != CONTRACT or value.get("environment") != benchmark.environment_identity():
        raise ValueError("The registered search, scoring, or runtime differs")
    for key, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(value[key]) != {str(path.resolve()) for path in paths}:
            raise ValueError("The exact source and input whitelists must be pinned")
        benchmark.verify_hashes(value[key])
    verify_prior(value)
    return value


def source_identity() -> dict[str, Any]:
    value = registration()
    return {
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        **{key: value[key] for key in ("source_hashes", "input_hashes", "environment", "prior_prerequisite")},
    }


def freeze() -> None:
    """Freeze provenance without loading policy or outcome arrays."""
    identity = source_identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Preserve the prior execution identity")
    benchmark.write_json(path, identity)


def execution_identity() -> dict[str, Any]:
    identity = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != identity:
        raise ValueError("Frozen all-fold identity changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": SCOPE,
        "registration_sha256": identity["registration_sha256"],
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != item for key, item in expected.items()):
        raise ValueError("Exact all-fold diagnostic approval is required")
    return identity


def contexts(folds: tuple[int, ...]) -> list[tuple[int, str, str]]:
    return [(fold, world["name"], noise) for fold in folds for world in oracle.WORLDS for noise in NOISES]


def cell_folder(fold: int, world: str, noise: str, candidate: str) -> Path:
    if fold == 0:
        return PRIOR / "cells" / world / noise / candidate
    return OUTPUT / "cells" / f"outer{fold}" / world / noise / candidate


def adopted_records() -> list[dict[str, Any]]:
    """Verify old cells against their original identity, without loading fitted heads."""
    identity = json.loads((PRIOR / "execution_identity.json").read_text())
    records = []
    for world in oracle.WORLDS:
        for noise in NOISES:
            for candidate in oracle.candidates(world):
                folder = cell_folder(0, world["name"], noise, candidate)
                spec = {"world": world, "noise": noise, "candidate": candidate, "outer_fold": 0, "status": STATUS}
                if not oracle.completed(folder, spec, identity):
                    raise ValueError("All forty original cells must be complete")
                records.append(
                    {
                        "spec": spec,
                        "source_complete": str(folder / "complete.json"),
                        "source_complete_sha256": benchmark.sha256(folder / "complete.json"),
                    }
                )
    if len(records) != CONTRACT["reused_heads"]:
        raise ValueError("The original forty-cell scope differs")
    return records


def verify_adoption(identity: dict[str, Any]) -> None:
    expected = {
        "identity": identity,
        "original_identity_sha256": benchmark.sha256(PRIOR / "execution_identity.json"),
        "records": adopted_records(),
    }
    if json.loads((OUTPUT / "adopted_outer0.json").read_text()) != expected:
        raise ValueError("The immutable outer-zero adoption record differs")


def seed_cache(source: Path, destination: Path, source_sha256: str) -> dict[str, Any]:
    """Copy a closed feature-only cache once; never overwrite an unexplained destination."""
    receipt_path = destination.parent / "cache_seed.json"
    if benchmark.sha256(source) != source_sha256:
        raise ValueError("The sealed cache source changed")
    wal = Path(str(source) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("A nonempty source WAL cannot be discarded")
    expected = {
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "destination": str(destination.resolve()),
        "initial_copy_sha256": source_sha256,
    }
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != expected or not destination.exists():
            raise ValueError("The existing cache seed receipt or destination differs")
        return expected
    temporary = destination.with_suffix(".copying")
    if destination.exists() or temporary.exists():
        raise ValueError("Preserve the unexplained cache copy; do not overwrite it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temporary)
    if benchmark.sha256(temporary) != source_sha256:
        raise ValueError("The copied feature cache failed its hash check")
    temporary.replace(destination)
    benchmark.write_json(receipt_path, expected)
    return expected


def verify_cache_seed() -> None:
    seed_cache(PRIOR / "features.sqlite", OUTPUT / "features.sqlite", benchmark.sha256(PRIOR / "features.sqlite"))


def prepare() -> None:
    identity = execution_identity()
    expected = {
        "identity": identity,
        "original_identity_sha256": benchmark.sha256(PRIOR / "execution_identity.json"),
        "records": adopted_records(),
    }
    path = OUTPUT / "adopted_outer0.json"
    if path.exists() and json.loads(path.read_text()) != expected:
        raise ValueError("Preserve the prior adoption record")
    benchmark.write_json(path, expected)
    verify_cache_seed()


def run_context(context: tuple[int, str, str]) -> dict[str, Any]:
    identity = execution_identity()
    fold, world_name, noise_name = context
    if context not in contexts(NEW_FOLDS):
        raise ValueError("Only registered folds one through four may be fitted")
    verify_adoption(identity)
    verify_cache_seed()
    world = next(item for item in oracle.WORLDS if item["name"] == world_name)
    if noise_name not in NOISES:
        raise ValueError("Unregistered noise context")
    data, folds = oracle.inputs()
    split = oracle.split_indices(data.rows, folds, fold)
    train, labels = split["train"], split["inner"]
    query0, query1, query = oracle.query_frame(data, split)
    with np.load(STAGE1 / "prepared/clean.npz") as original, np.load(STAGE1 / "prepared/noise.npz") as noise:
        clean = original[world_name]
        observed = prior.training_values(clean, noise[noise_name], train)
    with np.load(STAGE1 / "prepared/matched_tied.npz") as matched:
        truth = np.r_[clean[split["test"]], matched[world_name][split["test"]], clean[split["tied_test"]]]
    cache = models.BenefitCache(OUTPUT / "features.sqlite")
    aggregate, receipts = None, []
    for candidate in oracle.candidates(world):
        identity = execution_identity()
        spec = {"world": world, "noise": noise_name, "candidate": candidate, "outer_fold": fold, "status": STATUS}
        folder = cell_folder(fold, world_name, noise_name, candidate)
        if oracle.completed(folder, spec, identity):
            fit = models.JointFit.from_json(json.loads((folder / "fit.json").read_text()))
        else:
            if (folder / "numerical_failure.json").exists():
                raise ValueError("Preserve the recorded numerical failure; a retry needs a separate reviewed revision")
            config = oracle.feature_config(
                data.total_scales, oracle.CLOCKS[0] if candidate == "aggregate_mariner" else candidate
            )
            folder.mkdir(parents=True, exist_ok=True)
            try:
                with threadpool_limits(limits=1):
                    fit = prior.fit_family(
                        data.early[train],
                        data.late[train],
                        observed,
                        labels,
                        data.alpha,
                        config,
                        candidate,
                        aggregate,
                        cache,
                    )
                    inner = prior.selected_inner(fit, data.early[train], data.late[train], observed, labels, cache)
                    predicted = fit.predict(query0, query1, cache)
                    numerical = prior.prediction_status(fit, query0, query1, cache)
                if not numerical["finite_logits"] or not np.isfinite(predicted).all():
                    raise ArithmeticError("The selected head produced nonfinite query predictions")
                if abs(inner["selected_score_replay_difference"]) > 1e-12:
                    raise ArithmeticError("Selected inner score does not replay at the unchanged candidate")
            except (ArithmeticError, RuntimeError) as error:
                benchmark.write_json(
                    folder / "numerical_failure.json",
                    {
                        "spec": spec,
                        "identity": identity,
                        "exception": type(error).__name__,
                        "message": str(error),
                        "status": STATUS,
                    },
                )
                raise
            benchmark.write_json(folder / "fit.json", fit.to_json())
            benchmark.write_json(folder / "inner.json", inner)
            applicable = noise_name == "clean" and (world["kind"] == "native" or candidate == world["clock"])
            scores = oracle.prediction_scores(predicted, truth, len(split["test"]))
            scores.update(
                {
                    "status": STATUS,
                    "shape": fit.shape,
                    "ridge": fit.ridge,
                    "selected_mu": fit.parameters.get("mu", 0.0),
                    "selected_upper_boundary": fit.parameters.get("mu", 0.0) == oracle.STRENGTHS[-1],
                    "floor": fit.head.floor,
                    "gamma": fit.kappa,
                    "inner_rmse": fit.inner_rmse,
                    "clean_matching_recovery_reference_applicable": applicable,
                    "clean_matching_recovery_reference_met": (
                        None
                        if not applicable
                        else scores["query_max_abs_error"] <= CONTRACT["clean_recovery_reference_tolerance_bpb"]
                    ),
                    "numerical": numerical,
                    "scientific_promotion": False,
                    "learned_floor_domain_ready": False,
                }
            )
            benchmark.write_json(folder / "scores.json", scores)
            query.assign(truth=truth, predicted=predicted).to_csv(folder / "predictions.csv", index=False)
            dependencies = (OUTPUT / "adopted_outer0.json", OUTPUT / "cache_seed.json")
            if candidate != "aggregate_mariner":
                dependencies += (folder.parent / "aggregate_mariner/complete.json",)
            oracle.finish(
                folder, spec, identity, ("fit.json", "inner.json", "scores.json", "predictions.csv"), dependencies
            )
        if candidate == "aggregate_mariner":
            aggregate = fit
        receipts.append(str(folder / "complete.json"))
    return {"outer_fold": fold, "world": world_name, "noise": noise_name, "receipts": receipts}


def fit_all() -> None:
    prepare()
    for context in contexts(NEW_FOLDS):
        print(json.dumps(run_context(context)), flush=True)
    summarize()


def summarize() -> None:
    identity = execution_identity()
    verify_adoption(identity)
    records, dependencies = [], [OUTPUT / "adopted_outer0.json", OUTPUT / "cache_seed.json"]
    old_identity = json.loads((PRIOR / "execution_identity.json").read_text())
    for fold, world_name, noise in contexts((0, *NEW_FOLDS)):
        world = next(item for item in oracle.WORLDS if item["name"] == world_name)
        native = None
        for candidate in oracle.candidates(world):
            folder = cell_folder(fold, world_name, noise, candidate)
            spec = {"world": world, "noise": noise, "candidate": candidate, "outer_fold": fold, "status": STATUS}
            if not oracle.completed(folder, spec, old_identity if fold == 0 else identity):
                raise ValueError("Preserve incomplete or failed diagnostic cells")
            scores = json.loads((folder / "scores.json").read_text())
            query = pd.read_csv(folder / "predictions.csv", keep_default_na=False, float_precision="round_trip")
            descriptive = menu.scores(query.predicted.to_numpy(float), query.truth.to_numpy(float), query)
            if candidate == "aggregate_mariner":
                native = scores
            assert native is not None
            records.append(
                {
                    **spec,
                    **scores,
                    "origin": "reused_outer0" if fold == 0 else "new_fold",
                    "source_complete": str(folder / "complete.json"),
                    "source_complete_sha256": benchmark.sha256(folder / "complete.json"),
                    "menu_metrics": descriptive,
                    "delta_vs_native": {
                        key: scores[key] - native[key]
                        for key in ("asymmetric_rmse", "matched_tied_rmse", "actual_tied_rmse", "gain_rmse")
                    },
                }
            )
            dependencies.append(folder / "complete.json")
    if len(records) != CONTRACT["reported_heads"]:
        raise ValueError("All two hundred heads must remain visible")
    benchmark.write_json(
        OUTPUT / "summary.json",
        {
            "status": STATUS,
            "heads": len(records),
            "new_heads": CONTRACT["new_heads"],
            "reused_heads": CONTRACT["reused_heads"],
            "records": records,
            "learned_floor_domain_ready": False,
            "scientific_promotion": False,
            "scope": "Supplied-floor synthetic complete search; unchanged descriptive menu metrics; no new gate",
        },
    )
    oracle.finish(OUTPUT, {"stage": "summary", "status": STATUS}, identity, ("summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "fit", "summarize"))
    args = parser.parse_args()
    {"freeze": freeze, "prepare": prepare, "fit": fit_all, "summarize": summarize}[args.command]()


if __name__ == "__main__":
    main()
