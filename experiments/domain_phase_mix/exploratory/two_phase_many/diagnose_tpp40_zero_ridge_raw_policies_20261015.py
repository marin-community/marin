# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP052: apply the frozen TP051 raw procedure to all30 sealed zero-ridge heads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
FROZEN_SOURCE = BASE / "diagnose_tpp40_ridge_raw_policies_20261015.py"
FROZEN_SHA = "49b777089b14325ae89751edacd12fbe70b85ae0862b7d8a1397046aa767960f"
sys.path.insert(0, str(BASE.parents[3]))


def private_runner() -> Any:
    if hashlib.sha256(FROZEN_SOURCE.read_bytes()).hexdigest() != FROZEN_SHA:
        raise ValueError("Frozen TP051 raw procedure changed")
    spec = importlib.util.spec_from_file_location("_tp052_private_raw_procedure", FROZEN_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned raw procedure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backend = private_runner()
prior, oracle = backend.prior, backend.oracle
OUTPUT = oracle.PARENT / "zero_ridge_raw_policy_diagnostic"
UPSTREAM = oracle.PARENT / "whole_response_fold3_noise"
UPSTREAM_SEAL_SHA = "24466a9ea9d9e014279edd03801a163424f97dca9f06d8f1b201af53d53eb36e"
SCOPE = "tp052_all30_frozen_tp046_zero_ridge_raw_policy_falsifier_no_refits"
CONTRACT = dict(backend.CONTRACT) | {
    "candidate_id": "TP40-052",
    "scope": SCOPE,
    "paired_comparison": "all30 TP046 zero-ridge heads versus TP049 under unchanged TP051 raw procedure",
    "original_query_failures": "preserve all three sealed TP046 query failures; no head filtering",
    "upstream_replay": "60480 unique conditional records; same selected/full head and failed-query lineage",
}
_old_sources, _old_inputs = backend.source_paths, backend.input_paths


def source_paths() -> set[Path]:
    return _old_sources() | {Path(__file__), BASE / "test_tpp40_zero_ridge_raw_policies_20261015.py"}


def input_paths() -> set[Path]:
    return _old_inputs() | {UPSTREAM / "audit/independent_result_review_v1.json"}


def verify_upstream() -> None:
    if prior.digest(UPSTREAM / "result_manifest_v1.json") != UPSTREAM_SEAL_SHA:
        raise ValueError("The sealed TP046 archive changed")
    if prior.read(UPSTREAM / "complete.json")["completed_heads"] != 30:
        raise ValueError("All30 TP046 outcomes must be finalized")
    registration = prior.read(UPSTREAM / "registration.json")
    prior.verify_hashes(registration["source_hashes"])
    for spec in backend.specs():
        if spec["candidate"] == "aggregate_mariner":
            folder = prior.OUTPUT / "cells/outer3" / spec["noise"] / spec["world"] / "initialization"
            prior.verify_hashes(
                {
                    str(folder / name): registration["input_hashes"][str(folder / name)]
                    for name in ("starts.json", "complete.json")
                }
            )
    historical = prior.read(UPSTREAM / "execution_identity.json")
    if historical["source_hashes"] != registration["source_hashes"] or historical["registration_sha256"] != prior.digest(
        UPSTREAM / "registration.json"
    ):
        raise ValueError("TP046 source/registration identity changed")
    report = prior.read(UPSTREAM / "audit/author_replay_report_v1.json")
    if report["status"] != "author_saved_output_replay_passed" or report["unique_inner_conditional_records"] != 60480:
        raise ValueError("Require the sealed TP046 saved-head replay")
    population = [
        {key: row[key] for key in ("noise", "world", "candidate")} for row in report["rows"] + report["query_failures"]
    ]
    if sorted(population, key=prior.canonical_hash) != sorted(backend.specs(), key=prior.canonical_hash):
        raise ValueError("TP046 replay must account for exactly30 identities including failures")
    if len(report["query_failures"]) != 3:
        raise ValueError("Preserve the three original TP046 query failures")
    seal = prior.read(UPSTREAM / "result_manifest_v1.json")
    frozen_identity = prior.digest(UPSTREAM / "execution_identity.json")
    for spec in backend.specs():
        folder = backend.cell(UPSTREAM, spec)
        receipt = prior.read(folder / "fit_complete.json")
        if receipt["identity_sha256"] != frozen_identity:
            raise ValueError("A frozen full-fit receipt has a foreign identity")
        prior.verify_hashes({str(folder / name): sha for name, sha in receipt["files"].items()})
        fit = prior.read(folder / "fit.json")
        checkpoint = Path(fit["full_checkpoint"])
        prior.verify_hashes({str(checkpoint): receipt["conditional_files"][str(checkpoint)]})
        for path in (folder / "fit.json", folder / "fit_complete.json", checkpoint):
            prior.verify_hashes({str(path): seal["files"][str(path.relative_to(UPSTREAM))]["sha256"]})


backend.OUTPUT, backend.UPSTREAM, backend.SCOPE, backend.CONTRACT = OUTPUT, UPSTREAM, SCOPE, CONTRACT
backend.source_paths, backend.input_paths, backend.verify_upstream = source_paths, input_paths, verify_upstream
models, SEARCH, REPLAY = backend.models, backend.SEARCH, backend.REPLAY
specs, cell, approved = backend.specs, backend.cell, backend.approved


def optimize_cell(spec: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Public spawn-safe entry point into the isolated, unchanged raw procedure."""
    backend.optimize_cell(spec, frozen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "optimize", "score", "summarize"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "freeze":
        frozen = backend.identity()
        path = OUTPUT / "execution_identity.json"
        if path.exists() and prior.read(path) != frozen:
            raise ValueError("Preserve changed frozen identity")
        prior.write(path, frozen)
        return
    frozen = backend.approved()
    if args.command == "optimize":
        backend.select_observed_baselines(frozen)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(optimize_cell, backend.specs(), [frozen] * len(backend.specs())))
    elif args.command == "score":
        backend.score_all(frozen)
        backend.summarize(frozen)
    else:
        backend.summarize(frozen)


if __name__ == "__main__":
    main()
