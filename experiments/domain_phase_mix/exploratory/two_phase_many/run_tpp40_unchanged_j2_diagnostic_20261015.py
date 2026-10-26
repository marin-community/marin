# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP40-037: unchanged J2 under the inherited floor domain, diagnostic only.

Ten clean outer-zero heads. Original stage two remains blocked. No noise fits,
raw policy search, measured outcomes, altered estimator, or automatic promotion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_known_floor_20261015 as known,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_stage2_20261015 as stage,
)

oracle, models, benchmark = stage.oracle, stage.models, stage.benchmark
ROOT, PARENT = stage.ROOT, oracle.PARENT
OUTPUT = PARENT / "unchanged_j2_diagnostic"
KNOWN_SEAL = known.OUTPUT / "result_manifest_v1.json"
SOURCE_CACHE = known.OUTPUT / "features.sqlite"
CACHE = OUTPUT / "features.sqlite"
SCOPE = "tp037_unchanged_j2_ten_clean_outer_zero_heads"
STATUS = "inherited_floor_domain_diagnostic_not_stage2_qualification"
CONTRACT = {
    "scope": SCOPE,
    "candidate_id": "TP40-037",
    "status": STATUS,
    "worlds": list(oracle.WORLDS),
    "heads": 10,
    "outer_fold": 0,
    "repeat": 0,
    "noise_sd": 0.0,
    "strengths": list(oracle.STRENGTHS),
    "shapes": list(models.mariner.SHAPES),
    "ridges": list(models.mariner.RIDGE_GRID),
    "gamma_grid": list(models.frozen.GAMMA_GRID),
    "gamma_bounds": list(models.mariner.KAPPA_BOUNDS),
    "quadrature": oracle.CONTRACT["quadrature"],
    "full_refined_native_control_retained": True,
    "shape_override": None,
    "test_grid_override": None,
    "known_floor": None,
    "historical_thresholds_descriptive_only": stage.CONTRACT["gates"],
    "clean_recovery_reference_tolerance_bpb": 1e-8,
    "zero_gain_ratio_acceptance_gate": False,
    "original_stage2_remains_blocked": True,
    "scientific_promotion": False,
    "raw_optimization": False,
    "new_truth_generation": False,
    "empirical_outcomes": False,
    "cache_owner_count": 1,
}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def source_paths() -> set[Path]:
    return (
        stage.source_paths()
        | known.source_paths()
        | {
            Path(__file__),
            ROOT / "test_tpp40_unchanged_j2_diagnostic_20261015.py",
            PARENT / "reviews/unchanged_j2_floor_domain_diagnostic_proposal_v1.md",
        }
    )


def input_paths() -> set[Path]:
    return known.input_paths() | {
        KNOWN_SEAL,
        SOURCE_CACHE,
        known.OUTPUT / "execution_identity.json",
        known.OUTPUT / "registration.json",
        known.OUTPUT / "audit/author_replay_report_v1.json",
    }


def verify_prerequisites(value: dict[str, Any]) -> None:
    known.verify_stage1(value)
    if benchmark.sha256(KNOWN_SEAL) != value["known_floor_manifest_sha256"]:
        raise ValueError("The complete known-floor diagnostic seal differs")
    seal = read(KNOWN_SEAL)
    benchmark.verify_hashes({name: item["sha256"] for name, item in seal["files"].items()}, known.OUTPUT)
    if seal["scientific_promotion"] or not seal["learned_floor_domain_failure_preserved"]:
        raise ValueError("Preserve the known-floor diagnostic's limited disposition")
    if not read(known.OUTPUT / "audit/author_replay_report_v1.json")["passed"]:
        raise ValueError("Require the reviewed known-floor numerical replay")
    # Do not call the blocked stage-two registration/entrypoint or change its files.
    for suffix in ("-wal", "-shm"):
        if Path(str(SOURCE_CACHE) + suffix).exists():
            raise ValueError("The sealed source cache must be closed before copying")


def registration() -> dict[str, Any]:
    value = read(OUTPUT / "registration.json")
    if value.get("status") != "registered" or value.get("contract") != CONTRACT:
        raise ValueError("Register this separate diagnostic's exact contract")
    if value.get("environment") != benchmark.environment_identity():
        raise ValueError("The registered runtime differs")
    for name, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(value[name]) != {str(path.resolve()) for path in paths}:
            raise ValueError("The source/input pin inventory differs")
        benchmark.verify_hashes(value[name])
    verify_prerequisites(value)
    models.verify_sources()
    return value


def source_identity() -> dict[str, Any]:
    value = registration()
    return {
        key: value[key]
        for key in ("source_hashes", "input_hashes", "environment", "stage1_prerequisite", "known_floor_manifest_sha256")
    } | {"registration_sha256": benchmark.sha256(OUTPUT / "registration.json")}


def freeze() -> None:
    identity = source_identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and read(path) != identity:
        raise ValueError("Preserve the prior execution identity and review a separate revision")
    benchmark.write_json(path, identity)


def execution_identity() -> dict[str, Any]:
    identity = source_identity()
    if read(OUTPUT / "execution_identity.json") != identity:
        raise ValueError("The frozen diagnostic identity changed")
    required = {
        "scope": SCOPE,
        "registration_sha256": identity["registration_sha256"],
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    approval = read(OUTPUT / "execution_approval.json")
    if any(approval.get(key) != value for key, value in required.items()):
        raise ValueError("The exact separate diagnostic requires execution approval")
    return identity


def copy_closed_cache(source: Path, destination: Path, expected: str) -> dict[str, Any]:
    """Copy a closed sealed database before the unchanged writer opens its fork."""
    if source.resolve() == destination.resolve():
        raise ValueError("Never open the sealed cache for writing")
    if benchmark.sha256(source) != expected or any(Path(str(source) + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("Source cache differs or is not closed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if benchmark.sha256(destination) != expected:
            raise ValueError("Unreceipted cache contents differ; preserve and investigate")
    else:
        temporary = destination.with_name(destination.name + ".copying")
        if temporary.exists():
            raise ValueError("Preserve the unexplained interrupted cache copy before any retry")
        shutil.copyfile(source, temporary)
        if benchmark.sha256(temporary) != expected:
            raise ValueError("The closed cache copy does not reproduce its source")
        temporary.replace(destination)
    return {"source": str(source), "destination": str(destination), "initial_sha256": expected}


def prepare() -> None:
    identity = execution_identity()
    folder, spec = OUTPUT / "prepared", {"stage": "cache_copy", "contract": CONTRACT}
    if oracle.completed(folder, spec, identity):
        if not CACHE.exists():
            raise ValueError("The writable cache fork is missing")
        return
    origin = copy_closed_cache(SOURCE_CACHE, CACHE, identity["input_hashes"][str(SOURCE_CACHE.resolve())])
    benchmark.write_json(
        folder / "cache_origin.json",
        origin
        | {
            "original_stage2_remains_blocked": True,
            "policy": "Only this fork can receive unchanged numerical feature evaluations on frozen query policies",
        },
    )
    oracle.finish(folder, spec, identity, ("cache_origin.json",), (KNOWN_SEAL, SOURCE_CACHE))


def known_comparison(world: str, candidate: str, query: pd.DataFrame, truth: np.ndarray) -> tuple[dict[str, Any], Path]:
    folder = known.OUTPUT / "cells" / world / "clean" / candidate
    receipt = read(folder / "complete.json")
    if not oracle.completed(folder, receipt["spec"], read(known.OUTPUT / "execution_identity.json")):
        raise ValueError("The exact old known-floor cell is missing")
    old = pd.read_csv(folder / "predictions.csv")
    if not old[["role", "row_id", "source_index"]].equals(query[["role", "row_id", "source_index"]]):
        raise ValueError("The paired known-floor query identities differ")
    np.testing.assert_allclose(old.truth, truth, atol=2e-15, rtol=2e-15)
    scores = read(folder / "scores.json")
    return scores | stage.menu_scores(old.predicted.to_numpy(), truth, query), folder / "complete.json"


def comparison_readout(scores: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        "query_rmse",
        "query_max_abs_error",
        "asymmetric_rmse",
        "matched_tied_rmse",
        "actual_tied_rmse",
        "gain_rmse",
        "frontier_mean_optimism",
        "regret_at_1",
        "regret_at_3",
        "regret_at_5",
    )
    return {
        "delta_vs_known_floor": {name: scores[name] - supplied[name] for name in metrics},
        "gain_rmse_divided_by_zero_gain_baseline": (
            scores["gain_rmse"] / scores["true_gain_rms"] if scores["true_gain_rms"] > 1e-14 else None
        ),
        "scientific_promotion": False,
        "original_stage2_remains_blocked": True,
    }


def run() -> None:
    prepare()
    data, folds = oracle.inputs()
    split = oracle.split_indices(data.rows, folds, 0)
    train, labels = split["train"], split["inner"]
    query0, query1, query = oracle.query_frame(data, split)
    cache = models.BenefitCache(CACHE)
    with (
        np.load(known.STAGE1 / "prepared/clean.npz") as clean,
        np.load(known.STAGE1 / "prepared/matched_tied.npz") as tied,
    ):
        for world in oracle.WORLDS:
            aggregate = None
            values = clean[world["name"]]
            truth = np.r_[values[split["test"]], tied[world["name"]][split["test"]], values[split["tied_test"]]]
            for candidate in oracle.candidates(world):
                identity = execution_identity()
                spec = {"world": world["name"], "candidate": candidate, "outer_fold": 0, "status": STATUS}
                folder = OUTPUT / "cells" / world["name"] / candidate
                if oracle.completed(folder, spec, identity):
                    fit = models.JointFit.from_json(read(folder / "model.json"))
                    if candidate == "aggregate_mariner":
                        aggregate = fit
                    continue
                if (folder / "numerical_failure.json").exists():
                    raise ValueError("Preserve the failed cell; no automatic rescue or retry revision")
                config = oracle.feature_config(
                    data.total_scales, oracle.CLOCKS[0] if candidate == "aggregate_mariner" else candidate
                )
                fit = None
                try:
                    with threadpool_limits(limits=1):
                        fit = stage.fit_candidate(
                            data.early[train],
                            data.late[train],
                            values[train],
                            labels,
                            data.alpha,
                            candidate,
                            config,
                            cache,
                            aggregate,
                        )
                        predicted = fit.predict(query0, query1, cache)
                        inner, heads = stage.selected_inner(
                            fit, data.early[train], data.late[train], values[train], labels, cache
                        )
                        scores = stage.diagnostic_scores(fit, predicted, truth, query)
                        scores["inner_rmse_replay"] = float(np.sqrt(np.mean((inner - values[train]) ** 2)))
                        if abs(scores["inner_rmse_replay"] - fit.inner_rmse) > 1e-12 or not np.isfinite(predicted).all():
                            raise ArithmeticError("Selected score mismatch or nonfinite query prediction")
                        clips = stage.calibration_flags(fit, query0, query1, cache)
                        scores.update(
                            {
                                "query_clipping_count": int(np.sum(clips)),
                                "training_deficit_clip_count": int(
                                    np.sum(values[train] - fit.head.floor <= models.mariner.DEFICIT_FLOOR)
                                ),
                                "inner_training_deficit_clip_count": sum(
                                    head["training_deficit_clip_count"] for head in heads
                                ),
                                "inner_prediction_clip_count": sum(head["held_prediction_clip_count"] for head in heads),
                                "clean_matching_recovery_reference_applicable": (
                                    world["kind"] == "native" or candidate == world["clock"]
                                ),
                            }
                        )
                    supplied, prior_receipt = known_comparison(world["name"], candidate, query, truth)
                    scores.update(comparison_readout(scores, supplied))
                    scores["clean_matching_recovery_reference_met"] = (
                        scores["query_max_abs_error"] <= CONTRACT["clean_recovery_reference_tolerance_bpb"]
                        if scores["clean_matching_recovery_reference_applicable"]
                        else None
                    )
                    benchmark.write_json(folder / "model.json", fit.to_json())
                    benchmark.write_json(folder / "inner_heads.json", heads)
                    benchmark.write_json(folder / "scores.json", scores)
                    benchmark.write_json(folder / "known_floor_comparison.json", supplied)
                    pd.DataFrame(
                        {"source_index": train, "inner_fold": labels, "truth": values[train], "predicted": inner}
                    ).to_csv(folder / "inner_predictions.csv", index=False)
                    query.assign(truth=truth, predicted=predicted, clipped=clips).to_csv(
                        folder / "predictions.csv", index=False
                    )
                    dependencies = (OUTPUT / "prepared/complete.json", prior_receipt)
                    if candidate != "aggregate_mariner":
                        dependencies += (folder.parent / "aggregate_mariner/complete.json",)
                    oracle.finish(
                        folder,
                        spec,
                        identity,
                        (
                            "model.json",
                            "inner_heads.json",
                            "scores.json",
                            "known_floor_comparison.json",
                            "inner_predictions.csv",
                            "predictions.csv",
                        ),
                        dependencies,
                    )
                except (ArithmeticError, ValueError, RuntimeError) as error:
                    if fit is not None:
                        benchmark.write_json(folder / "failed_model.json", fit.to_json())
                    benchmark.write_json(
                        folder / "numerical_failure.json",
                        {"spec": spec, "identity": identity, "exception": type(error).__name__, "message": str(error)},
                    )
                    raise
                if candidate == "aggregate_mariner":
                    aggregate = fit
                print(json.dumps(spec), flush=True)
    with closing(sqlite3.connect(CACHE)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    summarize()


def summarize() -> None:
    identity = execution_identity()
    records, dependencies = [], []
    for world in oracle.WORLDS:
        native = read(OUTPUT / "cells" / world["name"] / "aggregate_mariner/scores.json")
        for candidate in oracle.candidates(world):
            spec = {"world": world["name"], "candidate": candidate, "outer_fold": 0, "status": STATUS}
            folder = OUTPUT / "cells" / world["name"] / candidate
            if not oracle.completed(folder, spec, identity):
                raise ValueError("All ten diagnostic heads, including failures, must remain accounted for")
            score = read(folder / "scores.json")
            records.append(
                spec
                | score
                | {"historical_threshold_checks_descriptive_only": stage.gate_checks(world, candidate, score, native)}
            )
            dependencies.append(folder / "complete.json")
    if len(records) != 10:
        raise ValueError("The ten-head inventory differs")
    benchmark.write_json(
        OUTPUT / "summary.json",
        {
            "status": STATUS,
            "heads": len(records),
            "records": records,
            "original_stage2_remains_blocked": True,
            "scientific_promotion": False,
            "learned_floor_domain_ready": False,
            "further_execution_authorized": False,
            "historical_checks_are_not_qualification": True,
        },
    )
    oracle.finish(OUTPUT, {"stage": "summary", "status": STATUS}, identity, ("summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "run", "summarize"))
    {"freeze": freeze, "prepare": prepare, "run": run, "summarize": summarize}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
