# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Full shape/kinetic/ridge CV with an oracle floor on fixed synthetic outcomes.

Exactly forty outer-fold-zero heads; no empirical outcomes, floor learning,
raw policy optimization, or scientific promotion. Requires its own approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_oracle_20261015 as oracle,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_acquisition_retention_joint_models_20261015 as models,
)

benchmark = oracle.benchmark
ROOT, PARENT = oracle.ROOT, oracle.PARENT
OUTPUT = PARENT / "known_floor_diagnostic"
STAGE1 = PARENT / "stage1"
STAGE1_SEAL = STAGE1 / "result_manifest.json"
STAGE1_AUDIT = STAGE1 / "audit/independent_replay_report_v1.json"
STATUS = "known_floor_diagnostic_not_method"
SCOPE = "tp034_complete_known_floor_40_heads"
NOISES = ("clean", *map(str, oracle.SEEDS))
SHAPES = tuple(dict(shape) for shape in models.mariner.SHAPES)
CONTRACT = {
    "scope": SCOPE,
    "status": STATUS,
    "worlds": list(oracle.WORLDS),
    "outer_fold": 0,
    "repeat": 0,
    "noise_conditions": list(NOISES),
    "noise_sd": oracle.CONTRACT["noise_sd"],
    "known_floor": oracle.HEAD["floor"],
    "strengths": list(oracle.STRENGTHS),
    "shapes": list(SHAPES),
    "ridges": list(models.mariner.RIDGE_GRID),
    "quadrature": oracle.CONTRACT["quadrature"],
    "heads": 40,
    "contexts": 16,
    "clean_recovery_reference_tolerance_bpb": 1e-8,
    "learned_floor_domain_ready_preserved": False,
    "new_truth_generation": False,
    "raw_optimization": False,
    "empirical_fitting": False,
}


def source_paths() -> set[Path]:
    return oracle.source_paths() | {
        Path(__file__),
        ROOT / "test_tpp40_acquisition_retention_known_floor_20261015.py",
        STAGE1_SEAL,
        STAGE1_AUDIT,
        PARENT / "reviews/floor_nuisance_isolation_proposal_v1.md",
    }


def input_paths() -> set[Path]:
    return oracle.input_paths() | {
        STAGE1 / "registration.json",
        STAGE1 / "execution_identity.json",
        STAGE1 / "complete.json",
        STAGE1 / "oracle_summary.json",
        *(
            STAGE1 / "prepared" / name
            for name in ("clean.npz", "matched_tied.npz", "noise.npz", "preflight.json", "complete.json")
        ),
    }


def verify_stage1(value: dict[str, Any]) -> dict[str, Any]:
    """Require the independently replayed oracle without changing its failed readiness."""
    prerequisite = value["stage1_prerequisite"]
    if benchmark.sha256(STAGE1_SEAL) != prerequisite["manifest_sha256"]:
        raise ValueError("The registered stage-one seal differs")
    if benchmark.sha256(STAGE1_AUDIT) != prerequisite["audit_sha256"]:
        raise ValueError("The registered independent numerical audit differs")
    seal = json.loads(STAGE1_SEAL.read_text())
    if seal["status"] != "numerical_replay_pass_learned_floor_domain_failed" or seal["learned_stage_ready"] is not False:
        raise ValueError("The prerequisite seal must preserve the original numerical and scientific dispositions")
    benchmark.verify_hashes({name: item["sha256"] for name, item in seal["files"].items()}, STAGE1)
    audit = json.loads(STAGE1_AUDIT.read_text())
    summary = json.loads((STAGE1 / "oracle_summary.json").read_text())
    preflight = json.loads((STAGE1 / "prepared/preflight.json").read_text())
    if not audit["passed"] or not audit["oracle_gates_passed"] or not summary["oracle_passed"]:
        raise ValueError("A passing independent numerical replay and conditional oracle are required")
    if audit["replayed_nnls_heads"] != 184 or audit["ode"]["policy_comparisons"] != 1468:
        raise ValueError("The prerequisite numerical replay scope differs")
    if preflight["learned_floor_domain_ready"] is not False or summary["learned_stage_ready"] is not False:
        raise ValueError("Preserve this prerequisite's failed learned-floor readiness")
    oracle.registration()
    return {
        "manifest_sha256": prerequisite["manifest_sha256"],
        "audit_sha256": prerequisite["audit_sha256"],
        "oracle_passed": True,
        "learned_floor_domain_ready": False,
        "learned_stage_ready": False,
    }


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    if value.get("status") != "registered" or value.get("scope") != SCOPE:
        raise ValueError("Register this separate known-floor diagnostic before execution")
    if value.get("contract") != CONTRACT or value.get("environment") != benchmark.environment_identity():
        raise ValueError("The registered complete-search contract or runtime differs")
    for name, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(value[name]) != {str(path.resolve()) for path in paths}:
            raise ValueError("The exact source and input whitelists must be pinned")
        benchmark.verify_hashes(value[name])
    verify_stage1(value)
    models.verify_sources()
    return value


def source_identity() -> dict[str, Any]:
    value = registration()
    return {
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "source_hashes": value["source_hashes"],
        "input_hashes": value["input_hashes"],
        "environment": value["environment"],
        "stage1_prerequisite": value["stage1_prerequisite"],
    }


def freeze() -> None:
    """Freeze provenance without parsing policy arrays or synthetic targets."""
    identity = source_identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Preserve the prior identity; use a separately registered revision")
    benchmark.write_json(path, identity)


def execution_identity() -> dict[str, Any]:
    identity = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != identity:
        raise ValueError("Frozen known-floor identity changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": SCOPE,
        "registration_sha256": identity["registration_sha256"],
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != item for key, item in expected.items()):
        raise ValueError("Exact known-floor diagnostic approval is required")
    return identity


def training_values(clean: np.ndarray, noise: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Apply the preserved full noise vector only to actual asymmetric training rows."""
    return clean[train] + noise[train]


def fit_family(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    config: models.FeatureConfig,
    candidate: str,
    aggregate: models.JointFit | None,
    cache: models.BenefitCache | None,
) -> models.JointFit:
    """Use the exact existing known-floor API, with every native shape and ridge."""
    native = candidate == "aggregate_mariner"
    if candidate not in ("aggregate_mariner", *oracle.CLOCKS) or (native != (aggregate is None)):
        raise ValueError("Fit one shared aggregate before each matching temporal family")
    # A native aggregate has no clock dependence; relabel only its feature provenance.
    control = None if aggregate is None else replace(aggregate, config=config)
    return models.fit_known_floor(
        early,
        late,
        observed,
        labels,
        alpha,
        "aggregate_mariner" if native else models.MODEL,
        floor=CONTRACT["known_floor"],
        config=config,
        strengths=oracle.STRENGTHS,
        cache=cache,
        shapes=SHAPES,
        test_grid=None,
        matched_null_fit=control,
    )


def selected_inner(
    fit: models.JointFit,
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    cache: models.BenefitCache | None,
) -> dict[str, Any]:
    """Save fresh training-local heads for the already selected candidate only."""
    matrix, offset = models.design_and_offset(
        early, late, fit.alpha, fit.shape, fit.model, fit.parameters, fit.config, cache
    )
    if np.any(offset) or fit.known_floor != CONTRACT["known_floor"]:
        raise ValueError("This diagnostic has a zero-offset law and fixed absolute floor")
    predicted, heads = np.empty(len(observed)), []
    for label in np.unique(labels):
        train, held = np.flatnonzero(labels != label), np.flatnonzero(labels == label)
        head = models.mariner.fit_head(
            matrix[train],
            observed[train],
            fit.ridge,
            models.mariner.FloorSpec(CONTRACT["known_floor"], 0.0, 0.0),
        )
        predicted[held] = head.predict(matrix[held])
        heads.append(
            {"label": int(label), "train": train.tolist(), "held": held.tolist(), "head": oracle.head_record(head)}
        )
    score = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    return {
        "inner_heads": heads,
        "inner_predictions": predicted.tolist(),
        "observed": observed.tolist(),
        "inner_rmse": score,
        "selected_score_replay_difference": score - fit.inner_rmse,
    }


def prediction_status(
    fit: models.JointFit, early: np.ndarray, late: np.ndarray, cache: models.BenefitCache
) -> dict[str, Any]:
    matrix, offset = models.design_and_offset(
        early, late, fit.alpha, fit.shape, fit.model, fit.parameters, fit.config, cache
    )
    linear = fit.head.intercept + matrix @ fit.head.coefficients + offset
    return {
        "finite_logits": bool(np.isfinite(linear).all()),
        "native_lower_clip_count": int(np.sum(linear < -models.mariner.LOG_CLIP)),
        "native_upper_clip_count": int(np.sum(linear > models.mariner.LOG_CLIP)),
    }


def run_context(context: tuple[str, str]) -> dict[str, Any]:
    identity = execution_identity()
    world_name, noise_name = context
    world = next(item for item in oracle.WORLDS if item["name"] == world_name)
    if noise_name not in NOISES:
        raise ValueError("Unregistered noise context")
    data, folds = oracle.inputs()
    split = oracle.split_indices(data.rows, folds, 0)
    train, labels = split["train"], split["inner"]
    query0, query1, query = oracle.query_frame(data, split)
    with np.load(STAGE1 / "prepared/clean.npz") as original, np.load(STAGE1 / "prepared/noise.npz") as noise:
        clean = original[world_name]
        observed = training_values(clean, noise[noise_name], train)
    with np.load(STAGE1 / "prepared/matched_tied.npz") as matched:
        truth = np.r_[clean[split["test"]], matched[world_name][split["test"]], clean[split["tied_test"]]]
    cache = models.BenefitCache(OUTPUT / "features.sqlite")
    aggregate, receipts = None, []
    for candidate in oracle.candidates(world):
        identity = execution_identity()
        spec = {"world": world, "noise": noise_name, "candidate": candidate, "outer_fold": 0, "status": STATUS}
        folder = OUTPUT / "cells" / world_name / noise_name / candidate
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
                    fit = fit_family(
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
                    inner = selected_inner(fit, data.early[train], data.late[train], observed, labels, cache)
                    predicted = fit.predict(query0, query1, cache)
                    numerical = prediction_status(fit, query0, query1, cache)
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
            dependencies = (STAGE1_SEAL, STAGE1_AUDIT)
            if candidate != "aggregate_mariner":
                dependencies += (folder.parent / "aggregate_mariner/complete.json",)
            oracle.finish(
                folder, spec, identity, ("fit.json", "inner.json", "scores.json", "predictions.csv"), dependencies
            )
        if candidate == "aggregate_mariner":
            aggregate = fit
        receipts.append(str(folder / "complete.json"))
    return {"world": world_name, "noise": noise_name, "receipts": receipts}


def contexts() -> list[tuple[str, str]]:
    return [(world["name"], noise) for world in oracle.WORLDS for noise in NOISES]


def fit_all(workers: int) -> None:
    execution_identity()
    if workers < 1:
        raise ValueError("Specify a positive local worker count")
    if workers == 1:
        for context in contexts():
            print(json.dumps(run_context(context)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(run_context, contexts()):
                print(json.dumps(result), flush=True)
    summarize()


def summarize() -> None:
    identity = execution_identity()
    records, dependencies = [], []
    for world_name, noise in contexts():
        world = next(item for item in oracle.WORLDS if item["name"] == world_name)
        native = None
        for candidate in oracle.candidates(world):
            folder = OUTPUT / "cells" / world_name / noise / candidate
            spec = {"world": world, "noise": noise, "candidate": candidate, "outer_fold": 0, "status": STATUS}
            if not oracle.completed(folder, spec, identity):
                raise ValueError("Preserve incomplete or failed diagnostic cells")
            scores = json.loads((folder / "scores.json").read_text())
            if candidate == "aggregate_mariner":
                native = scores
            assert native is not None
            records.append(
                {
                    **spec,
                    **scores,
                    "delta_vs_native": {
                        name: scores[name] - native[name]
                        for name in ("asymmetric_rmse", "matched_tied_rmse", "actual_tied_rmse", "gain_rmse")
                    },
                }
            )
            dependencies.append(folder / "complete.json")
    if len(records) != CONTRACT["heads"]:
        raise ValueError("Known-floor coverage differs from the forty-head contract")
    benchmark.write_json(
        OUTPUT / "summary.json",
        {
            "status": STATUS,
            "heads": len(records),
            "records": records,
            "learned_floor_domain_ready": False,
            "scientific_promotion": False,
            "scope": "Supplied absolute floor; full shape/kinetic/ridge selection on saved synthetic outcomes only",
        },
    )
    oracle.finish(OUTPUT, {"stage": "summary", "status": STATUS}, identity, ("summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "fit", "summarize"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "fit":
        fit_all(args.workers)
    else:
        summarize()


if __name__ == "__main__":
    main()
