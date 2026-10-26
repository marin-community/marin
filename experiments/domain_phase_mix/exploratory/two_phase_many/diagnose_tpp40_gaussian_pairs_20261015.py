# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP40-038: finite Gaussian distinguishability of frozen synthetic model pairs.

Saved-head evaluation only: no fit, feature integration, selection, or new query.
The pair is frozen before interpreting fresh Gaussian experiments. These are
existence witnesses, not uncertainty intervals for the realized fitted heads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sqlite3
import sys
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr

BASE = Path(__file__).resolve().parent
PARENT = BASE / "reference_outputs/tpp40_acquisition_retention_20261015"
OUTPUT = PARENT / "gaussian_pair_diagnostic"
STAGE1 = PARENT / "stage1"
KNOWN = PARENT / "known_floor_diagnostic"
ALL_FOLDS = PARENT / "known_floor_all_folds"
RESPONSE = PARENT / "response_profile"
UNCHANGED = PARENT / "unchanged_j2_diagnostic"
HELPER = ALL_FOLDS / "audit/author_replay_v1.py"
HELPER_SHA = "71cab6e615c0fdc8dceb52e8c68322445234dd92f7e7d0b46ee73d5363ad329f"
SCOPE = "tp038_saved_gaussian_pairs_450_heads"
SIGMA = 0.003
WORLDS = ("native_null", "heterogeneous_null", "equal_positive", "pool_positive")
NOISES = ("clean", "20261015", "20261016", "20261017")
ARMS = ("known_floor_log_nnls", "known_floor_response", "learned_nonnegative_floor_response")
SEALS = {
    STAGE1 / "result_manifest.json": "a8c38029614a5df658743f2a551f38b97f44fe7fcc1590a430b3ac91762b6718",
    KNOWN / "result_manifest_v1.json": "8d6c3fafd825b432e76dc270c1f2c3dbb63c068bd1af0d3ef942f8637ae626e0",
    ALL_FOLDS / "result_manifest_v1.json": "3cf46cfc976d35a11993fbffb084d1e4ee7a5d925994a73db76536dfad0300bc",
    RESPONSE / "result_manifest_v1.json": "c97acfbf0434a4a2fd818983d966781718d6b6388b828c73a6492e046783a4ed",
    UNCHANGED / "result_manifest_v1.json": "381233df0f94dae9927a1a1983687b3f1edeefa3b93d149a9770d8ae01d74ec6",
}
CONTRACT = {
    "scope": SCOPE,
    "candidate_id": "TP40-038",
    "worlds": list(WORLDS),
    "outer_folds": list(range(5)),
    "noise_names": list(NOISES),
    "known_floor_heads": 200,
    "response_profile_heads": 240,
    "unchanged_j2_clean_outer_zero_heads": 10,
    "total_heads": 450,
    "common_family_heads": 327,
    "misspecified_heads": 123,
    "gaussian_observation_sd": SIGMA,
    "distance": "L2(candidate full-head training mean - generating training mean) / sigma",
    "equal_prior_bayes_test_error": "Phi(-distance/2)",
    "scalar_minimax_mse_lower_bound": "functional_separation_squared * Phi(-distance/2) / 2",
    "query_roles": ["asymmetric", "matched_tied", "actual_tied"],
    "phase_gain_sign": "matched_tied minus asymmetric",
    "records": "every head, every training row, every query row, every matched gain pair",
    "M_world": "misspecification-only; Gaussian pair distance valid; no within-class risk bound",
    "common_class": "fixed clock; each function has one bucket-shared shape, which may differ across hypotheses",
    "unchanged_j2_floor_domain": "nonnegative absolute floor, independent of the data-dependent gamma fitting rule",
    "aggregate_positive_control": "embed mu=0 in the positive truth's matching clock family",
    "analytic_eligibility": "common family and zero evaluated clipping, exponent underflow, or nonfinite means",
    "native_clipping": "finite affected pairs keep numerical Gaussian distances but receive no analytic-class bound",
    "selection_dependence": "freeze the selected pair; testing formula concerns fresh repeated Gaussian observations",
    "independent_replicates": False,
    "original_stage2_remains_blocked": True,
    "scientific_promotion": False,
    "new_fits": False,
    "integration": False,
    "raw_optimization": False,
    "empirical_targets": False,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_helpers() -> Any:
    if digest(HELPER) != HELPER_SHA:
        raise ValueError("The sealed metadata/cache helper changed")
    spec = importlib.util.spec_from_file_location("_tp038_sealed_readonly_helpers", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the sealed helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helpers()


def write(path: Path, value: Any) -> None:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Preserve the previous artifact: {path}")
    path.write_text(text)


def runtime() -> dict[str, str]:
    return {"python": sys.version, "sqlite": sqlite3.sqlite_version} | {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "threadpoolctl")
    }


def source_paths() -> set[Path]:
    return {
        Path(__file__),
        BASE / "test_tpp40_gaussian_pairs_20261015.py",
        HELPER,
        helper.HELPER,
        OUTPUT / "proposal_v1.md",
    } | {Path(path) for path in read(ALL_FOLDS / "registration.json")["source_hashes"]}


def input_paths() -> set[Path]:
    return (
        set(SEALS)
        | {
            KNOWN / "features.sqlite",
            ALL_FOLDS / "features.sqlite",
            UNCHANGED / "features.sqlite",
            STAGE1 / "prepared/clean.npz",
            STAGE1 / "prepared/noise.npz",
            STAGE1 / "prepared/matched_tied.npz",
        }
        | {helper.DATA / name for name in ("metadata.json", "policies.npz", "rows.csv", "folds.csv")}
    )


def identity() -> dict[str, Any]:
    registration = read(OUTPUT / "registration.json")
    if registration.get("status") != "registered" or registration.get("contract") != CONTRACT:
        raise ValueError("Register the exact saved-pair scope before execution")
    if registration["runtime"] != runtime():
        raise ValueError("The registered runtime changed")
    for field, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(registration[field]) != {str(path) for path in paths}:
            raise ValueError("The exact provenance whitelist changed")
        for path, expected in registration[field].items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Pinned input changed: {path}")
    for path, expected in SEALS.items():
        if digest(path) != expected:
            raise ValueError("A prerequisite seal changed")
        for name, item in read(path)["files"].items():
            if digest(path.parent / name) != item["sha256"]:
                raise ValueError(f"Sealed artifact changed: {name}")
    if read(STAGE1 / "result_manifest.json")["learned_stage_ready"]:
        raise ValueError("Preserve the failed learned-stage prerequisite")
    preflight = read(STAGE1 / "prepared/preflight.json")
    if not preflight["oracle_ready"]:
        raise ValueError("Generating means require the sealed finite, unclipped, positive-deficit prerequisite")
    for truth in preflight["truths"].values():
        if truth["native_prediction_clip_count"] or truth["matched_tied_prediction_clip_count"]:
            raise ValueError("The generating truth must be unclipped on every registered policy and tied query")
    return {"registration_sha256": digest(OUTPUT / "registration.json")} | {
        key: registration[key] for key in ("source_hashes", "input_hashes", "runtime")
    }


def freeze() -> None:
    write(OUTPUT / "execution_identity.json", identity())


def execution_identity() -> dict[str, Any]:
    current = identity()
    if current != read(OUTPUT / "execution_identity.json"):
        raise ValueError("Frozen identity differs")
    expected = {
        "scope": SCOPE,
        "registration_sha256": current["registration_sha256"],
        "execution_identity_sha256": digest(OUTPUT / "execution_identity.json"),
    }
    approval = read(OUTPUT / "execution_approval.json")
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Require exact saved-output diagnostic approval")
    return current


def gaussian_pair(candidate: np.ndarray, truth: np.ndarray, sigma: float) -> dict[str, float]:
    """Return exact equal-prior testing difficulty for two finite Gaussian means."""
    candidate, truth = np.asarray(candidate, float), np.asarray(truth, float)
    if candidate.shape != truth.shape or candidate.ndim != 1 or not len(candidate):
        raise ValueError("Two nonempty training mean vectors must share their row order")
    if not np.isfinite(candidate).all() or not np.isfinite(truth).all() or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Finite means and a positive observation scale are required")
    residual = candidate - truth
    distance = float(np.linalg.norm(residual) / sigma)
    if not math.isfinite(distance):
        raise ValueError("Gaussian distance overflowed")
    return {
        "distance": distance,
        "bayes_error": float(ndtr(-distance / 2)),
        "log_bayes_error": float(log_ndtr(-distance / 2)),
        "training_mean_rmse": float(np.sqrt(np.mean(residual**2))),
        "training_mean_max_abs_separation": float(np.max(abs(residual))),
    }


def functional_rows(frame: pd.DataFrame, pair: dict[str, float], eligible: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain each endpoint and each matched phase-gain separation, with signs."""
    endpoint = frame.copy()
    endpoint["signed_separation"] = endpoint.predicted - endpoint.truth
    endpoint["absolute_separation"] = abs(endpoint.signed_separation)
    first, tied = frame.loc[frame.role == "asymmetric"], frame.loc[frame.role == "matched_tied"]
    if first[["source_index", "row_id"]].to_dict("list") != tied[["source_index", "row_id"]].to_dict("list"):
        raise ValueError("Every gain must pair the same aggregate's saved asymmetric and tied rows")
    gain = first[["source_index", "row_id"]].reset_index(drop=True).copy()
    gain["candidate_gain"] = tied.predicted.to_numpy() - first.predicted.to_numpy()
    gain["true_gain"] = tied.truth.to_numpy() - first.truth.to_numpy()
    gain["signed_separation"] = gain.candidate_gain - gain.true_gain
    gain["absolute_separation"] = abs(gain.signed_separation)
    for table in (endpoint, gain):
        delta = table.absolute_separation.to_numpy(float)
        nonzero = delta > 0
        log_bound = np.full(len(delta), np.nan)
        bound = np.full(len(delta), np.nan)
        if eligible:
            log_bound[nonzero] = 2 * np.log(delta[nonzero]) - math.log(2) + pair["log_bayes_error"]
            bound[nonzero], bound[~nonzero] = np.exp(log_bound[nonzero]), 0.0
            if pair["distance"] == 0:
                bound[nonzero] = (delta[nonzero] / 2) ** 2
        table["pair_mse_lower_bound"] = bound
        table["pair_mse_log_lower_bound"] = log_bound
        table["zero_functional_separation"] = ~nonzero
        table["bound_underflowed"] = (bound == 0) & nonzero & eligible
    return endpoint, gain


def candidate_names(world: str) -> tuple[str, ...]:
    return (
        ("aggregate_mariner", "equal_group", "global_pool")
        if world.endswith("null")
        else ("aggregate_mariner", "equal_group" if world == "equal_positive" else "global_pool")
    )


def known_folder(fold: int, world: str, noise: str, candidate: str) -> Path:
    return (KNOWN / "cells" if fold == 0 else ALL_FOLDS / "cells" / f"outer{fold}") / world / noise / candidate


def head_prediction(matrix: np.ndarray, head: dict[str, Any]) -> tuple[np.ndarray, int]:
    coefficients = np.asarray(head["coefficients"], float)
    if not np.isfinite(coefficients).all() or np.any(coefficients < 0) or not math.isfinite(head["intercept"]):
        raise ValueError("The fixed head violates its coefficient/intercept class")
    logits = head["intercept"] + matrix @ coefficients
    return head["floor"] + np.exp(np.clip(logits, -30, 30)), int(np.sum(abs(logits) > 30))


def membership(
    world: str,
    candidate: str,
    response_arm: str | None,
    floor: float,
    clips: int,
    source: str = "known_floor_search",
    underflows: int = 0,
) -> dict[str, Any]:
    """Define a common predictive class independently of any fitting rule."""
    learned = response_arm == "learned_nonnegative_floor_response" or source == "unchanged_j2"
    if not math.isfinite(floor) or (floor < 0 if learned else floor != 0.6):
        raise ValueError("Saved floor is outside the declared common predictive class")
    if world.endswith("positive"):
        clock = "equal_group" if world == "equal_positive" else "global_pool"
    else:
        clock = "native" if candidate == "aggregate_mariner" else candidate
    return {
        "common_family_member": world != "heterogeneous_null",
        "analytic_pair_eligible": world != "heterogeneous_null" and clips == 0 and underflows == 0,
        "common_clock": clock,
        "aggregate_embedded_as_zero_mu": world.endswith("positive") and candidate == "aggregate_mariner",
        "common_floor_domain": "nonnegative" if learned else "fixed_0.6",
        "common_scale_domain": "nonnegative" if response_arm and response_arm != ARMS[0] else "positive",
        "numerical_prediction_class": (
            "unclipped_response" if response_arm and response_arm != ARMS[0] else "native_clipped"
        ),
        "evaluated_unclipped_domain": clips == 0 and underflows == 0,
        "within_class_exclusion": "shared-shape model misspecification" if world == "heterogeneous_null" else None,
    }


def response_prediction(matrix: np.ndarray, selected: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Replay the frozen normalized response and count actual exponent underflow."""
    if selected["scale"] == 0:
        return np.full(len(matrix), selected["floor"]), 0
    log_response = math.log(selected["scale"]) + (
        matrix @ np.asarray(selected["amplitudes"]) - selected["normalization"]
    )
    response = np.exp(log_response)
    if not np.isfinite(response).all():
        raise ValueError("The saved response produces a nonfinite evaluated mean")
    return selected["floor"] + response, int(np.sum(response == 0))


def original_numerical_status(source: str, scores: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Retain original numerical warnings independently of predictive-class membership."""
    if source == "response_profile":
        return scores["unresolved"], {"unresolved": scores["unresolved"]}
    if source == "unchanged_j2":
        names = (
            "selected_from_failed_refinement",
            "failed_refinements",
            "query_clipping_count",
            "training_deficit_clip_count",
            "inner_training_deficit_clip_count",
            "inner_prediction_clip_count",
        )
        flags = {name: scores[name] for name in names}
        return any(bool(value) for value in flags.values()), flags
    flags = scores["numerical"]
    return (
        bool(not flags["finite_logits"] or flags["native_lower_clip_count"] or flags["native_upper_clip_count"]),
        flags,
    )


def geometry() -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    meta = read(helper.DATA / "metadata.json")
    if meta["buckets"] != ["nemotron_full", "starcoder"]:
        raise ValueError("Frozen bucket order differs")
    with np.load(helper.DATA / "policies.npz", allow_pickle=False) as archive:
        if set(archive.files) != helper.base.POLICY_KEYS:
            raise ValueError("Only the policy-array whitelist is admissible")
        first, last = archive["phase0_epochs"].copy(), archive["phase1_epochs"].copy()
        scales = archive["c0"].copy() + archive["c1"].copy()
    rows = pd.read_csv(helper.DATA / "rows.csv", usecols=list(helper.base.ROW_COLUMNS))
    folds = pd.read_csv(helper.DATA / "folds.csv", usecols=list(helper.base.FOLD_COLUMNS))
    check = helper.base.Checks()
    return {
        fold: helper.fold_geometry(rows, folds, first, last, meta["alpha"], scales, fold, check) for fold in range(5)
    }, rows


def validate_query(frame: pd.DataFrame, data: dict[str, Any], clean: np.ndarray, tied: np.ndarray) -> None:
    if frame[list(data["query"].columns)].to_dict("list") != data["query"].to_dict("list"):
        raise ValueError("Saved query row identities differ")
    truth = np.r_[clean[data["split"]["test"]], tied[data["split"]["test"]], clean[data["split"]["tied_test"]]]
    np.testing.assert_allclose(frame.truth, truth, atol=2e-15, rtol=0)
    if not np.isfinite(frame.predicted).all():
        raise ValueError("A saved query prediction is nonfinite")


def run() -> None:
    frozen = execution_identity()
    complete = OUTPUT / "complete.json"
    if complete.exists():
        prior = read(complete)
        if prior["identity"] != frozen or any(digest(OUTPUT / p) != sha for p, sha in prior["outputs"].items()):
            raise ValueError("Preserve the prior diagnostic output")
        return
    data, rows = geometry()
    heads, training_rows, endpoints, gains = [], [], [], []
    check = helper.base.Checks()
    caches = {root: root / "features.sqlite" for root in (KNOWN, ALL_FOLDS, UNCHANGED)}
    cache_hashes = {str(path): digest(path) for path in caches.values()}
    if any(Path(str(path) + suffix).exists() for path in caches.values() for suffix in ("-wal", "-shm")):
        raise ValueError("All feature caches must be closed before immutable reads")
    with ExitStack() as stack:
        clean = stack.enter_context(np.load(STAGE1 / "prepared/clean.npz", allow_pickle=False))
        tied = stack.enter_context(np.load(STAGE1 / "prepared/matched_tied.npz", allow_pickle=False))
        noise_arrays = stack.enter_context(np.load(STAGE1 / "prepared/noise.npz", allow_pickle=False))
        connections = {
            root: stack.enter_context(closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)))
            for root, path in caches.items()
        }
        for fold in range(5):
            context, train = data[fold], data[fold]["split"]["train"]
            root = KNOWN if fold == 0 else ALL_FOLDS
            cache = helper.base.Cache(connections[root], check, read(root / "registration.json"), context)
            unchanged_cache = (
                helper.base.Cache(connections[UNCHANGED], check, read(UNCHANGED / "registration.json"), context)
                if fold == 0
                else None
            )
            for world in WORLDS:
                truth_train = clean[world][train]
                for noise in NOISES:
                    sources = [("known_floor_search", candidate_names(world)), ("response_profile", ARMS)]
                    if fold == 0 and noise == "clean":
                        sources.append(("unchanged_j2", candidate_names(world)))
                    for source, candidates in sources:
                        for candidate in candidates:
                            common = {
                                "source": source,
                                "outer_fold": fold,
                                "world": world,
                                "noise": noise,
                                "candidate": candidate,
                            }
                            pair_id = "__".join(str(common[key]) for key in common)
                            if source != "response_profile":
                                if source == "known_floor_search":
                                    folder = known_folder(fold, world, noise, candidate)
                                    fit, inner = read(folder / "fit.json"), read(folder / "inner.json")
                                    np.testing.assert_allclose(
                                        inner["observed"], truth_train + noise_arrays[noise][train], atol=2e-15, rtol=0
                                    )
                                    cell_cache = cache
                                else:
                                    folder = UNCHANGED / "cells" / world / candidate
                                    fit = read(folder / "model.json")
                                    inner_frame = pd.read_csv(
                                        folder / "inner_predictions.csv", float_precision="round_trip"
                                    )
                                    if inner_frame.source_index.tolist() != train.tolist():
                                        raise ValueError("Unchanged J2 training row identities differ")
                                    np.testing.assert_allclose(inner_frame.truth, truth_train, atol=2e-15, rtol=0)
                                    cell_cache = unchanged_cache
                                    assert cell_cache is not None
                                if (
                                    fit["shape"] not in helper.base.SHAPES
                                    or fit["parameters"].get("mu", 0) not in helper.base.STRENGTHS
                                ):
                                    raise ValueError("Saved shape or kinetics is outside the frozen class")
                                matrix = cell_cache.matrix(context["first"][train], context["last"][train], fit)
                                predicted_train, train_clips = head_prediction(matrix, fit["head"])
                                query_matrix = cell_cache.matrix(context["query0"], context["query1"], fit)
                                predicted_query, query_clips = head_prediction(query_matrix, fit["head"])
                                arm, floor = None, fit["head"]["floor"]
                                total_clips, underflow = train_clips + query_clips, 0
                                training_response_underflow, query_response_underflow = 0, 0
                            else:
                                folder = RESPONSE / "cells" / f"outer{fold}" / noise / world / candidate
                                fit = read(folder / "fit.json")
                                saved_train = pd.read_csv(folder / "training.csv", float_precision="round_trip")
                                if (
                                    saved_train.source_index.tolist() != train.tolist()
                                    or saved_train.row_id.tolist() != rows.iloc[train].row_id.tolist()
                                ):
                                    raise ValueError("Response head training row identities differ")
                                np.testing.assert_allclose(
                                    saved_train.observed, truth_train + noise_arrays[noise][train], atol=2e-15, rtol=0
                                )
                                predicted_train = saved_train.predicted.to_numpy(float)
                                arm = candidate
                                if candidate == ARMS[0]:
                                    head = fit["profile"]["head"]
                                    coefficients, floor = np.asarray(head["coefficients"]), head["floor"]
                                    total_clips, underflow = fit["native_exponential_clipped_count"], 0
                                    training_response_underflow, query_response_underflow = 0, 0
                                    predicted_query = None
                                else:
                                    selected = fit["selected"]
                                    coefficients, floor = np.asarray(selected["amplitudes"]), selected["floor"]
                                    if not math.isfinite(selected["scale"]) or selected["scale"] < 0:
                                        raise ValueError("Response scale violates the nonnegative class")
                                    if not math.isfinite(selected["normalization"]):
                                        raise ValueError("The normalized response head is nonfinite")
                                    total_clips, underflow = 0, selected["training_exponential_underflow_count"]
                                    initialization = read(folder.parent / "initialization/starts.json")
                                    replayed_train, training_response_underflow = response_prediction(
                                        np.asarray(initialization["matrix"]), selected
                                    )
                                    np.testing.assert_allclose(replayed_train, predicted_train, atol=2e-12, rtol=2e-13)
                                    query_descriptor = {
                                        "shape": helper.base.TRUTH_SHAPE,
                                        "alpha": context["alpha"],
                                        "parameters": {"mu": 3.0 if world.endswith("positive") else 0.0},
                                        "feature_config": {
                                            "clock": "global_pool" if world == "pool_positive" else "equal_group"
                                        },
                                    }
                                    response_query_matrix = cache.matrix(
                                        context["query0"], context["query1"], query_descriptor
                                    )
                                    predicted_query, query_response_underflow = response_prediction(
                                        response_query_matrix, selected
                                    )
                                if not np.isfinite(coefficients).all() or np.any(coefficients < 0):
                                    raise ValueError("Response coefficients violate the class")
                                train_clips, query_clips = None, None
                            frame = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
                            unresolved, original_status = original_numerical_status(source, read(folder / "scores.json"))
                            validate_query(frame, context, clean[world], tied[world])
                            if predicted_query is not None:
                                np.testing.assert_allclose(frame.predicted, predicted_query, atol=2e-12, rtol=2e-13)
                            pair = gaussian_pair(predicted_train, truth_train, SIGMA)
                            member_candidate = (
                                candidate
                                if source != "response_profile"
                                else (
                                    "equal_group"
                                    if world == "equal_positive"
                                    else "global_pool"
                                    if world == "pool_positive"
                                    else "aggregate_mariner"
                                )
                            )
                            member = membership(
                                world,
                                member_candidate,
                                arm,
                                floor,
                                total_clips,
                                source,
                                underflow + training_response_underflow + query_response_underflow,
                            )
                            head_record = (
                                common
                                | pair
                                | member
                                | {
                                    "pair_id": pair_id,
                                    "train_rows": len(train),
                                    "query_rows": len(frame),
                                    "floor": floor,
                                    "selected_fit_unresolved": unresolved,
                                    "original_numerical_status": json.dumps(original_status, sort_keys=True),
                                    "training_clip_count": train_clips,
                                    "query_clip_count": query_clips,
                                    "total_train_query_clip_count": total_clips,
                                    "training_underflow_count": underflow,
                                    "training_response_underflow_count": training_response_underflow,
                                    "query_response_underflow_count": query_response_underflow,
                                }
                            )
                            heads.append(head_record)
                            training_rows.append(
                                pd.DataFrame(
                                    {
                                        "source_index": train,
                                        "row_id": rows.iloc[train].row_id.to_numpy(),
                                        "truth": truth_train,
                                        "candidate_mean": predicted_train,
                                        "signed_separation": predicted_train - truth_train,
                                    }
                                ).assign(pair_id=pair_id)
                            )
                            endpoint, gain = functional_rows(frame, pair, member["analytic_pair_eligible"])
                            extra = {
                                "pair_id": pair_id,
                                "distance": pair["distance"],
                                "bayes_error": pair["bayes_error"],
                                "common_family_member": member["common_family_member"],
                                "analytic_pair_eligible": member["analytic_pair_eligible"],
                            }
                            endpoints.append(endpoint.assign(**extra))
                            gains.append(gain.assign(**extra))
    if len(heads) != 450 or len({row["pair_id"] for row in heads}) != 450:
        raise ValueError("All 450 unique heads must remain represented")
    if sum(row["common_family_member"] for row in heads) != 327:
        raise ValueError("Preserve exactly 327 common-family and 123 misspecified entries")
    for path, sha in cache_hashes.items():
        if digest(Path(path)) != sha:
            raise ValueError("A read-only cache changed")
    tables = {
        "heads.csv": pd.DataFrame(heads),
        "training_rows.csv": pd.concat(training_rows, ignore_index=True),
        "endpoints.csv": pd.concat(endpoints, ignore_index=True),
        "gains.csv": pd.concat(gains, ignore_index=True),
    }
    for name, frame in tables.items():
        path, text = OUTPUT / name, frame.to_csv(index=False)
        if path.exists() and path.read_text() != text:
            raise ValueError("Preserve the prior partial output")
        path.write_text(text)
    write(
        OUTPUT / "summary.json",
        {
            "heads": len(heads),
            "common_family_heads": sum(row["common_family_member"] for row in heads),
            "analytic_eligible_heads": sum(row["analytic_pair_eligible"] for row in heads),
            "misspecification_heads": sum(not row["common_family_member"] for row in heads),
            "contract": CONTRACT,
            "interpretation": (
                "A frozen pair witness for repeated Gaussian experiments, not realized-fit uncertainty or policy regret."
            ),
            "unresolved_heads_preserved": sum(row["selected_fit_unresolved"] for row in heads),
            "cache_hashes": cache_hashes,
            "cache_checks": check.counts,
        },
    )
    write(complete, {"identity": frozen, "outputs": {name: digest(OUTPUT / name) for name in (*tables, "summary.json")}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
