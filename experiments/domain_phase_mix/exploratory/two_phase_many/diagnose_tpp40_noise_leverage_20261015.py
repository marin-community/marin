# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Describe saved fixed-law noise sensitivity without refitting any parameters."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PARENT = Path(__file__).resolve().parent / "reference_outputs/tpp40_acquisition_retention_20261015"
OUTPUT = PARENT / "noise_leverage"
EXTENSION = PARENT / "oracle_extension"
INSPECTOR = EXTENSION / "audit/replay_extension_v1.py"
INSPECTOR_SHA = "e4f716be9c764c84442081927270b1efa9b161bd85c0204aaaef59d2e3f299d4"
BASE = PARENT / "stage1/audit/independent_replay_v1.py"
BASE_SHA = "de4e44abbf5e2c42ade95b09413b54a2812d8d823b841bce145d1e8487e72ec4"
GEOMETRY = PARENT / "floor_geometry"
GEOMETRY_SHA = "08ea367e9949199bec74b5b1f16c7b5afdd4163906ec5179dc5a72e4e9fe1daf"
SIGMA = 0.003
FLOOR = 0.6
CONTRACT: dict[str, Any] = {
    "scope": "fixed_law_saved_profile_noise_leverage_no_fits",
    "worlds": [
        ["native_null", "aggregate_mariner", "equal_group", 0.0],
        ["equal_positive", "equal_group", "equal_group", 3.0],
        ["pool_positive", "global_pool", "global_pool", 3.0],
    ],
    "outer_folds": [0, 1, 2, 3, 4],
    "seeds": [20261015, 20261016, 20261017],
    "noise_sd": SIGMA,
    "floor": FLOOR,
    "shape": {"rate": 0.25, "power": 0.7, "threshold": 3.0},
    "faces": ["unrestricted_tangent", "saved_noisy_active_face"],
    "face_expansion": "project clean training logits into saved active span; retain query approximation offset",
    "fisher_scope": "unrestricted full head at the true response, independent of the retrospective active face",
    "clamp_policy": "fail if any noisy training deficit is at or below 1e-9",
    "active_rule": "intercept plus strictly positive saved NNLS coefficients; exact zero is inactive",
    "svd_rank_rule": "s > eps * max(n,p) * largest_s, after column scaling",
    "gradient_identification_rule": "scaled null residual <= 64*eps*max(n,p)*scaled gradient norm",
    "m_cosine": {
        "residual": "heterogeneous_null minus native_null BPB truth",
        "clocks": ["native_equal", "native_pool"],
        "tangent": "derivatives_0.0005 column 3",
        "nuisance": ["native amplitudes plus three shape tangents", "same plus floor"],
        "subsets": ["outer_train", "inner_train_0", "inner_train_1", "inner_train_2"],
        "rows": 80,
    },
    "expected_contexts": 45,
    "expected_query_and_gain_rows": 17964,
    "new_fits": 0,
    "new_integrations": 0,
    "new_policy_optimizations": 0,
    "scientific_gate": None,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def svd_geometry(matrix: np.ndarray) -> dict[str, Any]:
    """Factor a column-scaled design without forming its normal matrix."""
    matrix = np.asarray(matrix, float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or not matrix.size:
        raise ValueError("A finite nonempty design is required")
    scales = np.linalg.norm(matrix, axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    left, singular, right = np.linalg.svd(matrix / scales, full_matrices=False)
    cutoff = np.finfo(float).eps * max(matrix.shape) * singular[0]
    rank = int(np.sum(singular > cutoff))
    return {
        "left": left[:, :rank],
        "right": right[:rank],
        "singular": singular,
        "scales": scales,
        "rank": rank,
        "cutoff": cutoff,
        "shape": matrix.shape,
    }


def response_map(matrix: np.ndarray, gradients: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a linear response plus explicit scaled-gauge identification flags."""
    geometry = svd_geometry(matrix)
    scaled = np.asarray(gradients, float) / geometry["scales"]
    coordinates = scaled @ geometry["right"].T
    residual = scaled - coordinates @ geometry["right"]
    norm, gap = np.linalg.norm(scaled, axis=1), np.linalg.norm(residual, axis=1)
    tolerance = 64 * np.finfo(float).eps * max(matrix.shape) * norm
    identified = gap <= tolerance
    # An exactly zero gradient is an exact identity, including a paired null gain.
    identified[norm == 0], gap[norm == 0] = True, 0.0
    mapping = (coordinates / geometry["singular"][: geometry["rank"]]) @ geometry["left"].T
    return mapping, {
        "rank": geometry["rank"],
        "columns": matrix.shape[1],
        "singular_values": geometry["singular"].tolist(),
        "column_scales": geometry["scales"].tolist(),
        "rank_cutoff": geometry["cutoff"],
        "information_gap": gap,
        "relative_information_gap": np.divide(gap, norm, out=np.zeros_like(gap), where=norm > 0),
        "identified": identified,
    }


def noise_response(
    train: np.ndarray,
    query: np.ndarray,
    train_deficit: np.ndarray,
    query_deficit: np.ndarray,
    noise: np.ndarray,
    active: np.ndarray,
    paired: int,
    exact_null: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Calculate local full/active-face response; this does not solve NNLS."""
    if np.any(train_deficit <= 0) or np.any(query_deficit <= 0):
        raise ValueError("The known-floor clean response must have positive deficits")
    clamp_count = int(np.sum(train_deficit + noise <= 1e-9))
    if clamp_count:
        raise ValueError(f"Admitted oracle scope requires no deficit clamp: {clamp_count} rows")
    full_train, full_query = train, query
    fisher_map, fisher_info = response_map(
        train_deficit[:, None] * full_train / SIGMA, query_deficit[:, None] * full_query
    )
    train, query = train[:, active], query[:, active]
    mapping, info = response_map(train, query)
    face_logit = mapping @ np.log(train_deficit)
    offset = face_logit - np.log(query_deficit)
    face_deficit = np.exp(face_logit)
    approximation_bias = face_deficit - query_deficit
    ratio = noise / train_deficit
    exact_log = np.log(train_deficit + noise) - np.log(train_deficit)
    first = mapping @ ratio
    realized_second = mapping @ (-0.5 * ratio**2)
    jensen = mapping @ (-0.5 * (SIGMA / train_deficit) ** 2)
    exact = mapping @ exact_log
    eta_sd = np.linalg.norm(SIGMA * mapping / train_deficit, axis=1)
    bpb_map = face_deficit[:, None] * mapping
    values = {
        "face_clean_logit_offset": offset,
        "face_clean_approximation_bias_bpb": approximation_bias,
        "first_order_bpb": face_deficit * first,
        "first_order_total_error_bpb": approximation_bias + face_deficit * first,
        "first_order_exp_mapped_bpb": face_deficit * np.expm1(first),
        "second_order_exp_mapped_bpb": face_deficit * np.expm1(first + realized_second),
        "exact_log_exp_mapped_bpb": face_deficit * np.expm1(exact),
        "exact_face_error_bpb": approximation_bias + face_deficit * np.expm1(exact),
        "propagated_log_jensen_bpb": face_deficit * jensen,
        "response_curvature_bias_bpb": 0.5 * face_deficit * eta_sd**2,
        "expected_second_order_bpb_bias": face_deficit * (jensen + 0.5 * eta_sd**2),
        "expected_total_error_bpb": approximation_bias + face_deficit * (jensen + 0.5 * eta_sd**2),
        "log_estimator_sd": face_deficit * eta_sd,
        "fisher_sd": np.linalg.norm(fisher_map, axis=1),
        "log_information_gap": info["information_gap"],
        "fisher_information_gap": fisher_info["information_gap"],
        "log_identified": info["identified"],
        "fisher_identified": fisher_info["identified"],
    }
    gain_map = bpb_map[paired : 2 * paired] - bpb_map[:paired]
    gain_gradient = (
        query_deficit[paired : 2 * paired, None] * full_query[paired : 2 * paired]
        - query_deficit[:paired, None] * full_query[:paired]
    )
    gain_face_gradient = (
        face_deficit[paired : 2 * paired, None] * query[paired : 2 * paired]
        - face_deficit[:paired, None] * query[:paired]
    )
    if exact_null:
        gain_map, gain_gradient = np.zeros_like(gain_map), np.zeros_like(gain_gradient)
        gain_face_gradient = np.zeros_like(gain_face_gradient)
    gain_fisher, gain_info = response_map(train_deficit[:, None] * full_train / SIGMA, gain_gradient)
    # The log estimator's training-space information gap uses the BPB gain gradient.
    _, gain_log_info = response_map(train, gain_face_gradient)
    for name in list(values):
        if name in ("log_estimator_sd", "fisher_sd"):
            gain = (
                np.linalg.norm(SIGMA * gain_map / train_deficit, axis=1)
                if name == "log_estimator_sd"
                else np.linalg.norm(gain_fisher, axis=1)
            )
        elif name.endswith("information_gap"):
            gain = (gain_log_info if name.startswith("log") else gain_info)["information_gap"]
        elif name.endswith("identified"):
            gain = (gain_log_info if name.startswith("log") else gain_info)["identified"]
        else:
            gain = values[name][paired : 2 * paired] - values[name][:paired]
            if exact_null:
                gain = np.zeros_like(gain)
        values[name] = np.r_[values[name], gain]
    # Unidentified projected uncertainties are not reported as finite information bounds.
    values["log_estimator_sd"] = np.where(values["log_identified"], values["log_estimator_sd"], np.nan)
    values["fisher_sd"] = np.where(values["fisher_identified"], values["fisher_sd"], np.nan)
    details = {
        "log_design": {k: v for k, v in info.items() if not isinstance(v, np.ndarray)},
        "fisher_design": {k: v for k, v in fisher_info.items() if not isinstance(v, np.ndarray)},
        "fisher_scope": CONTRACT["fisher_scope"],
        "deficit_clamp_count": clamp_count,
        "maximum_absolute_noise_over_deficit": float(np.max(abs(ratio))),
        "active_indices": np.flatnonzero(active).tolist(),
    }
    return values, details


def residual_cosine(nuisance: np.ndarray, residual: np.ndarray, tangent: np.ndarray) -> dict[str, Any]:
    geometry = svd_geometry(nuisance)
    basis = geometry["left"]
    first, last = residual - basis @ (basis.T @ residual), tangent - basis @ (basis.T @ tangent)
    norms = np.linalg.norm(first), np.linalg.norm(last)
    return {
        "nuisance_rank": geometry["rank"],
        "nuisance_columns": nuisance.shape[1],
        "scaled_singular_values": geometry["singular"].tolist(),
        "raw_misspec_norm": float(np.linalg.norm(residual)),
        "raw_tangent_norm": float(np.linalg.norm(tangent)),
        "residual_misspec_norm": float(norms[0]),
        "residual_tangent_norm": float(norms[1]),
        "cosine": float(first @ last / np.prod(norms)) if min(norms) > 0 else None,
    }


def source_paths() -> set[Path]:
    return {
        Path(__file__).resolve(),
        Path(__file__).with_name("test_tpp40_noise_leverage_20261015.py").resolve(),
        INSPECTOR,
        BASE,
    }


def adopt() -> tuple[dict[str, Any], Any]:
    registration = read(OUTPUT / "registration.json")
    if registration["status"] != "registered" or registration["contract"] != CONTRACT:
        raise ValueError("The exact no-fit diagnostic must be registered before reading arrays")
    if set(registration["source_hashes"]) != {str(p) for p in source_paths()}:
        raise ValueError("Source pin scope changed")
    environment = {"python": sys.version} | {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")
    }
    if environment != registration["environment"]:
        raise ValueError("Registered runtime changed")
    expected = {
        str(EXTENSION / "result_manifest_v1.json"),
        str(GEOMETRY / "result_manifest_v1.json"),
        str(PARENT / "stage1/result_manifest.json"),
    }
    if set(registration["prerequisite_hashes"]) != expected:
        raise ValueError("Prerequisite pin scope changed")
    for mapping in (registration["source_hashes"], registration["prerequisite_hashes"]):
        for path, sha in mapping.items():
            if not sha or digest(Path(path)) != sha:
                raise ValueError(f"Source or seal changed: {path}")
    if (
        digest(INSPECTOR) != INSPECTOR_SHA
        or digest(BASE) != BASE_SHA
        or digest(GEOMETRY / "result_manifest_v1.json") != GEOMETRY_SHA
    ):
        raise ValueError("Adopted independent inspector or tangent geometry changed")
    for manifest in expected:
        for name, record in read(Path(manifest))["files"].items():
            if digest(Path(manifest).parent / name) != record["sha256"]:
                raise ValueError(f"Sealed input changed: {name}")
    approval = read(OUTPUT / "execution_approval.json")
    if approval != {"scope": CONTRACT["scope"], "registration_sha256": digest(OUTPUT / "registration.json")}:
        raise ValueError("Exact registration approval is required")
    wal = Path(str(EXTENSION / "features.sqlite") + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Require the closed, checkpointed feature cache with no pending WAL content")
    spec = importlib.util.spec_from_file_location("noise_leverage_readonly_inspector", INSPECTOR)
    assert spec is not None and spec.loader is not None
    inspector = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = inspector
    spec.loader.exec_module(inspector)
    return registration, inspector


def cached_benefits() -> dict[tuple[str, float, str], np.ndarray]:
    result = {}
    with closing(sqlite3.connect(f"file:{EXTENSION / 'features.sqlite'}?mode=ro&immutable=1", uri=True)) as connection:
        contexts = {key: json.loads(body) for key, body in connection.execute("SELECT key,identity FROM contexts")}
        for context, policy, response, sha in connection.execute(
            "SELECT context,policy,response,response_sha256 FROM features"
        ):
            if hashlib.sha256(response.encode()).hexdigest() != sha:
                raise ValueError("Cached response hash changed")
            value, identity = json.loads(response), contexts[context]
            if value["status"] != "complete":
                raise ValueError("Saved feature failure cannot be suppressed")
            result[(identity["config"]["clock"], identity["mu"], policy)] = np.asarray(value["benefit"])
    return result


def design(early: np.ndarray, late: np.ndarray, clock: str, mu: float, alpha: float, cache: dict) -> np.ndarray:
    benefits = []
    for first, last in zip(early, late, strict=True):
        total = first + last
        native = -np.expm1(-((0.25 * total) ** 0.7))
        a = np.divide(first, total, out=np.full_like(total, alpha), where=total > 0)
        b = np.divide(last, total, out=np.full_like(total, 1 - alpha), where=total > 0)
        tol = 64 * np.finfo(float).eps
        tied = np.all(abs(a - alpha) <= tol * alpha) and np.all(abs(b - (1 - alpha)) <= tol * (1 - alpha))
        policy = hashlib.sha256(
            json.dumps({"early": first.tolist(), "late": last.tolist()}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        benefits.append(native if mu == 0 or tied else cache[(clock, mu, policy)])
    harm = np.logaddexp(0, np.log1p(early + late) - 3) ** 2
    return np.column_stack((np.ones(len(early)), -np.asarray(benefits), harm))


def profile(world: str, fold: int, seed: str, candidate: str, mu: float) -> dict[str, Any]:
    fit = read(EXTENSION / "cells" / f"outer{fold}" / seed / world / candidate / "fit.json")
    matches = [item for item in fit["profiles"] if item["mu"] == mu]
    if len(matches) != 1:
        raise ValueError("The prescribed fixed-strength profile is absent or ambiguous")
    return matches[0]


def main() -> None:
    registration, inspector = adopt()
    identity = {
        "registration_sha256": digest(OUTPUT / "registration.json"),
        "source_hashes": registration["source_hashes"],
        "prerequisite_hashes": registration["prerequisite_hashes"],
    }
    complete = OUTPUT / "complete.json"
    if complete.exists():
        receipt = read(complete)
        if receipt["identity"] != identity or any(digest(OUTPUT / p) != sha for p, sha in receipt["files"].items()):
            raise ValueError("Completed diagnostic identity or outputs changed")
        return
    audit = inspector.b.Audit()
    data, clean, matched, noises = inspector.inputs(audit)
    if audit.failures:
        raise ValueError("Frozen covariate/noise inspection failed")
    cache, tables, contexts = cached_benefits(), [], []
    early, late, alpha = data["arrays"]["phase0_epochs"], data["arrays"]["phase1_epochs"], data["alpha"]
    for fold, split in enumerate(data["splits"]):
        train, held, tied = (split[name] for name in ("train", "test", "tied_test"))
        total = early[held] + late[held]
        q0, q1 = (
            np.vstack((early[held], alpha * total, early[tied])),
            np.vstack((late[held], (1 - alpha) * total, late[tied])),
        )
        n = len(held)
        ids = data["rows"].iloc[np.r_[held, held, tied, held]].row_id.to_numpy()
        roles = (
            ["asymmetric"] * n + ["matched_tied"] * n + ["actual_tied"] * len(tied) + ["gain_tied_minus_asymmetric"] * n
        )
        for world, candidate, clock, mu in CONTRACT["worlds"]:
            x, q = design(early[train], late[train], clock, mu, alpha, cache), design(q0, q1, clock, mu, alpha, cache)
            base = profile(world, fold, "clean", candidate, mu)
            truth = np.r_[clean[world][held], matched[world][held], clean[world][tied]]
            baseline = np.asarray(base["query_predictions"])
            for seed in CONTRACT["seeds"]:
                saved = profile(world, fold, str(seed), candidate, mu)
                coefficients = np.asarray(saved["head"]["coefficients"])
                predicted = np.asarray(saved["query_predictions"])
                replay = FLOOR + np.exp(saved["head"]["intercept"] + q @ np.r_[0.0, coefficients])
                if np.max(abs(replay - predicted)) > 1e-12:
                    raise ValueError("Saved fixed-strength query reconstruction disagrees")
                actual = predicted - baseline
                absolute = predicted - truth
                actual = np.r_[actual, actual[n : 2 * n] - actual[:n]]
                absolute = np.r_[absolute, absolute[n : 2 * n] - absolute[:n]]
                for face in CONTRACT["faces"]:
                    active = np.ones(5, bool) if face == "unrestricted_tangent" else np.r_[True, coefficients > 0]
                    values, details = noise_response(
                        x, q, clean[world][train] - FLOOR, truth - FLOOR, noises[str(seed)][train], active, n, mu == 0
                    )
                    frame = pd.DataFrame(values)
                    frame["world"], frame["fold"], frame["seed"], frame["face"] = world, fold, seed, face
                    frame["role"], frame["row_id"] = roles, ids
                    frame["actual_noisy_minus_clean_bpb"], frame["actual_absolute_error_bpb"] = actual, absolute
                    frame["actual_noise_about_projected_face_bpb"] = (
                        absolute - values["face_clean_approximation_bias_bpb"]
                    )
                    comparable = values["log_identified"]
                    exact_face_discrepancy = float(
                        np.max(abs(values["exact_face_error_bpb"][comparable] - absolute[comparable]), initial=0)
                    )
                    details["exact_face_vs_saved_maximum_error_bpb"] = exact_face_discrepancy
                    if face == "saved_noisy_active_face" and exact_face_discrepancy > 1e-8:
                        raise ValueError("Fixed saved active-face decomposition does not reconstruct its NNLS head")
                    tables.append(frame)
                    contexts.append(
                        {
                            "world": world,
                            "fold": fold,
                            "seed": seed,
                            "face": face,
                            "mu": mu,
                            "clean_head": base["head"],
                            "noisy_head": saved["head"],
                            "active_boundary_present": bool(np.any(coefficients == 0)),
                            **details,
                        }
                    )
    frame = pd.concat(tables, ignore_index=True)
    if len(contexts) != 2 * CONTRACT["expected_contexts"] or len(frame) != CONTRACT["expected_query_and_gain_rows"]:
        raise ValueError("Fixed diagnostic row scope changed")
    moments = (
        frame.groupby(["world", "fold", "face", "role", "row_id"], sort=True)
        .agg(
            seeds=("seed", "count"),
            actual_mean=("actual_noisy_minus_clean_bpb", "mean"),
            actual_sd=("actual_noisy_minus_clean_bpb", "std"),
            predicted_bias=("expected_total_error_bpb", "mean"),
            actual_noise_about_face_mean=("actual_noise_about_projected_face_bpb", "mean"),
            predicted_noise_about_face_bias=("expected_second_order_bpb_bias", "mean"),
            predicted_log_sd=("log_estimator_sd", "mean"),
            fisher_sd=("fisher_sd", "mean"),
        )
        .reset_index()
    )
    cosine_rows = []
    for clock in CONTRACT["m_cosine"]["clocks"]:
        with np.load(GEOMETRY / clock / "tangents.npz", allow_pickle=False) as saved:
            amplitudes, derivative = saved["amplitudes"], saved["derivatives_0.0005"]
        for fold, split in enumerate(data["splits"]):
            train, labels = split["train"], split["inner"]
            for subset, indices in [("outer_train", train)] + [
                (f"inner_train_{k}", train[labels != k]) for k in range(3)
            ]:
                known = np.column_stack((amplitudes[indices], derivative[indices, :3]))
                for nuisance, matrix in [
                    ("known_floor", known),
                    ("plus_floor", np.column_stack((known, np.ones(len(indices))))),
                ]:
                    cosine_rows.append(
                        {
                            "clock": clock,
                            "fold": fold,
                            "subset": subset,
                            "nuisance": nuisance,
                            "training_indices": indices.tolist(),
                            **residual_cosine(
                                matrix,
                                clean["heterogeneous_null"][indices] - clean["native_null"][indices],
                                derivative[indices, 3],
                            ),
                        }
                    )
    if len(cosine_rows) != CONTRACT["m_cosine"]["rows"]:
        raise ValueError("Misspecification projection scope changed")
    frame.to_csv(OUTPUT / "query_noise_response.csv", index=False)
    moments.to_csv(OUTPUT / "three_seed_moments.csv", index=False)
    (OUTPUT / "contexts.json").write_text(json.dumps(contexts, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (OUTPUT / "misspecification_cosines.json").write_text(
        json.dumps(cosine_rows, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    summary = {
        "completed": True,
        "scientific_qualification": False,
        "learned_floor_domain_ready": False,
        "new_fits": 0,
        "contexts": 45,
        "conditional_faces": 90,
        "rows": len(frame),
        "misspecification_rows": 80,
        "interpretation": (
            "Local, conditional synthetic diagnostics. Fisher values are unconstrained information geometry, "
            "not NNLS uncertainty guarantees. Active faces are retrospective; three seeds do not calibrate "
            "tail probabilities. A small training Jensen term may be amplified at queries. "
            "No gate, mechanism identification or policy-selection claim."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    names = [
        "query_noise_response.csv",
        "three_seed_moments.csv",
        "contexts.json",
        "misspecification_cosines.json",
        "summary.json",
    ]
    complete.write_text(
        json.dumps({"identity": identity, "files": {p: digest(OUTPUT / p) for p in names}}, indent=2, sort_keys=True)
        + "\n"
    )


if __name__ == "__main__":
    main()
