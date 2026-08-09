# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy==1.16.3"]
# ///
"""Test rank-one phase-field recovery on the actual 300M policy design."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "rank_one_phase_field_recovery_20260804"
CANDIDATE_ID = "WSD80-SUR-081"
RANDOM_SEED = 20260804
RANDOM_FACTOR_CASES = 8
STRESS_FACTOR_CASES = 4
STRESS_POOL_SIZE = 256
FOLDS = 5
NOISE_REPLICATES = 3
NOISE_SD = {
    "uncheatable": 0.0011270969148995812,
    "table9": 0.003330034591,
}
SIGNAL_RMS = (0.0028, 0.0039, 0.01)
PRIMARY_SIGNAL_RMS = 0.0039
ALS_STARTS_NOISELESS = 32
ALS_STARTS_NOISY = 8
ALS_MAX_ITERATIONS = 100
ALS_RIDGE = 1e-10
PROTOCOL = {
    "candidate_id": CANDIDATE_ID,
    "date": "2026-08-04",
    "purpose": "outcome-free rank-one recovery audit on the actual 300M asymmetric design",
    "data_use": {
        "policy_design": "238 asymmetric rows from the 520-row 300M development panel",
        "endpoint_targets_read": False,
        "noise_sd_source": "11-run tied proportional total run-level endpoint variation",
        "sealed_outcomes_used": False,
    },
    "operator": "r_i=(d_i^T u)(h_i^T v)",
    "contrast": "w1-w0 in a deterministic 38-dimensional simplex-tangent basis",
    "aggregate_bases": {
        "full_linear_tangent": "intercept plus centered 38-dimensional aggregate tangent",
        "declared_family_masses": "intercept plus two centered independent predeclared family masses",
    },
    "factor_cases": {
        "random": RANDOM_FACTOR_CASES,
        "geometry_stress": STRESS_FACTOR_CASES,
        "stress_pool_size": STRESS_POOL_SIZE,
        "selection": "largest local-Jacobian condition number from an outcome-free isotropic pool",
    },
    "simulation": {
        "random_seed": RANDOM_SEED,
        "signal_rms_bpb": SIGNAL_RMS,
        "primary_signal_rms_bpb": PRIMARY_SIGNAL_RMS,
        "noise_sd_bpb": NOISE_SD,
        "folds": FOLDS,
        "noise_replicates": NOISE_REPLICATES,
        "als_starts_noiseless": ALS_STARTS_NOISELESS,
        "als_starts_noisy": ALS_STARTS_NOISY,
        "als_max_iterations": ALS_MAX_ITERATIONS,
        "als_ridge": ALS_RIDGE,
    },
    "gates": {
        "local_jacobian_full_rank_fraction": 1.0,
        "noiseless_matrix_cosine_q10_min": 0.95,
        "noiseless_signal_rmse_ratio_q90_max": 0.05,
        "primary_noisy_matrix_cosine_median_min": 0.8,
        "primary_noisy_signal_rmse_ratio_median_max": 0.5,
        "primary_noisy_signal_rmse_ratio_q90_max": 1.0,
        "primary_noisy_sign_accuracy_median_min": 0.8,
        "must_pass_both_noise_levels": True,
    },
    "decision_rule": (
        "A basis is mathematically recoverable only if it passes the local-rank, noiseless, and primary noisy "
        "gates. A pass does not license endpoint fitting: factors still require a separately preregistered physical "
        "meaning and may not be selected from endpoint outcomes."
    ),
}


@dataclass(frozen=True)
class Design:
    name: str
    left: np.ndarray
    right: np.ndarray
    expected_rank_one_dof: int


@dataclass(frozen=True)
class FactorCase:
    basis: str
    name: str
    kind: str
    u: np.ndarray
    v: np.ndarray
    jacobian_rank: int
    jacobian_condition: float


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simplex_tangent_basis(dimension: int) -> np.ndarray:
    """Return a deterministic orthonormal Helmert basis for sum-zero vectors."""
    basis = np.zeros((dimension, dimension - 1), dtype=float)
    for column in range(dimension - 1):
        count = column + 1
        scale = np.sqrt(count * (count + 1))
        basis[:count, column] = 1.0 / scale
        basis[count, column] = -count / scale
    if not np.allclose(basis.T @ basis, np.eye(dimension - 1), atol=1e-12):
        raise AssertionError("invalid tangent basis")
    if not np.allclose(basis.sum(axis=0), 0.0, atol=1e-12):
        raise AssertionError("tangent basis columns must sum to zero")
    return basis


def rms_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(matrix**2, axis=0))
    if np.any(scale <= 1e-12):
        raise ValueError("design contains a constant-zero coordinate")
    return matrix / scale[None, :], scale


def policy_designs() -> tuple[dict[str, Design], dict[str, Any]]:
    dataset = benchmark.load_300m("uncheatable")
    beta0 = benchmark.geometry_300m(dataset).phase_0_fraction
    asymmetric = np.max(np.abs(dataset.weights[:, 0] - dataset.weights[:, 1]), axis=1) > 1e-10
    weights = dataset.weights[asymmetric]
    aggregate = beta0 * weights[:, 0] + (1.0 - beta0) * weights[:, 1]
    contrast = weights[:, 1] - weights[:, 0]
    tangent = simplex_tangent_basis(aggregate.shape[1])

    left, left_scale = rms_scale(contrast @ tangent)
    aggregate_tangent, aggregate_scale = rms_scale((aggregate - aggregate.mean(axis=0)) @ tangent)
    full_right = np.column_stack([np.ones(len(aggregate)), aggregate_tangent])

    family_count = int(dataset.family_index.max()) + 1
    family_mass = np.column_stack(
        [aggregate[:, dataset.family_index == family].sum(axis=1) for family in range(family_count)]
    )
    family_independent, family_scale = rms_scale(family_mass[:, :-1] - family_mass[:, :-1].mean(axis=0))
    family_right = np.column_stack([np.ones(len(aggregate)), family_independent])

    designs = {
        "full_linear_tangent": Design(
            name="full_linear_tangent",
            left=left,
            right=full_right,
            expected_rank_one_dof=left.shape[1] + full_right.shape[1] - 1,
        ),
        "declared_family_masses": Design(
            name="declared_family_masses",
            left=left,
            right=family_right,
            expected_rank_one_dof=left.shape[1] + family_right.shape[1] - 1,
        ),
    }
    metadata = {
        "rows": len(aggregate),
        "buckets": aggregate.shape[1],
        "phase_0_fraction": beta0,
        "left_dimension": left.shape[1],
        "full_right_dimension": full_right.shape[1],
        "family_right_dimension": family_right.shape[1],
        "family_count": family_count,
        "left_scale_min": float(left_scale.min()),
        "left_scale_max": float(left_scale.max()),
        "aggregate_scale_min": float(aggregate_scale.min()),
        "aggregate_scale_max": float(aggregate_scale.max()),
        "family_scale_min": float(family_scale.min()),
        "family_scale_max": float(family_scale.max()),
    }
    return designs, metadata


def rank_one_jacobian(design: Design, u: np.ndarray, v: np.ndarray) -> tuple[int, float]:
    left_block = design.left * (design.right @ v)[:, None]
    right_block = design.right * (design.left @ u)[:, None]
    jacobian = np.column_stack([left_block, right_block])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = singular[0] * max(jacobian.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > threshold))
    expected = design.expected_rank_one_dof
    condition = np.inf if rank < expected else float(singular[0] / singular[expected - 1])
    return rank, condition


def normalized_factor(rng: np.random.Generator, dimension: int) -> np.ndarray:
    vector = rng.normal(size=dimension)
    return vector / np.linalg.norm(vector)


def factor_cases(design: Design, basis_seed: int) -> tuple[FactorCase, ...]:
    rng = np.random.default_rng(basis_seed)
    pool: list[FactorCase] = []
    for index in range(STRESS_POOL_SIZE):
        u = normalized_factor(rng, design.left.shape[1])
        v = normalized_factor(rng, design.right.shape[1])
        rank, condition = rank_one_jacobian(design, u, v)
        pool.append(
            FactorCase(
                basis=design.name,
                name=f"pool_{index:03d}",
                kind="pool",
                u=u,
                v=v,
                jacobian_rank=rank,
                jacobian_condition=condition,
            )
        )
    selected: list[FactorCase] = []
    for index, case in enumerate(pool[:RANDOM_FACTOR_CASES]):
        selected.append(
            FactorCase(
                basis=case.basis,
                name=f"random_{index:02d}",
                kind="random",
                u=case.u,
                v=case.v,
                jacobian_rank=case.jacobian_rank,
                jacobian_condition=case.jacobian_condition,
            )
        )
    stress_pool = sorted(pool[RANDOM_FACTOR_CASES:], key=lambda case: case.jacobian_condition, reverse=True)
    for index, case in enumerate(stress_pool[:STRESS_FACTOR_CASES]):
        selected.append(
            FactorCase(
                basis=case.basis,
                name=f"geometry_stress_{index:02d}",
                kind="geometry_stress",
                u=case.u,
                v=case.v,
                jacobian_rank=case.jacobian_rank,
                jacobian_condition=case.jacobian_condition,
            )
        )
    return tuple(selected)


def ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    augmented_design = np.vstack([design, np.sqrt(ridge) * np.eye(design.shape[1])])
    augmented_target = np.concatenate([target, np.zeros(design.shape[1])])
    coefficient, *_ = np.linalg.lstsq(augmented_design, augmented_target, rcond=1e-12)
    return coefficient


def balance_factors(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_norm = max(float(np.linalg.norm(u)), 1e-15)
    v_norm = max(float(np.linalg.norm(v)), 1e-15)
    scale = np.sqrt(v_norm / u_norm)
    return u * scale, v / scale


def initial_factors(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    starts: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    adjoint = left.T @ (target[:, None] * right)
    left_singular, singular, right_singular = np.linalg.svd(adjoint, full_matrices=False)
    scale = np.sqrt(max(float(singular[0]), 1e-15) / max(len(target), 1))
    initial = [(left_singular[:, 0] * scale, right_singular[0] * scale)]
    for _ in range(starts - 1):
        initial.append(
            (
                normalized_factor(rng, left.shape[1]) * np.sqrt(np.std(target)),
                normalized_factor(rng, right.shape[1]) * np.sqrt(np.std(target)),
            )
        )
    return initial


def fit_rank_one(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    starts: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for initial_u, initial_v in initial_factors(left, right, target, rng, starts):
        u = initial_u
        v = initial_v
        previous: float | None = None
        for _ in range(ALS_MAX_ITERATIONS):
            u = ridge_solve(left * (right @ v)[:, None], target, ALS_RIDGE)
            v = ridge_solve(right * (left @ u)[:, None], target, ALS_RIDGE)
            u, v = balance_factors(u, v)
            prediction = (left @ u) * (right @ v)
            mse = float(np.mean((prediction - target) ** 2))
            if previous is not None and previous - mse <= 1e-12 * max(previous, 1.0):
                break
            previous = mse
        candidates.append((mse, u.copy(), v.copy()))
    mse, u, v = min(candidates, key=lambda candidate: candidate[0])
    return u, v, mse


def matrix_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.sum(left * right) / denominator) if denominator > 0 else 0.0


def scaled_truth(design: Design, case: FactorCase, signal_rms: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.outer(case.u, case.v)
    signal = np.einsum("ni,ij,nj->n", design.left, matrix, design.right)
    scale = signal_rms / float(np.sqrt(np.mean(signal**2)))
    return matrix * scale, signal * scale


def noiseless_recovery(designs: dict[str, Design], cases: tuple[FactorCase, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        design = designs[case.basis]
        truth, signal = scaled_truth(design, case, PRIMARY_SIGNAL_RMS)
        fitted_u, fitted_v, train_mse = fit_rank_one(
            design.left,
            design.right,
            signal,
            seed=RANDOM_SEED + case_index,
            starts=ALS_STARTS_NOISELESS,
        )
        fitted = np.outer(fitted_u, fitted_v)
        prediction = np.einsum("ni,ij,nj->n", design.left, fitted, design.right)
        rows.append(
            {
                "basis": case.basis,
                "factor_case": case.name,
                "factor_kind": case.kind,
                "jacobian_rank": case.jacobian_rank,
                "expected_jacobian_rank": design.expected_rank_one_dof,
                "jacobian_condition": case.jacobian_condition,
                "train_mse": train_mse,
                "matrix_cosine": matrix_cosine(truth, fitted),
                "signal_rmse_ratio": float(np.sqrt(np.mean((prediction - signal) ** 2)) / PRIMARY_SIGNAL_RMS),
            }
        )
    return pd.DataFrame(rows)


def fold_assignments(rows: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(RANDOM_SEED)
    return tuple(np.asarray(part, dtype=int) for part in np.array_split(rng.permutation(rows), FOLDS))


def noisy_recovery(designs: dict[str, Design], cases: tuple[FactorCase, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        design = designs[case.basis]
        folds = fold_assignments(len(design.left))
        for signal_rms in SIGNAL_RMS:
            truth, signal = scaled_truth(design, case, signal_rms)
            for target_index, (target, noise_sd) in enumerate(NOISE_SD.items()):
                for replicate in range(NOISE_REPLICATES):
                    noise_seed = RANDOM_SEED + 100_000 * case_index + 10_000 * target_index + 100 * replicate
                    noise = np.random.default_rng(noise_seed).normal(scale=noise_sd, size=len(signal))
                    observed = signal + noise
                    for fold_index, test_indices in enumerate(folds):
                        train_mask = np.ones(len(signal), dtype=bool)
                        train_mask[test_indices] = False
                        train_indices = np.flatnonzero(train_mask)
                        fitted_u, fitted_v, train_mse = fit_rank_one(
                            design.left[train_indices],
                            design.right[train_indices],
                            observed[train_indices],
                            seed=noise_seed + fold_index,
                            starts=ALS_STARTS_NOISY,
                        )
                        fitted = np.outer(fitted_u, fitted_v)
                        prediction = np.einsum(
                            "ni,ij,nj->n",
                            design.left[test_indices],
                            fitted,
                            design.right[test_indices],
                        )
                        true_test = signal[test_indices]
                        rows.append(
                            {
                                "basis": case.basis,
                                "factor_case": case.name,
                                "factor_kind": case.kind,
                                "signal_rms": signal_rms,
                                "target_noise": target,
                                "noise_sd": noise_sd,
                                "replicate": replicate,
                                "fold": fold_index,
                                "train_mse": train_mse,
                                "matrix_cosine": matrix_cosine(truth, fitted),
                                "test_signal_rmse_ratio": float(
                                    np.sqrt(np.mean((prediction - true_test) ** 2)) / signal_rms
                                ),
                                "test_sign_accuracy": float(np.mean(np.sign(prediction) == np.sign(true_test))),
                            }
                        )
    return pd.DataFrame(rows)


def quantile(series: pd.Series, value: float) -> float:
    return float(series.quantile(value))


def summarize_recovery(noiseless: pd.DataFrame, noisy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for basis, group in noiseless.groupby("basis", sort=True):
        rows.append(
            {
                "basis": basis,
                "target_noise": "noiseless",
                "signal_rms": PRIMARY_SIGNAL_RMS,
                "n": len(group),
                "jacobian_full_rank_fraction": float(np.mean(group["jacobian_rank"] == group["expected_jacobian_rank"])),
                "matrix_cosine_median": float(group["matrix_cosine"].median()),
                "matrix_cosine_q10": quantile(group["matrix_cosine"], 0.1),
                "signal_rmse_ratio_median": float(group["signal_rmse_ratio"].median()),
                "signal_rmse_ratio_q90": quantile(group["signal_rmse_ratio"], 0.9),
                "sign_accuracy_median": 1.0,
            }
        )
    for (basis, target, signal_rms), group in noisy.groupby(["basis", "target_noise", "signal_rms"], sort=True):
        rows.append(
            {
                "basis": basis,
                "target_noise": target,
                "signal_rms": signal_rms,
                "n": len(group),
                "jacobian_full_rank_fraction": np.nan,
                "matrix_cosine_median": float(group["matrix_cosine"].median()),
                "matrix_cosine_q10": quantile(group["matrix_cosine"], 0.1),
                "signal_rmse_ratio_median": float(group["test_signal_rmse_ratio"].median()),
                "signal_rmse_ratio_q90": quantile(group["test_signal_rmse_ratio"], 0.9),
                "sign_accuracy_median": float(group["test_sign_accuracy"].median()),
            }
        )
    return pd.DataFrame(rows)


def gate_decisions(summary: pd.DataFrame) -> dict[str, Any]:
    gates = PROTOCOL["gates"]
    decisions: dict[str, Any] = {}
    for basis in sorted(summary["basis"].unique()):
        structural = summary.loc[(summary["basis"] == basis) & (summary["target_noise"] == "noiseless")].iloc[0]
        structural_pass = bool(
            structural["jacobian_full_rank_fraction"] >= gates["local_jacobian_full_rank_fraction"]
            and structural["matrix_cosine_q10"] >= gates["noiseless_matrix_cosine_q10_min"]
            and structural["signal_rmse_ratio_q90"] <= gates["noiseless_signal_rmse_ratio_q90_max"]
        )
        target_decisions: dict[str, bool] = {}
        for target in NOISE_SD:
            row = summary.loc[
                (summary["basis"] == basis)
                & (summary["target_noise"] == target)
                & np.isclose(summary["signal_rms"], PRIMARY_SIGNAL_RMS)
            ].iloc[0]
            target_decisions[target] = bool(
                row["matrix_cosine_median"] >= gates["primary_noisy_matrix_cosine_median_min"]
                and row["signal_rmse_ratio_median"] <= gates["primary_noisy_signal_rmse_ratio_median_max"]
                and row["signal_rmse_ratio_q90"] <= gates["primary_noisy_signal_rmse_ratio_q90_max"]
                and row["sign_accuracy_median"] >= gates["primary_noisy_sign_accuracy_median_min"]
            )
        decisions[basis] = {
            "structural_pass": structural_pass,
            "target_noise_pass": target_decisions,
            "practical_both_targets_pass": structural_pass and all(target_decisions.values()),
        }
    decisions["any_basis_practical_both_targets_pass"] = any(
        value["practical_both_targets_pass"] for value in decisions.values() if isinstance(value, dict)
    )
    return decisions


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in selected.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def write_report(protocol_hash: str, summary: pd.DataFrame, decisions: dict[str, Any]) -> None:
    primary = summary.loc[
        summary["target_noise"].isin(NOISE_SD) & np.isclose(summary["signal_rms"], PRIMARY_SIGNAL_RMS)
    ].copy()
    primary = primary.round(4)
    noiseless = summary.loc[summary["target_noise"].eq("noiseless")].copy().round(6)
    decision_text = (
        "At least one basis passes mathematical recovery, but no endpoint model is promoted because factor meaning "
        "is not independently identified."
        if decisions["any_basis_practical_both_targets_pass"]
        else (
            "No basis passes structural and practical recovery on both measured noise levels; "
            "no endpoint model is promoted."
        )
    )
    structural_columns = [
        "basis",
        "n",
        "jacobian_full_rank_fraction",
        "matrix_cosine_median",
        "matrix_cosine_q10",
        "signal_rmse_ratio_q90",
    ]
    primary_columns = [
        "basis",
        "target_noise",
        "n",
        "matrix_cosine_median",
        "signal_rmse_ratio_median",
        "signal_rmse_ratio_q90",
        "sign_accuracy_median",
    ]
    structural_table = markdown_table(noiseless, structural_columns)
    primary_table = markdown_table(primary, primary_columns)
    report = f"""# Rank-one phase-field synthetic recovery audit

Protocol: `{protocol_hash}`

Registry candidate: `{CANDIDATE_ID}`

## Decision

**{decision_text}**

This audit uses the actual 238-row 300M asymmetric policy design but no endpoint targets. It simulates
rank-one phase responses at fixed RMS amplitudes and adds Gaussian noise calibrated to the exposed 11-run
proportional total run-level SD. The full-linear basis has 76 rank-one degrees of freedom; the predeclared
family-mass basis has 40.

## Noiseless structural recovery

{structural_table}

## Primary 0.0039-BPB recovery

{primary_table}

The noisy errors are measured against the latent noise-free held-fold signal, not the noisy observation.
Geometry-stress cases were chosen before simulation as the four worst local-Jacobian condition numbers from
256 isotropic factor draws. Random cases are the first eight draws.

## Interpretation

Passing this audit would establish numerical recovery only under the rank-one assumption and these design/noise
conditions. It cannot establish that the true phase field is rank one, assign physical meaning to fitted factors,
or license outcome-selected aggregate features. A subsequent endpoint candidate would still require a frozen,
mechanistically defined basis and independent review before target outcomes are fit.
"""
    (OUTPUT_DIR / "report.md").write_text(report)


def prepare() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    implementation_hash = file_hash(Path(__file__))
    protocol_hash = payload_hash(PROTOCOL)
    protocol = {
        **PROTOCOL,
        "implementation_sha256": implementation_hash,
        "protocol_hash": protocol_hash,
    }
    designs, metadata = policy_designs()
    all_cases = tuple(
        case
        for basis_index, design in enumerate(designs.values())
        for case in factor_cases(design, RANDOM_SEED + 10_000 * basis_index)
    )
    factor_rows = [
        {
            "basis": case.basis,
            "factor_case": case.name,
            "factor_kind": case.kind,
            "jacobian_rank": case.jacobian_rank,
            "jacobian_condition": case.jacobian_condition,
        }
        for case in all_cases
    ]
    arrays: dict[str, np.ndarray] = {}
    for name, design in designs.items():
        arrays[f"{name}__left"] = design.left
        arrays[f"{name}__right"] = design.right
    for index, case in enumerate(all_cases):
        arrays[f"factor_{index:03d}__u"] = case.u
        arrays[f"factor_{index:03d}__v"] = case.v
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "design_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(factor_rows).to_csv(OUTPUT_DIR / "factor_cases.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "frozen_design_and_factors.npz", **arrays)
    print(json.dumps({"protocol_hash": protocol_hash, "implementation_sha256": implementation_hash}, indent=2))


def load_frozen() -> tuple[dict[str, Design], tuple[FactorCase, ...], dict[str, Any]]:
    protocol = json.loads((OUTPUT_DIR / "protocol.json").read_text())
    if protocol["protocol_hash"] != payload_hash(PROTOCOL):
        raise RuntimeError("protocol changed after prepare")
    if protocol["implementation_sha256"] != file_hash(Path(__file__)):
        raise RuntimeError("implementation changed after prepare")
    factor_frame = pd.read_csv(OUTPUT_DIR / "factor_cases.csv")
    arrays = np.load(OUTPUT_DIR / "frozen_design_and_factors.npz")
    designs: dict[str, Design] = {}
    for name in PROTOCOL["aggregate_bases"]:
        left = arrays[f"{name}__left"]
        right = arrays[f"{name}__right"]
        designs[name] = Design(name, left, right, left.shape[1] + right.shape[1] - 1)
    cases: list[FactorCase] = []
    for index, row in factor_frame.iterrows():
        cases.append(
            FactorCase(
                basis=str(row["basis"]),
                name=str(row["factor_case"]),
                kind=str(row["factor_kind"]),
                u=arrays[f"factor_{index:03d}__u"],
                v=arrays[f"factor_{index:03d}__v"],
                jacobian_rank=int(row["jacobian_rank"]),
                jacobian_condition=float(row["jacobian_condition"]),
            )
        )
    return designs, tuple(cases), protocol


def run() -> None:
    designs, cases, protocol = load_frozen()
    noiseless = noiseless_recovery(designs, cases)
    noisy = noisy_recovery(designs, cases)
    summary = summarize_recovery(noiseless, noisy)
    decisions = gate_decisions(summary)
    decision = {
        "candidate_id": CANDIDATE_ID,
        "decision": "recovery_only_no_model_promoted",
        "basis_decisions": decisions,
        "endpoint_model_promoted": False,
        "protocol_hash": protocol["protocol_hash"],
    }
    noiseless.to_csv(OUTPUT_DIR / "noiseless_recovery.csv", index=False)
    noisy.to_csv(OUTPUT_DIR / "noisy_recovery.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "recovery_summary.csv", index=False)
    (OUTPUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    write_report(protocol["protocol_hash"], summary, decisions)
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
