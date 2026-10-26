# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Outcome-free local response-Jacobian audit of the signed phase designs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import audit_nextgen_two_phase_20260921 as geometry
import nextgen_two_phase_round3_20260921 as candidate
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
RELATIVE_CUTOFFS = (1e-6, 1e-10)


def spectrum(matrix: np.ndarray, scale: np.ndarray | None = None) -> dict[str, Any]:
    norms = np.linalg.norm(matrix, axis=0)
    normalizers = norms if scale is None else scale
    active = normalizers > np.finfo(float).eps
    scaled = matrix[:, active] / normalizers[active]
    singular = np.linalg.svd(scaled, compute_uv=False)
    peak = float(singular[0]) if len(singular) else 0.0
    rank = int(np.sum(singular > peak * 1e-6)) if peak else 0
    return {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "active_columns": int(active.sum()),
        "column_norms": norms.tolist(),
        "singular_values": singular.tolist(),
        "rank_relative_1e6": rank,
        "parameter_nullity_relative_1e6": int(matrix.shape[1] - rank),
        "full_column_rank_relative_1e6": rank == matrix.shape[1],
        "full_column_condition_relative_1e6": float(peak / singular[-1]) if rank == matrix.shape[1] and peak else None,
        "rank_relative_1e10": int(np.sum(singular > peak * 1e-10)) if peak else 0,
        "smallest_over_largest": float(singular[-1] / peak) if peak else 0.0,
        "normalization": "own column norms" if scale is None else "original phase column norms",
    }


def partitions(panel: candidate.benchmark.data.BenchPanel) -> list[tuple[str, np.ndarray]]:
    folds = pd.read_csv(geometry.PREPARED / panel.name / "folds.csv", dtype={"row_id": str})
    result = [("all_asymmetric", np.flatnonzero(~panel.tied))]
    for outer in sorted(folds.outer_fold.unique()):
        selected = folds[(folds["repeat"] == 0) & (folds.outer_fold == outer)]
        aligned = selected.set_index("row_id").reindex(panel.rows.row_id.astype(str))
        train = np.flatnonzero(aligned.role.eq("train"))
        assert not panel.tied[train].any()
        inner = aligned.iloc[train].inner_fold.to_numpy(int)
        result.append((f"repeat0_outer{outer}_train", train))
        for fold in sorted(set(inner)):
            result.append((f"repeat0_outer{outer}_inner{fold}_train", train[inner != fold]))
    return result


def response_jacobian(
    panel: candidate.benchmark.data.BenchPanel, model: str, nonzero_phase: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = np.arange(len(panel.rows))
    aggregate, phase, _, _ = candidate.matrices(panel, rows, SHAPE, model, 0.0)
    beta = geometry.generator_coefficients(panel)
    theta = (0.2 / len(panel.buckets)) * np.linspace(1.0, 2.0, phase.shape[1])
    if not nonzero_phase:
        theta = np.zeros_like(theta)
    eta = -0.2 + aggregate @ beta + phase @ theta
    deficit = np.exp(eta)
    derivative_columns, checks = [], {}
    for name in SHAPE:
        approximations = []
        for step in (1e-4, 5e-5):
            upper, lower = dict(SHAPE), dict(SHAPE)
            upper[name] *= np.exp(step)
            lower[name] *= np.exp(-step)
            ah, zh, _, _ = candidate.matrices(panel, rows, upper, model, 0.0)
            al, zl, _, _ = candidate.matrices(panel, rows, lower, model, 0.0)
            approximations.append(((ah - al) @ beta + (zh - zl) @ theta) / (2 * step))
        coarse, fine = approximations
        checks[name] = float(np.linalg.norm(fine - coarse) / max(np.linalg.norm(fine), 1e-30))
        derivative_columns.append(deficit * fine)
    nuisance = np.column_stack([np.ones(len(rows)), deficit, aggregate * deficit[:, None], *derivative_columns])
    return (
        nuisance,
        phase * deficit[:, None],
        {
            "shape": SHAPE,
            "floor": 0.5,
            "intercept": -0.2,
            "aggregate_coefficients": beta.tolist(),
            "phase_coefficients": theta.tolist(),
            "deficit_range": [float(min(deficit)), float(max(deficit))],
            "nuisance_order": [
                "floor",
                "intercept",
                "2M aggregate coefficients",
                "log rate",
                "log power",
                "log threshold",
            ],
            "central_difference_log_steps": [1e-4, 5e-5],
            "derivative_relative_change": checks,
        },
    )


def analyze(nuisance: np.ndarray, phase: np.ndarray) -> dict[str, Any]:
    original_norms = np.linalg.norm(phase, axis=0)
    original_singular = np.linalg.svd(phase / original_norms, compute_uv=False)
    norms = np.linalg.norm(nuisance, axis=0)
    scaled = nuisance[:, norms > np.finfo(float).eps] / norms[norms > np.finfo(float).eps]
    u, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    projections = {}
    for cutoff in RELATIVE_CUTOFFS:
        kept = singular > singular[0] * cutoff
        residual = phase - u[:, kept] @ (u[:, kept].T @ phase)
        residual_singular = np.linalg.svd(residual / original_norms, compute_uv=False)
        fractions = np.divide(
            np.linalg.norm(residual, axis=0), original_norms, out=np.zeros_like(original_norms), where=original_norms > 0
        )
        projections[str(cutoff)] = {
            "nuisance_rank": int(kept.sum()),
            "available_row_directions": int(len(phase) - kept.sum()),
            "phase_rank_relative_to_original_at_1e6": int(np.sum(residual_singular > original_singular[0] * 1e-6)),
            "phase_rank_relative_to_original_at_1e10": int(np.sum(residual_singular > original_singular[0] * 1e-10)),
            "retained_phase_norm_fraction": fractions.tolist(),
            "original_scaled_spectrum": spectrum(residual, original_norms),
            "residual_scaled_spectrum": spectrum(residual),
        }
    return {
        "rows": len(phase),
        "nuisance_parameters": nuisance.shape[1],
        "phase_parameters": phase.shape[1],
        "total_parameters": nuisance.shape[1] + phase.shape[1],
        "nuisance": spectrum(nuisance),
        "raw_phase": spectrum(phase),
        "joint_jacobian": spectrum(np.column_stack([nuisance, phase])),
        "projected_phase": projections,
    }


def main() -> None:
    sources = candidate.source_hashes()
    sources[str(Path(__file__).resolve())] = candidate.benchmark.sha256(Path(__file__))
    records = []
    for name in ("starcoder_wsd80_dense", "tpp40"):
        panel = geometry.geometry_panel(geometry.PREPARED, name)
        subset_list = partitions(panel)
        for model in ("NG2-009", "NG2-010"):
            for phase_state in (False, True):
                nuisance, phase, params = response_jacobian(panel, model, phase_state)
                assert max(params["derivative_relative_change"].values()) < 1e-6
                for subset, indices in subset_list:
                    record: dict[str, Any] = {
                        "panel": name,
                        "model": model,
                        "phase_state": "positive_generic" if phase_state else "zero_phase",
                        "subset": subset,
                        "parameters": params,
                        **analyze(nuisance[indices], phase[indices]),
                    }
                    records.append(record)
                    if subset == "all_asymmetric" and phase_state:
                        projected = record["projected_phase"]["1e-06"]
                        fractions = projected["retained_phase_norm_fraction"]
                        print(
                            json.dumps(
                                {
                                    "panel": name,
                                    "model": model,
                                    "rows": len(indices),
                                    "parameters": record["total_parameters"],
                                    "nuisance_rank": projected["nuisance_rank"],
                                    "joint_rank": record["joint_jacobian"]["rank_relative_1e6"],
                                    "phase_rank": projected["phase_rank_relative_to_original_at_1e6"],
                                    "retained_min_median_max": [
                                        min(fractions),
                                        float(np.median(fractions)),
                                        max(fractions),
                                    ],
                                }
                            ),
                            flush=True,
                        )
    assert candidate.source_hashes() == {
        key: val for key, val in sources.items() if key != str(Path(__file__).resolve())
    }
    identity = {}
    for name in ("starcoder_wsd80_dense", "tpp40"):
        identity[name] = {
            filename: candidate.benchmark.sha256(geometry.PREPARED / name / filename)
            for filename in ("metadata.json", "rows.csv", "policies.npz", "folds.csv")
        }
    output = {
        "sources": sources,
        "geometry_files": identity,
        "status": "diagnostic_only",
        "outcome_files_read": [],
        "scope": (
            "Repeat 0, all five outer training sets and every actual inner training subset, plus all asymmetric rows"
        ),
        "checks": {
            "source_stability": True,
            "no_tied_training_rows": True,
            "finite_difference_convergence_relative_limit": 1e-6,
        },
        "interpretation": [
            "Column normalization removes parameter units; original-scaled residual spectra retain effect size.",
            "Use phase_rank_relative_to_original: normalizing a vanishing residual by its own peak counts roundoff.",
            "Projection cutoff sensitivity is explicit at relative singular values 1e-6 and 1e-10.",
            "The floor column describes a train-data-fixed floor; positivity is inactive at positive coefficients.",
            "Continuous shape derivatives enlarge the nuisance span beyond the discrete frozen grid; "
            "this diagnoses nearby-shape aliasing rather than formal fixed-grid nonidentifiability.",
            "Rank and conditioning are local at two fixed generic heads, not empirical noise-weighted recovery "
            "or evidence of positive phase gain.",
            "Full column rank cannot establish off-support optimum reliability. One contrast per aggregate "
            "does not isolate phase effects nonparametrically.",
        ],
        "records": records,
    }
    candidate.benchmark.write_json(candidate.ROOT / "geometry_rank.json", output)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
