# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Outcome-free geometry audit and final-head tied mapping for round four."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import audit_nextgen_two_phase_20260921 as geometry
import audit_two_phase_round3_geometry_20260921 as rank_audit
import nextgen_two_phase_round4_20260921_v2 as candidate
import numpy as np
from threadpoolctl import threadpool_limits

LIMIT = 1e-8
SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
OUTPUT = candidate.ROOT / "numerical_v2"


def audit_sources() -> dict[str, str]:
    paths = (Path(__file__), Path(geometry.__file__), Path(rank_audit.__file__))
    return {str(path.resolve()): candidate.benchmark.sha256(path) for path in paths}


def load_fit(record: dict[str, Any]) -> candidate.Fit:
    parameters = (
        record["intercept"],
        np.asarray(record["aggregate_coefficients"]),
        np.asarray(record["phase_coefficients"]),
        record["floor"],
    )
    if candidate.MODELS[record["model"]] == "mean":
        solver = record["solver"]
        head = candidate.mean_head.Head(
            *parameters, solver["objective"], solver["iterations"], solver["certified_kkt_residual"]
        )
    else:
        head = candidate.log_head.Head(*parameters)
    return candidate.Fit(
        record["model"],
        record["shape"],
        record["q"],
        record["kappa"],
        record["ridge"],
        record["gamma"],
        record["flat_profile"],
        record["inner_rmse"],
        head,
    )


def native_prediction(fit: candidate.Fit, panel: candidate.benchmark.data.BenchPanel, rows: np.ndarray) -> np.ndarray:
    head = candidate.mariner.Head(fit.head.intercept, fit.head.aggregate_coefficients, fit.head.floor)
    return head.predict(candidate.mariner.design_matrix(panel.epochs[rows], fit.shape))


def mapping_checks(fit: candidate.Fit, panel: candidate.benchmark.data.BenchPanel) -> dict[str, float]:
    rows = np.flatnonzero(panel.tied)
    all_rows = np.arange(len(panel.rows))
    actual = fit.predict(panel, rows)
    single = np.concatenate([fit.predict(panel, np.asarray([row])) for row in rows])
    foreign = dataclasses.replace(
        panel,
        phase0=panel.phase0[rows],
        phase1=panel.phase1[rows],
        rows=panel.rows.iloc[rows].reset_index(drop=True),
        targets={},
        metadata={
            **panel.metadata,
            "realized_phase0_epochs": panel.phase0_epochs[rows],
            "realized_phase1_epochs": panel.phase1_epochs[rows],
        },
    )
    tied = geometry.tied_queries(panel)
    return {
        "max_bpb_error": float(np.max(np.abs(actual - native_prediction(fit, panel, rows)))),
        "actual_tied_rowwise_vs_batch_max_bpb_error": float(np.max(np.abs(actual - single))),
        "actual_tied_foreign_rows_removed_max_bpb_error": float(
            np.max(np.abs(actual - fit.predict(foreign, np.arange(len(rows)))))
        ),
        "actual_tied_full_batch_max_bpb_error": float(np.max(np.abs(actual - fit.predict(panel, all_rows)[rows]))),
        "counterfactual_all_aggregates_max_bpb_error": float(
            np.max(np.abs(fit.predict(tied, all_rows) - native_prediction(fit, tied, all_rows)))
        ),
    }


def mapping_receipt() -> None:
    sources = candidate.source_hashes()
    panel = geometry.geometry_panel(geometry.PREPARED, "starcoder_wsd80_dense")
    assert int(panel.tied.sum()) == 20
    generic: list[dict[str, Any]] = []
    learned: list[dict[str, Any]] = []
    for model in candidate.MODELS:
        states = candidate.BOUNDARY_STATES[:1] if model == "NG2-011" else candidate.BOUNDARY_STATES
        for q, kappa in states:
            for shape in (
                {"rate": 0.05, "power": 0.3, "threshold": 1.0},
                SHAPE,
                {"rate": 4.0, "power": 1.0, "threshold": 6.0},
            ):
                for floor in (-3.0, 0.7):
                    parameters = (-0.2, np.array([0.8, 0.3, 0.1, 0.04]), np.empty(0), floor)
                    head = (
                        candidate.mean_head.Head(*parameters, 0.0, 0, 0.0)
                        if candidate.MODELS[model] == "mean"
                        else candidate.log_head.Head(*parameters)
                    )
                    fit = candidate.Fit(model, shape, q, kappa, 0.0, 1.5, True, 0.0, head)
                    generic.append(
                        {
                            "model": model,
                            "shape": shape,
                            "q": q,
                            "kappa": kappa,
                            "floor": floor,
                            **mapping_checks(fit, panel),
                        }
                    )
    for path in sorted((candidate.ROOT / "starcoder_screen_v2/shards").glob("*/fit.json")):
        complete = json.loads((path.parent / "complete.json").read_text())
        assert complete["identity"]["sources"] == sources
        assert complete["outputs"]["fit.json"] == candidate.benchmark.sha256(path)
        fit = load_fit(json.loads(path.read_text()))
        learned.append(
            {
                "model": fit.model,
                "target": complete["identity"]["target"],
                "outer_fold": complete["identity"]["outer_fold"],
                "fit": candidate.benchmark.receipt(path),
                **mapping_checks(fit, panel),
            }
        )
    assert len(learned) == 75, "Final structural mapping requires all75 learned heads"
    model_status = {}
    for model in candidate.MODELS:
        actual_heads = [row for row in learned if row["model"] == model]
        assert len(actual_heads) == 25
        checks = actual_heads + [row for row in generic if row["model"] == model]
        maximum = max(value for row in checks for key, value in row.items() if key.endswith("bpb_error"))
        model_status[model] = "pass" if maximum <= LIMIT else "fail"
    assert candidate.source_hashes() == sources
    candidate.benchmark.write_json(
        OUTPUT / "structural_mapping.json",
        {
            "sources": sources,
            "audit_sources": audit_sources(),
            "audit_amendment": {
                "reason": "Correct the audit input directory to the existing v2 screen output",
                "previous_driver": candidate.benchmark.receipt(
                    Path(__file__).with_name("audit_round4_mapping_geometry_20260921_v2.py")
                ),
                "old_input": str((OUTPUT / "starcoder_screen_v1").resolve()),
                "corrected_input": str((candidate.ROOT / "starcoder_screen_v2").resolve()),
                "model_sources_unchanged": True,
            },
            "status": "pass" if all(status == "pass" for status in model_status.values()) else "fail",
            "model_status": model_status,
            "numeric_limit": LIMIT,
            "query_row_ids": panel.rows.loc[panel.tied, "row_id"].astype(str).tolist(),
            "geometry": candidate.previous.geometry_identity(panel),
            "generic": generic,
            "learned_heads": learned,
            "outcomes_read": False,
            "new_fits": 0,
        },
    )
    print(json.dumps({"mapping_model_status": model_status, "generic_checks": len(generic)}), flush=True)


def jacobian(
    panel: candidate.benchmark.data.BenchPanel, q: float, kappa: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = np.arange(len(panel.rows))
    matrix = candidate.matrices(panel, rows, SHAPE, q, kappa)
    beta = geometry.generator_coefficients(panel)
    deficit = np.exp(-0.2 + matrix @ beta)
    derivatives, checks = [], {}
    for name in SHAPE:
        estimates = []
        for step in (1e-4, 5e-5):
            upper, lower = dict(SHAPE), dict(SHAPE)
            upper[name] *= np.exp(step)
            lower[name] *= np.exp(-step)
            estimates.append(
                (candidate.matrices(panel, rows, upper, q, kappa) - candidate.matrices(panel, rows, lower, q, kappa))
                @ beta
                / (2 * step)
            )
        checks[name] = float(np.linalg.norm(estimates[0] - estimates[1]) / max(np.linalg.norm(estimates[1]), 1e-30))
        derivatives.append(deficit * estimates[1])
    benefit = candidate.mariner.benefit
    total = benefit(panel.epochs, SHAPE["rate"], SHAPE["power"])
    prefix = benefit(panel.phase0_epochs, SHAPE["rate"], SHAPE["power"])
    tied = benefit(panel.alpha * panel.epochs, SHAPE["rate"], SHAPE["power"])
    delta = np.where(panel.phase0 == panel.phase1, 0.0, prefix - tied)
    dq = (-2 * kappa * total * delta) @ beta[: len(panel.buckets)]
    dk = (delta * (prefix + tied - 2 * q * total)) @ beta[: len(panel.buckets)]
    step = 1e-5
    finite_q = (
        (
            candidate.matrices(panel, rows, SHAPE, q + step, kappa)
            - candidate.matrices(panel, rows, SHAPE, q - step, kappa)
        )
        @ beta
        / (2 * step)
    )
    finite_k = (candidate.matrices(panel, rows, SHAPE, q, kappa + step) - matrix) @ beta / step
    for name, expected, measured in (("q", dq, finite_q), ("kappa", dk, finite_k)):
        checks[name] = float(np.linalg.norm(expected - measured) / max(np.linalg.norm(expected), 1e-30))
    assert max(checks.values()) < 1e-6
    nuisance = np.column_stack([np.ones(len(rows)), deficit, matrix * deficit[:, None], *derivatives])
    phase = deficit[:, None] * np.column_stack([dq, dk])
    if kappa == 0:
        assert np.array_equal(phase[:, 0], np.zeros(len(rows)))
    return (
        nuisance,
        phase,
        {
            "shape": SHAPE,
            "q": q,
            "kappa": kappa,
            "floor": 0.5,
            "intercept": -0.2,
            "aggregate_coefficients": beta.tolist(),
            "phase_order": ["q", "kappa"],
            "nuisance_order": [
                "floor",
                "intercept",
                "2M modified aggregate coefficients",
                "log rate",
                "log power",
                "log threshold",
            ],
            "derivative_checks_relative": checks,
        },
    )


def rank_record(nuisance: np.ndarray, phase: np.ndarray) -> dict[str, Any]:
    phase_norms = np.linalg.norm(phase, axis=0)
    scaled_phase = np.divide(phase, phase_norms, out=np.zeros_like(phase), where=phase_norms > 0)
    original_peak = np.linalg.svd(scaled_phase, compute_uv=False)[0]
    nuisance_norms = np.linalg.norm(nuisance, axis=0)
    selected = nuisance_norms > np.finfo(float).eps
    u, singular, _ = np.linalg.svd(nuisance[:, selected] / nuisance_norms[selected], full_matrices=False)
    projections = {}
    for cutoff in rank_audit.RELATIVE_CUTOFFS:
        keep = singular > singular[0] * cutoff
        residual = phase - u[:, keep] @ (u[:, keep].T @ phase)
        scaled_residual = np.divide(residual, phase_norms, out=np.zeros_like(residual), where=phase_norms > 0)
        residual_singular = np.linalg.svd(scaled_residual, compute_uv=False)
        projections[str(cutoff)] = {
            "nuisance_rank": int(keep.sum()),
            "available_row_directions": int(len(phase) - keep.sum()),
            "phase_rank_relative_to_original_at_1e6": int(np.sum(residual_singular > original_peak * 1e-6)),
            "phase_rank_relative_to_original_at_1e10": int(np.sum(residual_singular > original_peak * 1e-10)),
            "retained_phase_norm_fraction": (
                np.divide(
                    np.linalg.norm(residual, axis=0), phase_norms, out=np.zeros_like(phase_norms), where=phase_norms > 0
                ).tolist()
            ),
            "original_scaled_spectrum": rank_audit.spectrum(residual, phase_norms),
        }
    return {
        "rows": len(phase),
        "nuisance_parameters": nuisance.shape[1],
        "phase_parameters": 2,
        "total_parameters": nuisance.shape[1] + 2,
        "nuisance": rank_audit.spectrum(nuisance),
        "raw_phase": rank_audit.spectrum(phase),
        "joint_jacobian": rank_audit.spectrum(np.column_stack([nuisance, phase])),
        "projected_phase": projections,
    }


def geometry_receipt() -> None:
    sources = candidate.source_hashes()
    records: list[dict[str, Any]] = []
    identity = {}
    for name in ("starcoder_wsd80_dense", "tpp40"):
        panel = geometry.geometry_panel(geometry.PREPARED, name)
        identity[name] = {
            file: candidate.benchmark.sha256(geometry.PREPARED / name / file)
            for file in ("metadata.json", "rows.csv", "policies.npz", "folds.csv")
        }
        for kappa in (0.0, 0.5):
            nuisance, phase, parameters = jacobian(panel, 0.5, kappa)
            for subset, rows in rank_audit.partitions(panel):
                records.append(
                    {
                        "panel": name,
                        "models": ["NG2-012", "NG2-013"],
                        "phase_state": "null" if kappa == 0 else "positive_generic",
                        "subset": subset,
                        "parameters": parameters,
                        **rank_record(nuisance[rows], phase[rows]),
                    }
                )
    assert candidate.source_hashes() == sources
    candidate.benchmark.write_json(
        OUTPUT / "geometry_rank.json",
        {
            "sources": sources,
            "audit_sources": audit_sources(),
            "geometry_files": identity,
            "status": "diagnostic_only",
            "outcomes_read": False,
            "new_fits": 0,
            "scope": "Repeat0 all asymmetric rows, all5 outer training sets, all15 actual inner training subsets",
            "qualifications": [
                "Aggregate nuisance uses the modified boundary feature matrix at the evaluated state.",
                "Continuous derivatives relax discrete grids: local conditioning, not fixed-grid identifiability.",
                "At kappa0, q is unidentified and its derivative vanishes. Varying q gives a wider tangent cone.",
                "Rank uses original phase scales so projection roundoff is not counted as residual information.",
                "No outcome noise weighting or gain inference; fewer phase parameters do not repair saturated nuisance.",
            ],
            "records": records,
        },
    )
    for row in records:
        if row["subset"] == "all_asymmetric":
            print(
                json.dumps(
                    {"panel": row["panel"], "state": row["phase_state"], "projection": row["projected_phase"]["1e-06"]}
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("geometry", "mapping"), required=True)
    args = parser.parse_args()
    if args.mode == "geometry":
        geometry_receipt()
    else:
        mapping_receipt()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
