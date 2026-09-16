# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Fit an intervention-identified signed aggregate dose potential.

The candidate is a convex, phase-blind aggregate response fitted only to tied
policies. For proportional weights ``p_i`` and relative materialized dose
``r_i = e_i / e_i^prop = w_i / p_i``, the response is

    A(w) = b + sum_i p_i g_i (r_i - 1)
             + sum_f K_f sum_{i in f} p_i Phi_q(r_i),

where ``sum_i p_i g_i = 0``, ``K_f >= 0``, and ``Phi_q`` is a convex
Cressie-Read power-divergence generator with ``Phi_q(1)=Phi_q'(1)=0``.
The global-curvature ablation replaces the family coefficients by one shared
``K``. All candidate generators remain finite at zero dose.

The materially new identification argument is the independent conditional
epoch-dose panel. If bucket ``i`` is assigned multiplier ``m``, every other
bucket is rescaled by ``c_i(m)=(1-m p_i)/(1-p_i)``. The model therefore fits
all 39 curves jointly rather than treating each observed curve as a lookup
table. Structure is selected on the 60M full panel, x32 is held out, and the
Delphi 3e18 full panel is reserved for cross-scale validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import null_space
from scipy.special import xlogy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_bucket_epoch_dose_response as dose_launcher  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    swarm39_harness_20260725 as swarm39,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "intervention_identified_signed_dose_potential_20260731"

CANDIDATE_ID = "WSD80-SUR-073"
MODEL_ID = "intervention_identified_signed_dose_potential"
PROTOCOL_VERSION = "signed-dose-potential-v1"

# q=0 is the entropic primary form. The remaining values are a small frozen
# shape portfolio, not an invitation to extend the grid after seeing outcomes.
GENERATOR_ORDERS = (-0.5, 0.0, 0.5, 1.0)
CURVATURE_MODES = ("global", "family")
RIDGE_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)
SELECTION_MULTIPLIERS = (0.0, 0.25, 0.5, 2.0, 4.0, 8.0, 16.0)
EXTRAPOLATION_MULTIPLIER = 32.0
PRIMARY_SCALE = "60m"
CROSS_SCALE_VALIDATION = "delphi_3e18"
TARGETS = ("uncheatable", "table9")

OUTER_FOLDS = 5
OUTER_SEED = 7_317_301
BOOTSTRAP_DRAWS = 4_000
BOOTSTRAP_SEED = 7_317_302
OPTIMUM_BOOTSTRAP_REPLICATES = 200
OPTIMUM_BOOTSTRAP_SEED = 7_317_303

GATES = {
    "dose_linear_ablation_relative_rmse_max": 0.90,
    "family_curvature_retain_only_if_bootstrap_difference_high_max": 0.0,
    "nonentropy_shape_retain_only_if_bootstrap_difference_high_max": 0.0,
    "cross_scale_form_relative_rmse_max": 1.05,
    "cross_scale_strict_transfer_spearman_min": 0.50,
    "cross_scale_strict_transfer_sign_accuracy_min": 0.65,
    "300m_tied_oof_relative_to_reference_max": 1.05,
    "300m_raw_support_tv_max": 0.35,
    "300m_raw_max_bucket_weight_max": 0.30,
    "300m_raw_optimism_oof_rmse_multiple_max": 2.0,
    "300m_optimum_bootstrap_median_tv_max": 0.10,
    "all_selected_curvatures_positive": True,
    "generator_order_fold_mode_fraction_min": 0.60,
}

FROZEN_300M_REFERENCES = {
    "uncheatable": 0.004713404694656708,
    "table9": 0.010357273218151471,
}

PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Geometry:
    """Shared bucket order, proportional policy, and predeclared families."""

    domains: tuple[str, ...]
    proportional: np.ndarray
    family_index: np.ndarray
    family_names: tuple[str, ...]
    gauge_basis: np.ndarray


@dataclass(frozen=True)
class FeatureDesign:
    """Linear design and parameter metadata for a fixed generator."""

    matrix: np.ndarray
    utility_slice: slice
    curvature_slice: slice
    parameter_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def geometry() -> Geometry:
    domains, _c0, _c1, family_index, family_names = swarm39._exposure("300m_two_phase_fit")
    launcher_domains = tuple(dose_launcher.DOMAIN_NAMES)
    if set(domains) != set(launcher_domains):
        raise ValueError("Dose panel and 300M panel use different bucket sets")

    launcher_proportional = dose_launcher._proportional_weights()
    proportional = np.asarray([launcher_proportional[domain] for domain in domains], dtype=float)
    if not np.isclose(proportional.sum(), 1.0, atol=1e-12):
        raise ValueError("Proportional policy is not normalized")

    # Every signed utility vector has a unique representative in the gauge
    # p^T g=0. The null-space columns span that 38-dimensional subspace.
    gauge_basis = null_space(proportional[None, :])
    if gauge_basis.shape != (len(domains), len(domains) - 1):
        raise ValueError(f"Unexpected gauge basis shape: {gauge_basis.shape}")
    if not np.allclose(proportional @ gauge_basis, 0.0, atol=1e-12):
        raise ValueError("Utility basis violates the proportional gauge")
    return Geometry(
        domains=domains,
        proportional=proportional,
        family_index=family_index,
        family_names=family_names,
        gauge_basis=gauge_basis,
    )


def power_divergence(relative_dose: np.ndarray, order: float) -> np.ndarray:
    """Return the finite-at-zero convex Cressie-Read generator."""
    values = np.asarray(relative_dose, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("Relative materialized dose must be nonnegative")
    if order <= -1.0:
        raise ValueError("Orders at or below -1 are not finite at zero dose")
    if order == 0.0:
        return xlogy(values, values) - values + 1.0
    return (np.power(values, order + 1.0) - (order + 1.0) * values + order) / (order * (order + 1.0))


def feature_design(
    weights: np.ndarray,
    geometry_: Geometry,
    *,
    generator_order: float,
    curvature_mode: str,
) -> FeatureDesign:
    """Construct the identified linear head for a fixed nonlinear generator."""
    policies = np.asarray(weights, dtype=float)
    if policies.ndim != 2 or policies.shape[1] != len(geometry_.domains):
        raise ValueError(f"Expected an n x {len(geometry_.domains)} policy matrix")
    if np.any(policies < -1e-12) or not np.allclose(policies.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Policies must lie on the mixture simplex")
    policies = np.maximum(policies, 0.0)

    relative_dose = policies / geometry_.proportional[None, :]
    utility = (policies - geometry_.proportional[None, :]) @ geometry_.gauge_basis
    divergence = geometry_.proportional[None, :] * power_divergence(relative_dose, generator_order)

    if curvature_mode == "global":
        curvature = divergence.sum(axis=1, keepdims=True)
        curvature_names = ("curvature::global",)
    elif curvature_mode == "family":
        curvature = np.column_stack(
            [divergence[:, geometry_.family_index == index].sum(axis=1) for index in range(len(geometry_.family_names))]
        )
        curvature_names = tuple(f"curvature::{name}" for name in geometry_.family_names)
    else:
        raise ValueError(f"Unknown curvature mode: {curvature_mode}")

    utility_names = tuple(f"utility_gauge::{index:02d}" for index in range(utility.shape[1]))
    matrix = np.column_stack([np.ones(len(policies)), utility, curvature])
    return FeatureDesign(
        matrix=matrix,
        utility_slice=slice(1, 1 + utility.shape[1]),
        curvature_slice=slice(1 + utility.shape[1], matrix.shape[1]),
        parameter_names=("intercept", *utility_names, *curvature_names),
    )


def recover_bucket_utility(gauge_coefficients: np.ndarray, geometry_: Geometry) -> np.ndarray:
    utility = geometry_.gauge_basis @ np.asarray(gauge_coefficients, dtype=float)
    if not np.isclose(geometry_.proportional @ utility, 0.0, atol=1e-10):
        raise ValueError("Recovered bucket utilities violate the frozen gauge")
    return utility


def panel_design() -> tuple[list[dose_launcher.EpochSweepPoint], Geometry, np.ndarray]:
    geometry_ = geometry()
    points = dose_launcher.build_points()
    weights = np.asarray(
        [
            [point.phase_weights[dose_launcher.PHASE_NAMES[0]][domain] for domain in geometry_.domains]
            for point in points
        ],
        dtype=float,
    )
    phase1 = np.asarray(
        [
            [point.phase_weights[dose_launcher.PHASE_NAMES[1]][domain] for domain in geometry_.domains]
            for point in points
        ],
        dtype=float,
    )
    if not np.allclose(weights, phase1, atol=1e-12):
        raise ValueError("Conditional epoch-dose panel contains an asymmetric policy")
    return points, geometry_, weights


def design_payload(
    points: list[dose_launcher.EpochSweepPoint],
    geometry_: Geometry,
    weights: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for point, row in zip(points, weights, strict=True):
        rows.append(
            {
                "point_id": point.point_id,
                "point_kind": point.point_kind,
                "focal_domain": point.focal_domain,
                "epoch_multiplier": point.epoch_multiplier,
                "weights": row,
            }
        )
    return {
        "domains": geometry_.domains,
        "family_index": geometry_.family_index,
        "family_names": geometry_.family_names,
        "proportional": geometry_.proportional,
        "rows": rows,
    }


def scaled_condition_number(matrix: np.ndarray) -> float:
    scale = np.maximum(np.sqrt(np.mean(matrix**2, axis=0)), 1e-12)
    return float(np.linalg.cond(matrix / scale[None, :]))


def preflight_payload() -> dict[str, Any]:
    points, geometry_, weights = panel_design()
    multiplier_counts: dict[str, int] = {}
    for point in points:
        key = f"{point.epoch_multiplier:g}"
        multiplier_counts[key] = multiplier_counts.get(key, 0) + 1

    candidate_rows: list[dict[str, Any]] = []
    for order in GENERATOR_ORDERS:
        probe = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0, 8.0, 32.0])
        values = power_divergence(probe, order)
        if np.any(values < -1e-12) or abs(float(values[3])) > 1e-12:
            raise ValueError(f"Generator invariants failed for order {order:g}")
        for mode in CURVATURE_MODES:
            design = feature_design(weights, geometry_, generator_order=order, curvature_mode=mode)
            rank = int(np.linalg.matrix_rank(design.matrix))
            if rank != design.matrix.shape[1]:
                raise ValueError(f"Rank-deficient {mode}/q={order:g} design: {rank}/{design.matrix.shape[1]}")
            candidate_rows.append(
                {
                    "generator_order": order,
                    "curvature_mode": mode,
                    "rows": design.matrix.shape[0],
                    "parameters": design.matrix.shape[1],
                    "rank": rank,
                    "scaled_condition_number": scaled_condition_number(design.matrix),
                    "generator_at_zero": float(values[0]),
                    "generator_at_32": float(values[-1]),
                }
            )

    focal_checks = []
    for point, row in zip(points, weights, strict=True):
        if point.focal_domain is None:
            continue
        focal_index = geometry_.domains.index(point.focal_domain)
        ratio = row[focal_index] / geometry_.proportional[focal_index]
        if not np.isclose(ratio, point.epoch_multiplier, atol=1e-10):
            raise ValueError(f"Relative-dose mismatch for {point.point_id}")
        complement = np.delete(row / geometry_.proportional, focal_index)
        if not np.allclose(complement, point.complement_scale, atol=1e-10):
            raise ValueError(f"Complement renormalization mismatch for {point.point_id}")
        focal_checks.append(point.point_id)

    return {
        "candidate_id": CANDIDATE_ID,
        "points": len(points),
        "buckets": len(geometry_.domains),
        "families": geometry_.family_names,
        "physically_tied": True,
        "utility_gauge_rank": int(np.linalg.matrix_rank(geometry_.gauge_basis)),
        "utility_gauge_max_error": float(np.abs(geometry_.proportional @ geometry_.gauge_basis).max()),
        "multiplier_counts": multiplier_counts,
        "selection_rows": int(
            sum(
                point.point_kind == "focal_bucket_dose" and point.epoch_multiplier in SELECTION_MULTIPLIERS
                for point in points
            )
        ),
        "x32_holdout_rows": int(sum(point.epoch_multiplier == EXTRAPOLATION_MULTIPLIER for point in points)),
        "focal_formula_checks": len(focal_checks),
        "candidate_designs": candidate_rows,
        "design_sha256": canonical_hash(design_payload(points, geometry_, weights)),
    }


def protocol_payload() -> dict[str, Any]:
    sources = (
        Path(__file__),
        Path(dose_launcher.__file__),
        Path(swarm39.__file__),
    )
    payload: dict[str, Any] = {
        "candidate_id": CANDIDATE_ID,
        "model_id": MODEL_ID,
        "version": PROTOCOL_VERSION,
        "purpose": "Identify bounded aggregate geometry from independent tied conditional-dose interventions",
        "equation": (
            "A(w)=b+sum_i p_i*g_i*(r_i-1)+sum_f K_f*sum_{i in f}p_i*Phi_q(r_i); "
            "r_i=e_i/e_i_prop=w_i/p_i; p^T g=0; K_f>=0"
        ),
        "generator": {
            "formula": "Phi_q(r)=[r^(q+1)-(q+1)r+q]/[q(q+1)]; Phi_0(r)=r log r-r+1",
            "orders": GENERATOR_ORDERS,
            "finite_at_zero": True,
            "convexity": "Phi_q''(r)=r^(q-1)>0 for r>0",
            "primary_ablation": "q=0 entropic generator",
        },
        "mechanistic_interpretation": {
            "relative_dose": "dimensionless materialized epochs relative to proportional training",
            "bucket_utility": "target-specific local BPB slope at proportional allocation, identified up to a constant",
            "curvature": "nonnegative global or predeclared-family resistance to finite dose substitution",
            "response": "BPB; bucket utility and curvature coefficients have BPB units",
        },
        "nearest_prior_routes": (
            "Power-Ridge",
            "prior_L",
            "prior_T",
            "prior_V",
            "WSD80-SUR-064",
            "WSD80-SUR-065",
            "WSD80-SUR-066",
            "WSD80-SUR-067",
            "WSD80-SUR-072",
        ),
        "material_novelty": (
            "The functional shape and signed bucket utility are selected from an independent 39-direction tied "
            "conditional-dose experiment. The compensating change in the other 38 buckets is modeled exactly. "
            "The selected form is frozen before any 300M endpoint fit or raw optimization."
        ),
        "candidate_portfolio": {
            "curvature_modes": CURVATURE_MODES,
            "generator_orders": GENERATOR_ORDERS,
            "ridge_grid": RIDGE_GRID,
            "linear_ablation": "K_f=0",
            "global_curvature_ablation": "K_f=K for all families",
            "entropy_ablation": "q=0",
        },
        "data_use": {
            "pilot": "already exposed; noise estimation and affine cross-scale mapping only",
            "primary_structure_selection": f"{PRIMARY_SCALE} full outcomes at multipliers {SELECTION_MULTIPLIERS}",
            "excluded_from_selection": (
                f"multiplier {EXTRAPOLATION_MULTIPLIER:g} and all {CROSS_SCALE_VALIDATION} full outcomes"
            ),
            "cross_scale_validation": CROSS_SCALE_VALIDATION,
            "300m": (
                "known development data evaluated only after the 60M structure is frozen; report strict "
                "source-potential transfer and a practical nested refit of the frozen form separately"
            ),
            "sealed_exclusion": "Never inspect any path containing targeted_pairwise",
        },
        "selection": {
            "targets": TARGETS,
            "primary_scale": PRIMARY_SCALE,
            "folds": (
                "Leave one intervention multiplier out across buckets for nonlinear shape selection; nested "
                "five-fold bucket-stratified rows for ridge and coefficient prediction"
            ),
            "outer_folds": OUTER_FOLDS,
            "outer_seed": OUTER_SEED,
            "tie_break": "lower RMSE, then q=0, then global curvature, then larger ridge",
            "retain_extension_only_beyond_uncertainty": True,
        },
        "cross_scale_tests": (
            "Fixed-form Delphi refit with no structural reselection",
            "Strict 60M potential transfer with only an affine positive BPB scale estimated from exposed pilot rows",
            "Exclude pilot-overlap coordinates from strict-transfer scoring",
        ),
        "300m_tests": (
            "strict source-potential transfer with intercept and positive scale only",
            "nested refit of the frozen form on physically tied 300M policies",
            "raw tied optimum before deployment regularization",
            "bootstrap coefficient and optimum stability",
        ),
        "gates": GATES,
        "frozen_300m_references": FROZEN_300M_REFERENCES,
        "bootstrap": {
            "metric_draws": BOOTSTRAP_DRAWS,
            "metric_seed": BOOTSTRAP_SEED,
            "optimum_replicates": OPTIMUM_BOOTSTRAP_REPLICATES,
            "optimum_seed": OPTIMUM_BOOTSTRAP_SEED,
            "unit": "focal bucket for intervention comparisons; correspondence group for 300M",
        },
        "decision": (
            "Failure of the 60M nonlinear dose gate blocks the mechanism. Passing licenses one frozen read of "
            "Delphi full outcomes and 300M evaluation. No post-outcome grid extension is allowed."
        ),
        "source_hashes": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in sources},
        "preflight": preflight_payload(),
    }
    payload["protocol_sha256"] = canonical_hash(payload)
    return payload


def write_if_absent_or_equal(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Frozen artifact differs from current protocol: {path}")
        return
    path.write_text(content)


def freeze_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json_ready(protocol_payload())
    write_if_absent_or_equal(
        output_dir / "protocol.json",
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
    )
    preflight = protocol["preflight"]
    write_if_absent_or_equal(
        output_dir / "preflight.json",
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
    )
    ledger_path = output_dir / "data_use_ledger.csv"
    if not ledger_path.exists():
        with ledger_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "timestamp",
                    "candidate_id",
                    "stage",
                    "outcomes_inspected",
                    "purpose",
                    "protocol_sha256",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": "2026-07-31T09:30:00-07:00",
                    "candidate_id": CANDIDATE_ID,
                    "stage": "preregistered",
                    "outcomes_inspected": "pilot only; no full 60M or Delphi outcomes",
                    "purpose": "freeze equations, selection, transfer, and raw-optimum gates",
                    "protocol_sha256": protocol["protocol_sha256"],
                }
            )
    report = f"""# Intervention-identified signed dose potential

Candidate `{CANDIDATE_ID}` is preregistered under protocol
`{protocol["protocol_sha256"]}` before inspecting either full conditional-dose
panel.

The algebraic preflight contains {preflight["points"]} tied policies over
{preflight["buckets"]} buckets. Every frozen candidate design is full rank.
Multiplier `{EXTRAPOLATION_MULTIPLIER:g}` contributes
{preflight["x32_holdout_rows"]} held-out extrapolation rows and cannot select the
model.

No outcome evaluation has been performed by this script.
"""
    write_if_absent_or_equal(output_dir / "report.md", report)
    print(json.dumps({"protocol_sha256": protocol["protocol_sha256"], "preflight": preflight}, indent=2))


def verify_protocol(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Freeze the protocol before evaluation: {path}")
    frozen = json.loads(path.read_text())
    current = json_ready(protocol_payload())
    if frozen != current:
        raise ValueError("Current source, design, or protocol differs from the frozen artifact")
    return frozen


def evaluate(output_dir: Path) -> None:
    protocol = verify_protocol(output_dir)
    raise RuntimeError(
        "Outcome loading is intentionally unavailable until the frozen 60M panel is complete. "
        f"Protocol {protocol['protocol_sha256']} is ready for the evaluation implementation."
    )


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        freeze_protocol(args.output_dir)
        return
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
