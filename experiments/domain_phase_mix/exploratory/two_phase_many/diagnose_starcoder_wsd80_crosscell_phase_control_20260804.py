# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///
"""Test whether a shared scale-conditioned phase law transfers across WSD80 cells.

This is a development diagnostic, not a new surrogate candidate. Each held-out
``(N, D)`` cell contributes its tied diagonal, which identifies that cell's
aggregate response, but none of its untied outcomes are used to fit or select
the phase-control law. The phase law is selected on the remaining cells with a
nested leave-cell-out ridge comparison.

The diagnostic reopens prior optimizer-clock work only through a new
identification design: ten dense cells vary model size and token horizon
independently. It does not retune the rejected autonomous optimizer-time,
fast/slow, or consolidation-cascade transitions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import chebvander
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
INPUT_DIR = (
    SCRIPT_DIR
    / "reference_outputs"
    / "starcoder_wsd80_matched_nd_stage1_20260731"
    / "stage3_dense_surface_results_20260802"
)
INPUT_CSV = INPUT_DIR / "combined_discovery_observations.csv"
REFERENCE_SURFACES_CSV = INPUT_DIR / "fitted_surface_candidates.csv"
SCHEDULE_SOURCE = SCRIPT_DIR.parents[1] / "launch_starcoder_wsd_80_20_surface.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "wsd80_crosscell_phase_control_v3_20260804"

PROTOCOL_VERSION = "wsd80-crosscell-phase-control-v3"
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
WARMUP_FRACTION = 0.01
AGGREGATE_SCALE = 0.25
SPINE_DEGREE = 4
SPINE_RIDGES = (0.0, 1e-6, 1e-4, 1e-2, 1.0)
PHASE_RIDGES = (1e-4, 1e-2, 1.0, 100.0)
GRID_SIZE = 201
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_804
REFERENCE_GAIN_PROBABILITY_THRESHOLD = 0.95
CLOCK_SELECTION_STABILITY_FOLDS = 8
SIGN_FLIP_ALPHA = 0.05
SUPPORT_BOUNDARY_TOLERANCE = 0.01

BASE_FEATURE_NAMES = (
    "d",
    "u*d",
    "u^2*d",
    "d^2",
    "u*d^2",
)
CLOCK_COLUMNS = {
    "token_horizon": "log_materialized_tokens",
    "total_tpp": "log_total_tpp",
    "nonembedding_tpp": "log_nonembedding_tpp",
    "model_size": "log_total_parameters",
}


@dataclass(frozen=True)
class PhaseSpec:
    """A fixed phase-residual feature family."""

    name: str
    base: str
    clocks: tuple[str, ...]
    offset: str = "zero"


@dataclass(frozen=True)
class SpineFit:
    """One cell's tied quartic-ridge aggregate spine."""

    coefficients: np.ndarray
    ridge: float
    loocv_rmse: float
    optimum: float

    def predict(self, aggregate: np.ndarray) -> np.ndarray:
        return chebyshev_design(aggregate) @ self.coefficients


@dataclass(frozen=True)
class PhaseFit:
    """A ridge phase law and its train-only transformations."""

    spec: PhaseSpec
    ridge: float
    coefficients: np.ndarray
    feature_scale: np.ndarray
    clock_center: dict[str, float]
    clock_scale: dict[str, float]


PHASE_SPECS = (
    PhaseSpec("odd_linear", "odd", ()),
    PhaseSpec("aggregate_taylor", "taylor", ()),
    PhaseSpec("lr_dose_plus_taylor", "taylor", (), "lr_dose"),
    PhaseSpec("lr_dose_plus_taylor_token_horizon", "taylor", ("token_horizon",), "lr_dose"),
    PhaseSpec("lr_dose_plus_taylor_total_tpp", "taylor", ("total_tpp",), "lr_dose"),
    PhaseSpec("lr_dose_plus_taylor_nonembedding_tpp", "taylor", ("nonembedding_tpp",), "lr_dose"),
    PhaseSpec("lr_dose_plus_taylor_model_size", "taylor", ("model_size",), "lr_dose"),
    PhaseSpec("lr_dose_plus_taylor_joint_d_n", "taylor", ("token_horizon", "model_size"), "lr_dose"),
)
CLOCK_PHASE_SPECS = tuple(spec for spec in PHASE_SPECS if spec.clocks)
CLOCK_BASELINE_NAME = "lr_dose_plus_taylor"
CLOCK_SELECTOR_NAME = "nested_clock_selector"


def protocol_payload() -> dict[str, Any]:
    """Return the frozen outcome-evaluation protocol."""
    source_sha256 = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    input_sha256 = hashlib.sha256(INPUT_CSV.read_bytes()).hexdigest()
    reference_sha256 = hashlib.sha256(REFERENCE_SURFACES_CSV.read_bytes()).hexdigest()
    schedule_sha256 = hashlib.sha256(SCHEDULE_SOURCE.read_bytes()).hexdigest()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_sha256": source_sha256,
        "input_sha256": input_sha256,
        "reference_surface_sha256": reference_sha256,
        "schedule_source_sha256": schedule_sha256,
        "status": "exposed development diagnostic; not confirmatory evidence",
        "estimand": (
            "Untied Programming-Languages BPB residual after fitting a held-cell tied aggregate spine. "
            "The held cell's untied outcomes are excluded from phase-law fitting and ridge selection."
        ),
        "outer_split": "leave one complete (N,D) cell out",
        "inner_split": "leave one remaining cell out for ridge and clock-family selection",
        "held_cell_data_allowed": "tied rows, declared N,D metadata, and untied design coordinates; no untied outcomes",
        "aggregate_spine": {
            "basis": f"degree-{SPINE_DEGREE} Chebyshev polynomial in nominal 80/20 aggregate",
            "selection": "leave-one-tied-row-out RMSE",
            "ridges": SPINE_RIDGES,
        },
        "phase_coordinate": {
            "aggregate": "a=0.8*p0+0.2*p1",
            "contrast": "d=p1-p0",
            "centered_aggregate": f"u=(a-a_tied_star)/{AGGREGATE_SCALE}",
            "base_features": BASE_FEATURE_NAMES,
        },
        "phase_weighted_dose_null": {
            "definition": "z=m0*p0+m1*p1, where m0 and m1 are exact peak-normalized LR-integral shares",
            "schedule": "1% linear warmup, stable through the 80% boundary, then cosine decay to zero",
            "prediction": "A_cell(z)-A_cell(a), using only the held cell's tied aggregate spine",
            "role": "mandatory mechanistic null evaluated before any learned phase correction",
        },
        "phase_specs": [asdict(spec) for spec in PHASE_SPECS],
        "clock_selection": {
            "candidate_specs": tuple(spec.name for spec in CLOCK_PHASE_SPECS),
            "selection": "minimum cell-balanced inner leave-cell-out RMSE within each outer fold",
            "reported_model": CLOCK_SELECTOR_NAME,
            "stability_requirement": (
                f"one candidate_spec must be selected in at least {CLOCK_SELECTION_STABILITY_FOLDS} of 10 outer folds"
            ),
        },
        "phase_ridges": PHASE_RIDGES,
        "primary_metrics": (
            "untied residual RMSE",
            "mean held-cell RMSE and improvement over the LR-dose-plus-Taylor baseline",
            "observed-on-predicted calibration slope",
            "phase-effect Spearman",
            "reference-surface optimum coordinate error",
            "reference-surface phase-gain error",
            "exact paired sign-flip test over the ten held-cell RMSE differences",
            "descriptive paired cell bootstrap interval for mean RMSE difference",
        ),
        "optimization_support": {
            "domain": "convex hull of all observed untied (aggregate, contrast) coordinates",
            "boundary_tolerance": SUPPORT_BOUNDARY_TOLERANCE,
            "reason": "prevent the quadratic phase law from winning through unsupported square-boundary extrapolation",
        },
        "reference_optimum_gate": {
            "minimum_bootstrap_positive_gain_probability": REFERENCE_GAIN_PROBABILITY_THRESHOLD,
            "full_panel_metrics": "descriptive only",
        },
        "secondary_sensitivity": (
            "leave one scaling track out, retaining only the shared root cell from that track",
            "leave one nonzero rung out across all three tracks",
            "within-cell leave-one-row-out basis oracle to diagnose representational underfit",
        ),
        "promotion_gate": {
            "scope": "licenses a scale covariate for a later candidate; does not promote a model",
            "requirements": (
                "at least 5% lower mean held-cell RMSE than lr_dose_plus_taylor",
                "lower RMSE in at least 7 of 10 held cells",
                f"exact paired sign-flip p-value is at most {SIGN_FLIP_ALPHA}",
                f"one clock family is selected in at least {CLOCK_SELECTION_STABILITY_FOLDS} of 10 outer folds",
                "no worse mean optimum coordinate error on reference cells with positive-gain probability >=0.95",
                "no worse mean absolute phase-gain error on reference cells with positive-gain probability >=0.95",
                "no more observed-support-boundary optima than lr_dose_plus_taylor",
            ),
        },
        "prior_route_boundary": {
            "blocked_controls": ("OTTPF", "OTFSC", "AAGF", "MCCF"),
            "material_difference": (
                "Those routes fit autonomous token/optimizer-time state transitions on two old surfaces. "
                "This diagnostic asks only whether a low-dimensional phase residual transfers across ten "
                "cells that independently vary N and D. It introduces no autonomous temporal state."
            ),
            "forbidden_repair": (
                "Do not add a learned clock exponent, track indicator, per-cell phase head, output calibrator, "
                "or additional phase basis after inspecting these results."
            ),
        },
        "known_coordinate_caveat": (
            "Aggregate uses the experiment's nominal 0.8/0.2 design. Physical epoch accounting in the "
            "historical 3814-step panel uses 3040/3814; this diagnostic does not silently relabel fibers."
        ),
        "adaptive_design_caveat": (
            "Stage-2 and Stage-3 coordinates were selected using earlier outcomes within each cell. "
            "Candidate and baseline "
            "are compared on identical rows, but absolute RMSE averages over cell-specific development supports."
        ),
    }


def write_protocol(output_dir: Path) -> Path:
    """Persist the preregistration before fitting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# WSD80 Cross-Cell Phase-Control Diagnostic",
        "",
        f"Protocol `{payload['source_sha256']}` under `{PROTOCOL_VERSION}`.",
        "",
        "This is an exposed development diagnostic. It does not confirm a surrogate.",
        "",
        "## Question",
        "",
        "After a held cell's tied diagonal identifies its aggregate response, does a phase law learned only",
        "from other `(N,D)` cells predict its untied surface? In particular, does one declared scale clock",
        "improve over an aggregate-conditioned odd/even Taylor law?",
        "",
        "## Models",
        "",
        "The tied spine is a degree-four Chebyshev ridge selected by tied-row leave-one-out CV. Untied",
        "predictions first test the exact LR-integrated phase-weighted-dose null",
        "`A(m0*p0+m1*p1)-A(a)`. Learned corrections use `d`, `u*d`, `u^2*d`, `d^2`, and",
        f"`u*d^2`, where `d=p1-p0` and `u=(a-a*)/{AGGREGATE_SCALE}`. Clock models add only",
        "clock-by-correction interactions. Clock-family choice is nested inside every outer fold.",
        "",
        "No held-cell untied outcome may enter fitting or ridge selection. No track indicators, per-cell",
        "phase heads, learned clock exponents, or post-hoc output calibration are admissible.",
        "Leave-track-out, leave-rung-out, a paired cell-cluster bootstrap, and a within-cell basis oracle",
        "are frozen sensitivity analyses rather than opportunities to modify the candidate family.",
        "Raw optimum diagnostics are restricted to the pooled observed untied convex hull in `(a,d)`.",
        "Only reference cells with bootstrap positive-gain probability at least 0.95 enter optimum gates.",
        "A named clock is licensed only if one family is selected in at least "
        f"{CLOCK_SELECTION_STABILITY_FOLDS}/10 folds.",
        "The cell bootstrap is descriptive; the gate uses the exact paired sign-flip test.",
        "",
        "## Interpretation boundary",
        "",
        "A pass licenses a scale covariate for a subsequent mechanistic candidate. It does not reopen the",
        "rejected optimizer-time task-potential, fast/slow, activated-mobility, or cascade routes, and it",
        "does not establish causality for the winning clock.",
    ]
    (output_dir / "preregistration.md").write_text("\n".join(lines) + "\n")
    return protocol_path


def chebyshev_design(aggregate: np.ndarray) -> np.ndarray:
    """Degree-four Chebyshev basis on the unit mixture interval."""
    return chebvander(2.0 * np.asarray(aggregate, dtype=float) - 1.0, SPINE_DEGREE)


def fit_spine_coefficients(aggregate: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    design = chebyshev_design(aggregate)
    degree_penalty = np.diag(np.arange(SPINE_DEGREE + 1, dtype=float))
    degree_penalty[0, 0] = 0.0
    augmented_design = np.vstack([design, np.sqrt(ridge) * degree_penalty])
    augmented_target = np.concatenate([target, np.zeros(SPINE_DEGREE + 1)])
    return np.linalg.lstsq(augmented_design, augmented_target, rcond=None)[0]


def fit_spine(aggregate: np.ndarray, target: np.ndarray) -> SpineFit:
    """Select and fit one tied aggregate spine without untied outcomes."""
    if len(aggregate) <= SPINE_DEGREE + 2:
        raise ValueError(f"Need more tied rows than coefficients, got {len(aggregate)}")
    scores: list[tuple[float, float]] = []
    for ridge in SPINE_RIDGES:
        predicted = np.empty(len(target), dtype=float)
        for held in range(len(target)):
            keep = np.arange(len(target)) != held
            coefficients = fit_spine_coefficients(aggregate[keep], target[keep], ridge)
            predicted[held] = (chebyshev_design(aggregate[[held]]) @ coefficients).item()
        scores.append((float(np.sqrt(np.mean(np.square(predicted - target)))), ridge))
    loocv_rmse, ridge = min(scores)
    coefficients = fit_spine_coefficients(aggregate, target, ridge)
    grid = np.linspace(0.0, 1.0, 4001)
    optimum = float(grid[np.argmin(chebyshev_design(grid) @ coefficients)])
    return SpineFit(coefficients, ridge, loocv_rmse, optimum)


def phase_learning_rate_masses(total_steps: int, boundary_step: int) -> tuple[float, float]:
    """Integrate the exact warmup-stable-cosine schedule over both phases."""
    if not 0 < boundary_step < total_steps:
        raise ValueError(f"Invalid phase boundary {boundary_step} for {total_steps} steps")
    warmup_steps = int(total_steps * WARMUP_FRACTION)
    steps = np.arange(total_steps, dtype=float)
    learning_rate = np.ones(total_steps, dtype=float)
    if warmup_steps > 0:
        warmup = steps < warmup_steps
        learning_rate[warmup] = steps[warmup] / warmup_steps
    decay = steps >= boundary_step
    decay_steps = total_steps - boundary_step
    progress = np.clip((steps[decay] - boundary_step) / decay_steps, 0.0, 1.0)
    learning_rate[decay] = 0.5 * (1.0 + np.cos(np.pi * progress))
    total_mass = float(learning_rate.sum())
    early_mass = float(learning_rate[:boundary_step].sum()) / total_mass
    late_mass = float(learning_rate[boundary_step:].sum()) / total_mass
    if not np.isclose(early_mass + late_mass, 1.0, atol=1e-12):
        raise AssertionError("Phase LR masses do not sum to one")
    return early_mass, late_mass


def optimization_support_equations(frame: pd.DataFrame) -> np.ndarray:
    """Return normalized half-space equations for observed untied `(a, d)` support."""
    untied = frame.loc[~np.isclose(frame["contrast"], 0.0, atol=1e-12)]
    points = untied[["aggregate", "contrast"]].drop_duplicates().to_numpy(dtype=float)
    hull = ConvexHull(points)
    equations = hull.equations.copy()
    equations /= np.linalg.norm(equations[:, :-1], axis=1, keepdims=True)
    return equations


def support_mask(aggregate: np.ndarray, contrast: np.ndarray, equations: np.ndarray) -> np.ndarray:
    """Return whether coordinates lie inside the frozen pooled convex support."""
    points = np.column_stack([aggregate, contrast])
    signed_distance = points @ equations[:, :-1].T + equations[:, -1]
    return np.max(signed_distance, axis=1) <= 1e-10


def support_margin(aggregate: np.ndarray, contrast: np.ndarray, equations: np.ndarray) -> np.ndarray:
    """Return distance to the nearest convex-support facet for inside points."""
    points = np.column_stack([aggregate, contrast])
    signed_distance = points @ equations[:, :-1].T + equations[:, -1]
    return -np.max(signed_distance, axis=1)


def load_frame() -> tuple[pd.DataFrame, dict[str, SpineFit]]:
    """Load the dense panel and attach tied-only aggregate predictions."""
    frame = pd.read_csv(INPUT_CSV)
    required = {
        "cell_id",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "starcoder_bpb",
        "total_parameters",
        "non_embedding_parameters",
        "materialized_tokens",
        "total_steps",
        "boundary_step",
        "rung",
        "track_memberships",
        "policy_class",
        "final_metric_step",
        "expected_final_metric_step",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    frame = frame.copy()
    frame["aggregate"] = PHASE_0_FRACTION * frame["phase_0_starcoder"] + PHASE_1_FRACTION * frame["phase_1_starcoder"]
    frame["contrast"] = frame["phase_1_starcoder"] - frame["phase_0_starcoder"]
    tied = np.isclose(frame["contrast"].to_numpy(dtype=float), 0.0, atol=1e-12)
    if not np.array_equal(tied, frame["policy_class"].eq("tied").to_numpy()):
        raise ValueError("policy_class disagrees with the phase contrast")
    if not np.allclose(frame["boundary_step"] / frame["total_steps"], PHASE_0_FRACTION, atol=1e-12):
        raise ValueError("The panel does not realize the declared 80/20 phase boundary")
    if not frame["final_metric_step"].eq(frame["expected_final_metric_step"]).all():
        raise ValueError("The panel contains a nonterminal target observation")
    frame["total_tpp"] = frame["materialized_tokens"] / frame["total_parameters"]
    frame["nonembedding_tpp"] = frame["materialized_tokens"] / frame["non_embedding_parameters"]
    for column in ("materialized_tokens", "total_parameters", "total_tpp", "nonembedding_tpp"):
        frame[f"log_{column}"] = np.log(frame[column].astype(float))

    spines: dict[str, SpineFit] = {}
    spine_prediction = np.empty(len(frame), dtype=float)
    tied_optimum = np.empty(len(frame), dtype=float)
    for cell_id, indices in frame.groupby("cell_id", sort=True).groups.items():
        cell = frame.loc[indices]
        tied = np.isclose(cell["contrast"].to_numpy(dtype=float), 0.0, atol=1e-12)
        spine = fit_spine(
            cell.loc[tied, "aggregate"].to_numpy(dtype=float),
            cell.loc[tied, "starcoder_bpb"].to_numpy(dtype=float),
        )
        spines[str(cell_id)] = spine
        spine_prediction[indices] = spine.predict(cell["aggregate"].to_numpy(dtype=float))
        tied_optimum[indices] = spine.optimum
    frame["spine_prediction"] = spine_prediction
    frame["phase_residual"] = frame["starcoder_bpb"] - frame["spine_prediction"]
    frame["tied_optimum"] = tied_optimum
    frame["u"] = (frame["aggregate"] - frame["tied_optimum"]) / AGGREGATE_SCALE
    lr_early_mass = np.empty(len(frame), dtype=float)
    lr_late_mass = np.empty(len(frame), dtype=float)
    for _, indices in frame.groupby("cell_id", sort=True).groups.items():
        cell = frame.loc[indices]
        total_steps = int(cell["total_steps"].iloc[0])
        boundary_step = int(cell["boundary_step"].iloc[0])
        if not cell["total_steps"].eq(total_steps).all() or not cell["boundary_step"].eq(boundary_step).all():
            raise ValueError("A cell contains inconsistent optimizer schedules")
        early_mass, late_mass = phase_learning_rate_masses(total_steps, boundary_step)
        lr_early_mass[indices] = early_mass
        lr_late_mass[indices] = late_mass
    frame["lr_phase_0_mass"] = lr_early_mass
    frame["lr_phase_1_mass"] = lr_late_mass
    frame["lr_effective_weight"] = (
        frame["lr_phase_0_mass"] * frame["phase_0_starcoder"] + frame["lr_phase_1_mass"] * frame["phase_1_starcoder"]
    )
    lr_dose_prediction = np.empty(len(frame), dtype=float)
    for cell_id, indices in frame.groupby("cell_id", sort=True).groups.items():
        lr_dose_prediction[indices] = spines[str(cell_id)].predict(
            frame.loc[indices, "lr_effective_weight"].to_numpy(dtype=float)
        ) - frame.loc[indices, "spine_prediction"].to_numpy(dtype=float)
    frame["lr_dose_prediction"] = lr_dose_prediction
    frame["track_memberships_parsed"] = frame["track_memberships"].map(ast.literal_eval)
    return frame, spines


def clock_statistics(frame: pd.DataFrame, spec: PhaseSpec) -> tuple[dict[str, float], dict[str, float]]:
    """Train-cell mean and scale for declared clock interactions."""
    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    cells = frame.drop_duplicates("cell_id")
    for clock in spec.clocks:
        column = CLOCK_COLUMNS[clock]
        values = cells[column].to_numpy(dtype=float)
        center[clock] = float(values.mean())
        spread = float(values.std(ddof=0))
        scale[clock] = max(spread, 1e-12)
    return center, scale


def base_phase_design(frame: pd.DataFrame, base: str) -> tuple[np.ndarray, list[str]]:
    d = frame["contrast"].to_numpy(dtype=float)
    if base == "odd":
        return d[:, None], ["d"]
    if base != "taylor":
        raise ValueError(f"Unknown base feature family {base}")
    u = frame["u"].to_numpy(dtype=float)
    design = np.column_stack([d, u * d, np.square(u) * d, np.square(d), u * np.square(d)])
    return design, list(BASE_FEATURE_NAMES)


def phase_design(
    frame: pd.DataFrame,
    spec: PhaseSpec,
    center: dict[str, float],
    scale: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Build a phase law that is exactly zero for tied policies."""
    base, base_names = base_phase_design(frame, spec.base)
    blocks = [base]
    names = list(base_names)
    for clock in spec.clocks:
        value = (frame[CLOCK_COLUMNS[clock]].to_numpy(dtype=float) - center[clock]) / scale[clock]
        blocks.append(base * value[:, None])
        names.extend([f"{clock}*{name}" for name in base_names])
    return np.column_stack(blocks), names


def ridge_coefficients(design: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a no-intercept ridge after train-only RMS feature scaling."""
    feature_scale = np.maximum(np.sqrt(np.mean(np.square(design), axis=0)), 1e-10)
    scaled = design / feature_scale
    penalty = np.sqrt(ridge) * np.eye(scaled.shape[1])
    coefficients = np.linalg.lstsq(
        np.vstack([scaled, penalty]),
        np.concatenate([target, np.zeros(scaled.shape[1])]),
        rcond=None,
    )[0]
    return coefficients, feature_scale


def phase_offset(frame: pd.DataFrame, offset: str) -> np.ndarray:
    if offset == "zero":
        return np.zeros(len(frame), dtype=float)
    if offset == "lr_dose":
        return frame["lr_dose_prediction"].to_numpy(dtype=float)
    raise ValueError(f"Unknown phase offset {offset}")


def predict_phase(model: PhaseFit, frame: pd.DataFrame) -> np.ndarray:
    design, _ = phase_design(frame, model.spec, model.clock_center, model.clock_scale)
    correction = (design / model.feature_scale) @ model.coefficients
    return phase_offset(frame, model.spec.offset) + correction


def select_phase_fit(train: pd.DataFrame, spec: PhaseSpec) -> tuple[PhaseFit, pd.DataFrame]:
    """Select ridge with a nested leave-cell-out comparison."""
    cells = sorted(train["cell_id"].unique())
    if len(cells) < 3:
        raise ValueError("Need at least three train cells for nested selection")
    score_rows: list[dict[str, float | str]] = []
    for ridge in PHASE_RIDGES:
        cell_scores = []
        for held_cell in cells:
            inner_train = train.loc[train["cell_id"] != held_cell]
            inner_test = train.loc[train["cell_id"] == held_cell]
            center, clock_scale = clock_statistics(inner_train, spec)
            design, _ = phase_design(inner_train, spec, center, clock_scale)
            coefficients, feature_scale = ridge_coefficients(
                design,
                inner_train["phase_residual"].to_numpy(dtype=float) - phase_offset(inner_train, spec.offset),
                ridge,
            )
            model = PhaseFit(spec, ridge, coefficients, feature_scale, center, clock_scale)
            predicted = predict_phase(model, inner_test)
            rmse = float(np.sqrt(np.mean(np.square(predicted - inner_test["phase_residual"].to_numpy(dtype=float)))))
            cell_scores.append(rmse)
            score_rows.append({"ridge": ridge, "held_cell": held_cell, "rmse": rmse})
        score_rows.append({"ridge": ridge, "held_cell": "mean", "rmse": float(np.mean(cell_scores))})
    scores = pd.DataFrame(score_rows)
    means = scores.loc[scores["held_cell"].eq("mean")]
    selected_ridge = float(means.sort_values(["rmse", "ridge"]).iloc[0]["ridge"])
    center, clock_scale = clock_statistics(train, spec)
    design, _ = phase_design(train, spec, center, clock_scale)
    coefficients, feature_scale = ridge_coefficients(
        design,
        train["phase_residual"].to_numpy(dtype=float) - phase_offset(train, spec.offset),
        selected_ridge,
    )
    return PhaseFit(spec, selected_ridge, coefficients, feature_scale, center, clock_scale), scores


def calibration(target: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(predicted)), predicted])
    intercept, slope = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(intercept), float(slope)


def rank_correlation(target: np.ndarray, predicted: np.ndarray) -> float:
    if np.allclose(predicted, predicted[0]):
        return 0.0
    value = spearmanr(target, predicted).statistic
    return float(value) if np.isfinite(value) else 0.0


def metric_row(model_name: str, frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float | str]:
    target = frame["phase_residual"].to_numpy(dtype=float)
    intercept, slope = calibration(target, predicted)
    return {
        "model": model_name,
        "rows": len(frame),
        "rmse": float(np.sqrt(np.mean(np.square(predicted - target)))),
        "mae": float(np.mean(np.abs(predicted - target))),
        "bias": float(np.mean(predicted - target)),
        "spearman": rank_correlation(target, predicted),
        "observed_on_predicted_intercept": intercept,
        "observed_on_predicted_slope": slope,
        "sign_accuracy": float(np.mean(np.signbit(predicted) == np.signbit(target))),
    }


def grid_frame(cell: pd.DataFrame, spine: SpineFit, support_equations: np.ndarray) -> pd.DataFrame:
    axis = np.linspace(0.0, 1.0, GRID_SIZE)
    phase_0, phase_1 = np.meshgrid(axis, axis, indexing="ij")
    first = cell.iloc[0]
    grid = pd.DataFrame(
        {
            "cell_id": first["cell_id"],
            "phase_0_starcoder": phase_0.ravel(),
            "phase_1_starcoder": phase_1.ravel(),
            "materialized_tokens": first["materialized_tokens"],
            "total_parameters": first["total_parameters"],
            "non_embedding_parameters": first["non_embedding_parameters"],
            "total_steps": first["total_steps"],
            "boundary_step": first["boundary_step"],
            "log_materialized_tokens": first["log_materialized_tokens"],
            "log_total_parameters": first["log_total_parameters"],
            "log_total_tpp": first["log_total_tpp"],
            "log_nonembedding_tpp": first["log_nonembedding_tpp"],
        }
    )
    grid["aggregate"] = PHASE_0_FRACTION * grid["phase_0_starcoder"] + PHASE_1_FRACTION * grid["phase_1_starcoder"]
    grid["contrast"] = grid["phase_1_starcoder"] - grid["phase_0_starcoder"]
    inside = support_mask(
        grid["aggregate"].to_numpy(dtype=float),
        grid["contrast"].to_numpy(dtype=float),
        support_equations,
    )
    grid = grid.loc[inside].reset_index(drop=True)
    if grid.empty:
        raise ValueError("The optimization grid does not intersect observed untied support")
    grid["support_margin"] = support_margin(
        grid["aggregate"].to_numpy(dtype=float),
        grid["contrast"].to_numpy(dtype=float),
        support_equations,
    )
    grid["tied_optimum"] = spine.optimum
    grid["u"] = (grid["aggregate"] - spine.optimum) / AGGREGATE_SCALE
    grid["spine_prediction"] = spine.predict(grid["aggregate"].to_numpy(dtype=float))
    early_mass, late_mass = phase_learning_rate_masses(int(first["total_steps"]), int(first["boundary_step"]))
    grid["lr_phase_0_mass"] = early_mass
    grid["lr_phase_1_mass"] = late_mass
    grid["lr_effective_weight"] = early_mass * grid["phase_0_starcoder"] + late_mass * grid["phase_1_starcoder"]
    grid["lr_dose_prediction"] = spine.predict(grid["lr_effective_weight"].to_numpy(dtype=float)) - grid[
        "spine_prediction"
    ].to_numpy(dtype=float)
    return grid


def optimum_row(
    model_name: str,
    cell: pd.DataFrame,
    spine: SpineFit,
    predicted_phase: np.ndarray,
    reference: pd.Series,
    support_equations: np.ndarray,
) -> dict[str, float | str]:
    grid = grid_frame(cell, spine, support_equations)
    total_prediction = grid["spine_prediction"].to_numpy(dtype=float) + predicted_phase
    optimum_index = int(np.argmin(total_prediction))
    optimum = grid.iloc[optimum_index]
    tied_grid = np.linspace(0.0, 1.0, 4001)
    tied_bpb = float(np.min(spine.predict(tied_grid)))
    predicted_gain = tied_bpb - float(total_prediction[optimum_index])
    reference_p0 = float(reference["fitted_untied_p0"])
    reference_p1 = float(reference["fitted_untied_p1"])
    return {
        "model": model_name,
        "cell_id": str(cell.iloc[0]["cell_id"]),
        "predicted_p0": float(optimum["phase_0_starcoder"]),
        "predicted_p1": float(optimum["phase_1_starcoder"]),
        "predicted_aggregate": float(optimum["aggregate"]),
        "predicted_contrast": float(optimum["contrast"]),
        "predicted_bpb": float(total_prediction[optimum_index]),
        "predicted_gain": predicted_gain,
        "support_margin": float(optimum["support_margin"]),
        "optimum_on_support_boundary": bool(float(optimum["support_margin"]) <= SUPPORT_BOUNDARY_TOLERANCE),
        "reference_p0": reference_p0,
        "reference_p1": reference_p1,
        "reference_gain": float(reference["fitted_gain_tied_minus_untied_bpb"]),
        "reference_positive_gain_probability": float(reference["bootstrap_positive_gain_probability"]),
        "reference_optimum_qualified": bool(
            float(reference["bootstrap_positive_gain_probability"]) >= REFERENCE_GAIN_PROBABILITY_THRESHOLD
        ),
        "coordinate_error": float(
            np.hypot(
                float(optimum["phase_0_starcoder"]) - reference_p0,
                float(optimum["phase_1_starcoder"]) - reference_p1,
            )
        ),
        "gain_error": predicted_gain - float(reference["fitted_gain_tied_minus_untied_bpb"]),
    }


def selected_inner_rmse(model: PhaseFit, scores: pd.DataFrame) -> float:
    row = scores.loc[scores["held_cell"].eq("mean") & np.isclose(scores["ridge"], model.ridge)]
    if len(row) != 1:
        raise ValueError(f"Could not identify selected inner score for {model.spec.name}")
    return float(row["rmse"].iloc[0])


def fit_clock_selector(
    train: pd.DataFrame,
) -> tuple[PhaseFit, dict[str, tuple[PhaseFit, pd.DataFrame]], pd.DataFrame]:
    """Select one declared clock family using train cells only."""
    fitted: dict[str, tuple[PhaseFit, pd.DataFrame]] = {}
    selection_rows: list[dict[str, float | str]] = []
    for spec in CLOCK_PHASE_SPECS:
        model, scores = select_phase_fit(train, spec)
        fitted[spec.name] = (model, scores)
        selection_rows.append(
            {
                "candidate_spec": spec.name,
                "selected_ridge": model.ridge,
                "inner_cell_balanced_rmse": selected_inner_rmse(model, scores),
            }
        )
    selection = pd.DataFrame(selection_rows).sort_values(["inner_cell_balanced_rmse", "candidate_spec"], kind="stable")
    selected_name = str(selection.iloc[0]["candidate_spec"])
    return fitted[selected_name][0], fitted, selection


def prediction_block(frame: pd.DataFrame, model_name: str, predicted: np.ndarray) -> pd.DataFrame:
    block = frame[["cell_id", "phase_0_starcoder", "phase_1_starcoder", "phase_residual"]].copy()
    block["model"] = model_name
    block["predicted_phase_residual"] = predicted
    return block


def paired_cell_bootstrap(
    baseline_cells: pd.DataFrame,
    candidate_cells: pd.DataFrame,
) -> dict[str, float]:
    """Bootstrap the mean held-cell RMSE difference with cells as clusters."""
    baseline = baseline_cells.set_index("cell_id")["rmse"].sort_index()
    candidate = candidate_cells.set_index("cell_id")["rmse"].sort_index()
    if not baseline.index.equals(candidate.index):
        raise ValueError("Bootstrap models do not cover identical cells")
    differences = candidate.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences)))
    means = differences[indices].mean(axis=1)
    return {
        "mean_rmse_difference": float(differences.mean()),
        "bootstrap_ci_low": float(np.quantile(means, 0.025)),
        "bootstrap_ci_high": float(np.quantile(means, 0.975)),
    }


def exact_paired_sign_flip_test(
    baseline_cells: pd.DataFrame,
    candidate_cells: pd.DataFrame,
) -> dict[str, float]:
    """Enumerate the one-sided paired sign-flip null over held-cell RMSE differences."""
    baseline = baseline_cells.set_index("cell_id")["rmse"].sort_index()
    candidate = candidate_cells.set_index("cell_id")["rmse"].sort_index()
    if not baseline.index.equals(candidate.index):
        raise ValueError("Sign-flip models do not cover identical cells")
    differences = candidate.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    assignments = np.arange(1 << len(differences), dtype=np.uint64)[:, None]
    bits = (assignments >> np.arange(len(differences), dtype=np.uint64)) & 1
    signs = 2.0 * bits.astype(float) - 1.0
    null_means = (signs * np.abs(differences)[None, :]).mean(axis=1)
    observed = float(differences.mean())
    return {
        "sign_flip_assignments": float(len(null_means)),
        "sign_flip_one_sided_p": float(np.mean(null_means <= observed + 1e-15)),
    }


def within_cell_basis_oracle(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure whether the frozen low-dimensional basis can fit each cell at all."""
    rows: list[dict[str, float | str]] = []
    oracle_specs = tuple(spec for spec in PHASE_SPECS if spec.name in ("aggregate_taylor", CLOCK_BASELINE_NAME))
    for cell_id, cell in frame.groupby("cell_id", sort=True):
        target = cell["phase_residual"].to_numpy(dtype=float)
        rows.append(
            {
                "cell_id": str(cell_id),
                "model": "zero_phase",
                "selected_ridge": np.nan,
                "loocv_rmse": float(np.sqrt(np.mean(np.square(target)))),
            }
        )
        dose = cell["lr_dose_prediction"].to_numpy(dtype=float)
        rows.append(
            {
                "cell_id": str(cell_id),
                "model": "lr_dose_null",
                "selected_ridge": np.nan,
                "loocv_rmse": float(np.sqrt(np.mean(np.square(dose - target)))),
            }
        )
        for spec in oracle_specs:
            ridge_predictions: dict[float, np.ndarray] = {}
            for ridge in PHASE_RIDGES:
                predicted = np.empty(len(cell), dtype=float)
                for held in range(len(cell)):
                    keep = np.arange(len(cell)) != held
                    train = cell.iloc[keep]
                    test = cell.iloc[[held]]
                    center, clock_scale = clock_statistics(train, spec)
                    design, _ = phase_design(train, spec, center, clock_scale)
                    coefficients, feature_scale = ridge_coefficients(
                        design,
                        train["phase_residual"].to_numpy(dtype=float) - phase_offset(train, spec.offset),
                        ridge,
                    )
                    model = PhaseFit(spec, ridge, coefficients, feature_scale, center, clock_scale)
                    predicted[held] = predict_phase(model, test)[0]
                ridge_predictions[ridge] = predicted
            selected_ridge, selected_prediction = min(
                ridge_predictions.items(),
                key=lambda item: (float(np.sqrt(np.mean(np.square(item[1] - target)))), item[0]),
            )
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "model": spec.name,
                    "selected_ridge": selected_ridge,
                    "loocv_rmse": float(np.sqrt(np.mean(np.square(selected_prediction - target)))),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_splits(frame: pd.DataFrame) -> list[tuple[str, str, set[str]]]:
    """Return preregistered track and rung cell holdouts."""
    metadata = frame.drop_duplicates("cell_id")
    splits: list[tuple[str, str, set[str]]] = []
    tracks = sorted({track for values in metadata["track_memberships_parsed"] for track in values})
    for track in tracks:
        held = {str(row.cell_id) for row in metadata.itertuples() if row.track_memberships_parsed == [track]}
        if held:
            splits.append(("track", track, held))
    for rung in (1, 2, 3):
        held = set(metadata.loc[metadata["rung"].eq(rung), "cell_id"].astype(str))
        if held:
            splits.append(("rung", str(rung), held))
    return splits


def run_sensitivity_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the baseline and nested clock selector on structured cell holdouts."""
    prediction_rows: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    for split_kind, split_value, held_cells in sensitivity_splits(frame):
        train = frame.loc[~frame["cell_id"].isin(held_cells)]
        test = frame.loc[frame["cell_id"].isin(held_cells)]
        baseline_spec = next(spec for spec in PHASE_SPECS if spec.name == CLOCK_BASELINE_NAME)
        baseline, _ = select_phase_fit(train, baseline_spec)
        selected, _, selection = fit_clock_selector(train)
        for model_name, model in ((CLOCK_BASELINE_NAME, baseline), (CLOCK_SELECTOR_NAME, selected)):
            block = prediction_block(test, model_name, predict_phase(model, test))
            block["split_kind"] = split_kind
            block["split_value"] = split_value
            block["selected_spec"] = model.spec.name
            prediction_rows.append(block)
        selection = selection.copy()
        selection["split_kind"] = split_kind
        selection["split_value"] = split_value
        selection["selected"] = selection["candidate_spec"].eq(selected.spec.name)
        selection_rows.append(selection)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    cell_rows: list[dict[str, float | str]] = []
    for keys, block in predictions.groupby(["split_kind", "split_value", "model", "cell_id"], sort=True):
        row = metric_row(str(keys[2]), block, block["predicted_phase_residual"].to_numpy(dtype=float))
        row.update({"split_kind": str(keys[0]), "split_value": str(keys[1]), "cell_id": str(keys[3])})
        cell_rows.append(row)
    cell_metrics = pd.DataFrame(cell_rows)
    summary = cell_metrics.groupby(["split_kind", "split_value", "model"], as_index=False).agg(
        mean_cell_rmse=("rmse", "mean"), median_cell_rmse=("rmse", "median"), cells=("cell_id", "nunique")
    )
    return summary, cell_metrics, pd.concat(selection_rows, ignore_index=True)


def run_diagnostic(output_dir: Path) -> None:
    """Run the frozen nested cross-cell diagnostic."""
    protocol_path = output_dir / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError("Run --prepare-only before --run")
    frozen = json.loads(protocol_path.read_text())
    current = protocol_payload()
    for key in (
        "protocol_version",
        "source_sha256",
        "input_sha256",
        "reference_surface_sha256",
        "schedule_source_sha256",
    ):
        if frozen[key] != current[key]:
            raise RuntimeError(f"Protocol drift for {key}: frozen={frozen[key]} current={current[key]}")

    frame, spines = load_frame()
    untied = frame.loc[~np.isclose(frame["contrast"], 0.0, atol=1e-12)].copy()
    support_equations = optimization_support_equations(frame)
    references = pd.read_csv(REFERENCE_SURFACES_CSV).set_index("cell_id")
    prediction_rows: list[pd.DataFrame] = []
    inner_rows: list[pd.DataFrame] = []
    optimum_rows: list[dict[str, float | str]] = []
    coefficient_rows: list[dict[str, float | str]] = []
    selector_rows: list[pd.DataFrame] = []

    for held_cell in sorted(untied["cell_id"].unique()):
        train = untied.loc[untied["cell_id"] != held_cell]
        test = untied.loc[untied["cell_id"] == held_cell]
        zero = np.zeros(len(test), dtype=float)
        prediction_rows.append(prediction_block(test, "zero_phase", zero))
        zero_grid = grid_frame(test, spines[held_cell], support_equations)
        optimum_rows.append(
            optimum_row(
                "zero_phase",
                test,
                spines[held_cell],
                np.zeros(len(zero_grid), dtype=float),
                references.loc[held_cell],
                support_equations,
            )
        )
        dose = test["lr_dose_prediction"].to_numpy(dtype=float)
        prediction_rows.append(prediction_block(test, "lr_dose_null", dose))
        optimum_rows.append(
            optimum_row(
                "lr_dose_null",
                test,
                spines[held_cell],
                zero_grid["lr_dose_prediction"].to_numpy(dtype=float),
                references.loc[held_cell],
                support_equations,
            )
        )

        fitted: dict[str, tuple[PhaseFit, pd.DataFrame]] = {}
        for spec in PHASE_SPECS:
            model, inner = select_phase_fit(train, spec)
            fitted[spec.name] = (model, inner)
            predicted = predict_phase(model, test)
            prediction_rows.append(prediction_block(test, spec.name, predicted))
            inner = inner.copy()
            inner["outer_held_cell"] = held_cell
            inner["model"] = spec.name
            inner_rows.append(inner)

            design, names = phase_design(train, spec, model.clock_center, model.clock_scale)
            unscaled_coefficients = model.coefficients / model.feature_scale
            assert design.shape[1] == len(names) == len(unscaled_coefficients)
            for name, coefficient in zip(names, unscaled_coefficients, strict=True):
                coefficient_rows.append(
                    {
                        "outer_held_cell": held_cell,
                        "model": spec.name,
                        "ridge": model.ridge,
                        "feature": name,
                        "coefficient": float(coefficient),
                    }
                )

            grid = grid_frame(test, spines[held_cell], support_equations)
            optimum_rows.append(
                optimum_row(
                    spec.name,
                    test,
                    spines[held_cell],
                    predict_phase(model, grid),
                    references.loc[held_cell],
                    support_equations,
                )
            )

        selection = pd.DataFrame(
            [
                {
                    "candidate_spec": spec.name,
                    "selected_ridge": fitted[spec.name][0].ridge,
                    "inner_cell_balanced_rmse": selected_inner_rmse(*fitted[spec.name]),
                }
                for spec in CLOCK_PHASE_SPECS
            ]
        ).sort_values(["inner_cell_balanced_rmse", "candidate_spec"], kind="stable")
        selected_name = str(selection.iloc[0]["candidate_spec"])
        selected_model = fitted[selected_name][0]
        prediction_rows.append(prediction_block(test, CLOCK_SELECTOR_NAME, predict_phase(selected_model, test)))
        selection["outer_held_cell"] = held_cell
        selection["selected"] = selection["candidate_spec"].eq(selected_name)
        selector_rows.append(selection)
        grid = grid_frame(test, spines[held_cell], support_equations)
        optimum_rows.append(
            optimum_row(
                CLOCK_SELECTOR_NAME,
                test,
                spines[held_cell],
                predict_phase(selected_model, grid),
                references.loc[held_cell],
                support_equations,
            )
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(output_dir / "phase_predictions.csv", index=False)
    pd.concat(inner_rows, ignore_index=True).to_csv(output_dir / "inner_ridge_scores.csv", index=False)
    pd.concat(selector_rows, ignore_index=True).to_csv(output_dir / "nested_clock_selection.csv", index=False)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(output_dir / "fold_coefficients.csv", index=False)
    optima = pd.DataFrame(optimum_rows)
    optima.to_csv(output_dir / "optimum_diagnostics.csv", index=False)

    cell_metric_rows: list[dict[str, float | str]] = []
    for model_name, block in predictions.groupby("model", sort=True):
        for cell_id, cell in block.groupby("cell_id", sort=True):
            row = metric_row(model_name, cell, cell["predicted_phase_residual"].to_numpy(dtype=float))
            row["cell_id"] = str(cell_id)
            cell_metric_rows.append(row)
    cell_metrics = pd.DataFrame(cell_metric_rows)
    cell_metrics.to_csv(output_dir / "cell_metrics.csv", index=False)

    metric_rows: list[dict[str, float | str]] = []
    for model_name, block in predictions.groupby("model", sort=True):
        row = metric_row(model_name, block, block["predicted_phase_residual"].to_numpy(dtype=float))
        per_cell = cell_metrics.loc[cell_metrics["model"].eq(model_name)]
        row["mean_cell_rmse"] = float(per_cell["rmse"].mean())
        row["median_cell_rmse"] = float(per_cell["rmse"].median())
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    zero_rmse = float(metrics.loc[metrics["model"].eq("zero_phase"), "rmse"].iloc[0])
    metrics["rmse_improvement_over_zero"] = 1.0 - metrics["rmse"] / zero_rmse
    zero_mean_cell_rmse = float(metrics.loc[metrics["model"].eq("zero_phase"), "mean_cell_rmse"].iloc[0])
    metrics["mean_cell_rmse_improvement_over_zero"] = 1.0 - metrics["mean_cell_rmse"] / zero_mean_cell_rmse
    optimum_summary_all = (
        optima.assign(abs_gain_error=lambda value: value["gain_error"].abs())
        .groupby("model", as_index=False)
        .agg(
            mean_coordinate_error_all=("coordinate_error", "mean"),
            median_coordinate_error_all=("coordinate_error", "median"),
            mean_abs_gain_error_all=("abs_gain_error", "mean"),
            support_boundary_optima=("optimum_on_support_boundary", "sum"),
        )
    )
    qualified_optima = optima.loc[optima["reference_optimum_qualified"]].copy()
    if qualified_optima["cell_id"].nunique() != 8:
        raise ValueError("Expected exactly eight reference-qualified optimum cells")
    optimum_summary_qualified = (
        qualified_optima.assign(abs_gain_error=lambda value: value["gain_error"].abs())
        .groupby("model", as_index=False)
        .agg(
            mean_coordinate_error_qualified=("coordinate_error", "mean"),
            median_coordinate_error_qualified=("coordinate_error", "median"),
            mean_abs_gain_error_qualified=("abs_gain_error", "mean"),
            qualified_optimum_cells=("cell_id", "nunique"),
        )
    )
    metrics = metrics.merge(optimum_summary_all, on="model", how="left")
    metrics = metrics.merge(optimum_summary_qualified, on="model", how="left")
    metrics.to_csv(output_dir / "phase_model_metrics.csv", index=False)

    baseline = metrics.set_index("model").loc[CLOCK_BASELINE_NAME]
    candidate = metrics.set_index("model").loc[CLOCK_SELECTOR_NAME]
    baseline_cells = cell_metrics.loc[cell_metrics["model"].eq(CLOCK_BASELINE_NAME)]
    candidate_cells = cell_metrics.loc[cell_metrics["model"].eq(CLOCK_SELECTOR_NAME)]
    aligned_baseline = baseline_cells.set_index("cell_id").sort_index()
    aligned_candidate = candidate_cells.set_index("cell_id").sort_index()
    wins = int((aligned_candidate["rmse"] < aligned_baseline["rmse"]).sum())
    bootstrap = paired_cell_bootstrap(baseline_cells, candidate_cells)
    sign_flip = exact_paired_sign_flip_test(baseline_cells, candidate_cells)
    selected_clock_counts = (
        pd.concat(selector_rows, ignore_index=True)
        .loc[lambda value: value["selected"]]
        .groupby("candidate_spec", as_index=False)
        .agg(selected_outer_folds=("outer_held_cell", "nunique"))
        .sort_values(["selected_outer_folds", "candidate_spec"], ascending=[False, True], kind="stable")
    )
    selected_clock_counts.to_csv(output_dir / "clock_selection_stability.csv", index=False)
    dominant_clock = str(selected_clock_counts.iloc[0]["candidate_spec"])
    dominant_clock_folds = int(selected_clock_counts.iloc[0]["selected_outer_folds"])
    checks = {
        "mean_cell_rmse_5pct": bool(candidate["mean_cell_rmse"] <= 0.95 * baseline["mean_cell_rmse"]),
        "cell_wins_7of10": wins >= 7,
        "sign_flip_p_at_most_0p05": sign_flip["sign_flip_one_sided_p"] <= SIGN_FLIP_ALPHA,
        "clock_family_stable_8of10": dominant_clock_folds >= CLOCK_SELECTION_STABILITY_FOLDS,
        "coordinate_error_noninferior": bool(
            candidate["mean_coordinate_error_qualified"] <= baseline["mean_coordinate_error_qualified"]
        ),
        "gain_error_noninferior": bool(
            candidate["mean_abs_gain_error_qualified"] <= baseline["mean_abs_gain_error_qualified"]
        ),
        "support_boundary_noninferior": bool(
            candidate["support_boundary_optima"] <= baseline["support_boundary_optima"]
        ),
    }
    decision_frame = pd.DataFrame(
        [
            {
                "model": CLOCK_SELECTOR_NAME,
                "baseline": CLOCK_BASELINE_NAME,
                "cell_rmse_wins": wins,
                "dominant_clock": dominant_clock,
                "dominant_clock_outer_folds": dominant_clock_folds,
                **bootstrap,
                **sign_flip,
                **checks,
                "licenses_scale_covariate": all(checks.values()),
            }
        ]
    )
    decision_frame.to_csv(output_dir / "clock_gate.csv", index=False)

    oracle = within_cell_basis_oracle(untied)
    oracle.to_csv(output_dir / "within_cell_basis_oracle.csv", index=False)
    sensitivity_summary, sensitivity_cells, sensitivity_selection = run_sensitivity_splits(untied)
    sensitivity_summary.to_csv(output_dir / "structured_holdout_summary.csv", index=False)
    sensitivity_cells.to_csv(output_dir / "structured_holdout_cell_metrics.csv", index=False)
    sensitivity_selection.to_csv(output_dir / "structured_holdout_clock_selection.csv", index=False)

    spine_rows = [
        {
            "cell_id": cell_id,
            "ridge": spine.ridge,
            "loocv_rmse": spine.loocv_rmse,
            "tied_optimum": spine.optimum,
        }
        for cell_id, spine in sorted(spines.items())
    ]
    pd.DataFrame(spine_rows).to_csv(output_dir / "spine_diagnostics.csv", index=False)

    lr_mass_rows = (
        frame[
            [
                "cell_id",
                "total_steps",
                "boundary_step",
                "lr_phase_0_mass",
                "lr_phase_1_mass",
                "total_tpp",
                "nonembedding_tpp",
            ]
        ]
        .drop_duplicates("cell_id")
        .sort_values("cell_id")
    )
    lr_mass_rows.to_csv(output_dir / "optimizer_clock_diagnostics.csv", index=False)

    best = metrics.sort_values("mean_cell_rmse").iloc[0]
    passed = decision_frame.loc[decision_frame["licenses_scale_covariate"], "model"].tolist()
    report_lines = [
        "# WSD80 Cross-Cell Phase-Control Diagnostic",
        "",
        f"Protocol `{frozen['source_sha256']}`. Exposed development evidence; not confirmation.",
        "",
        "## Result",
        "",
        f"Descriptive lowest mean held-cell RMSE among all reported models: `{best['model']}` at "
        f"`{best['mean_cell_rmse']:.6f}` BPB.",
        f"Scale covariates clearing every frozen diagnostic gate: `{passed or 'none'}`.",
        "",
        metrics.sort_values("rmse").to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "A passing clock licenses only that declared scale coordinate for a later mechanistic candidate.",
        "Failure means the ten-cell panel does not support a shared linear clock modulation of this",
        "aggregate-conditioned odd/even basis. It does not prove phase dynamics are scale-invariant.",
        "The LR-dose null, within-cell oracle, and track/rung holdouts diagnose whether any failure comes",
        "from the clock, the correction basis, or transfer across the deliberately confounded scaling tracks.",
        "Stage-2 and Stage-3 coordinates were adaptively chosen within each cell. Candidate and baseline",
        "share every scored row, but absolute cell RMSEs average over different development supports.",
        "The cell bootstrap is descriptive because leave-cell-out fits overlap. The gate instead reports",
        "the exact paired sign-flip statistic, conditional on exchangeable cellwise signs.",
        "Reference optimum gates use only cells whose fitted positive-gain probability is at least 0.95;",
        "full-panel optimum errors remain descriptive.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    print(metrics.sort_values("mean_cell_rmse").to_string(index=False))
    print("\nClock gate:\n" + decision_frame.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        path = write_protocol(args.output_dir)
        print(path)
        return
    run_diagnostic(args.output_dir)


if __name__ == "__main__":
    main()
