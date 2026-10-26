# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Inspect zero-kinetic StarCoder geometry without outcomes, heads, or fitting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, roots_legendre
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
NATIVE = REPO.parent / "mixture-selection/mixture_selection.py"
OUTPUT = ROOT / "reference_outputs/tpp40_acquisition_retention_20261015/geometry"
DATA = ROOT / "reference_outputs/two_phase_benchmark_20260921/v3/prepared/starcoder_wsd80_dense"
PANEL = "starcoder_wsd80_dense"
NATIVE_SHA256 = "61b7c017accb920d6874db848bc65fc0fdb729301a00a2fcf69a8595f1474396"
INPUT_HASHES = {
    "policies.npz": "295ede0439073c7f07ccf1d53fde85eb283ec2c3892deaa680325096f96d7721",
    "metadata.json": "dae9685ffd6eac5c24f4a4f83d43b4463fcf71ef22fd41560817c91cd62c4bc0",
    "rows.csv": "872d3139fc69a4839cbf2edf7cd31e326f59a048c2a4e880403a356b6fd225fd",
    "folds.csv": "2ec5e29d05e02d23836af165128bac304eacbb674fb66ddf58ff9a09872d4dbd",
}
POLICY_KEYS = ("phase0", "phase1", "phase0_epochs", "phase1_epochs", "c0", "c1")
METADATA_KEYS = ("name", "alpha", "buckets")
ROW_COLUMNS = ("row_id", "policy_id", "group_id", "source", "region", "tied")
FOLD_COLUMNS = ("panel", "repeat", "outer_fold", "row_id", "role", "inner_fold")
QUADRATURE_ORDERS = (16, 32, 64, 128, 256, 512, 1024)
QUADRATURE_ATOL = 2e-11
QUADRATURE_RTOL = 2e-9
CLOCKS = ("equal_group", "global_pool")
NUISANCE_SCOPES = ("fixed_shape_known_floor", "columnwise_shape_known_floor")
APPROVAL_SCOPE = "tp034_starcoder_zero_kinetic_geometry"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_native() -> Any:
    """Import only the pinned standalone source; do not load its data."""
    if sha256(NATIVE) != NATIVE_SHA256:
        raise ValueError("Native MARINER source differs from the reviewed source")
    spec = importlib.util.spec_from_file_location("tp034_geometry_native", NATIVE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mariner = load_native()


@dataclass(frozen=True)
class Geometry:
    early: np.ndarray
    late: np.ndarray
    alpha: float
    total_scales: np.ndarray
    buckets: tuple[str, ...]
    rows: pd.DataFrame
    subsets: dict[str, np.ndarray]


@dataclass(frozen=True)
class Integral:
    value: np.ndarray
    refinement_error: np.ndarray
    converged: np.ndarray
    order: np.ndarray


@cache
def quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_legendre(order)
    return (nodes + 1) / 2, weights / 2


def hazard_increment(start: np.ndarray, increment: np.ndarray, rate: float, power: float) -> np.ndarray:
    """Avoid subtracting nearly equal cumulative hazards."""
    start, increment = np.broadcast_arrays(start, increment)
    result = (rate * increment) ** power
    positive = start > 0
    result[positive] = (rate * start[positive]) ** power * np.expm1(
        power * np.log1p(increment[positive] / start[positive])
    )
    return result


def phase_pair_integral(
    start: np.ndarray,
    increment: np.ndarray,
    final: np.ndarray,
    receiver: int,
    source: int,
    rate: float,
    power: float,
    orders: tuple[int, ...] = QUADRATURE_ORDERS,
    atol: float = QUADRATURE_ATOL,
    rtol: float = QUADRATURE_RTOL,
) -> Integral:
    """Integrate a directed pair over one exact linear-exposure phase.

    Integrate over the competitor's benefit increment, then transform its
    fraction as u=t**(1/p). All quadrature weights and integrands are positive.
    This differs from quadrature over the physical phase parameter.
    """
    n = len(start)
    sj, dj = start[:, source], increment[:, source]
    hi = (rate * start[:, receiver]) ** power
    hj = (rate * sj) ** power
    delta_hj = hazard_increment(sj, dj, rate, power)
    source_fraction = -np.expm1(-delta_hj)
    source_mass = np.exp(-hj) * source_fraction
    value = np.zeros(n)
    error = np.zeros(n)
    converged = np.ones(n, dtype=bool)
    used_order = np.zeros(n, dtype=int)
    fixed = (increment[:, receiver] == 0) & (source_mass > 0)
    remaining = hazard_increment(start[:, receiver], final[:, receiver] - start[:, receiver], rate, power)
    value[fixed] = np.exp(-remaining[fixed]) * (-np.expm1(-hi[fixed])) * source_mass[fixed]
    active = np.flatnonzero((increment[:, receiver] > 0) & (source_mass > 0))
    converged[active] = False
    previous = np.full(n, np.nan)
    for order in orders:
        if not len(active):
            break
        t, weights = quadrature_rule(order)
        u = t ** (1 / power)
        density = t ** (1 / power - 1) / power
        dh = -np.log1p(-source_fraction[active, None] * u)
        own_start = sj[active, None]
        source_exposure = (dh / rate**power) ** (1 / power)
        positive = sj[active] > 0
        source_exposure[positive] = own_start[positive] * np.expm1(
            np.log1p(dh[positive] / hj[active[positive], None]) / power
        )
        fraction = source_exposure / dj[active, None]
        if np.any(fraction < -1e-12) or np.any(fraction > 1 + 1e-12):
            raise ArithmeticError("Competitor-benefit inversion left the physical phase")
        fraction = np.clip(fraction, 0, 1)
        ei = start[active, receiver, None] + increment[active, receiver, None] * fraction
        rest = np.maximum(final[active, receiver, None] - ei, 0)
        remaining_h = hazard_increment(ei, rest, rate, power)
        # Integrate the constant starting kernel exactly. The remaining kernel
        # increment is positive and avoids a fractional-power constant at zero.
        delta_hi = hazard_increment(
            start[active, receiver, None], increment[active, receiver, None] * fraction, rate, power
        )
        kernel_increment = np.exp(-remaining_h) * (-np.expm1(-delta_hi))
        kernel_start = np.exp(-remaining[active]) * (-np.expm1(-hi[active]))
        current = source_mass[active] * (kernel_start + kernel_increment @ (weights * density))
        difference = np.abs(current - previous[active])
        done = np.isfinite(previous[active]) & (difference <= atol + rtol * np.abs(current))
        value[active] = current
        error[active] = difference
        used_order[active] = order
        previous[active] = current
        converged[active[done]] = True
        active = active[~done]
    if np.any(~np.isfinite(value)) or np.any(~np.isfinite(error)):
        raise ArithmeticError("Pair quadrature has no finite refinement comparison")
    return Integral(value, error, converged, used_order)


def pair_integrals(early: np.ndarray, late: np.ndarray, rate: float, power: float) -> Integral:
    """Return both directed, unit-source-mass integrals for two buckets."""
    early, late = np.asarray(early, float), np.asarray(late, float)
    if early.shape != late.shape or early.ndim != 2 or early.shape[1] != 2:
        raise ValueError("Expected equally shaped row-by-two phase epochs")
    if not np.all(np.isfinite(early)) or not np.all(np.isfinite(late)) or np.any(early < 0) or np.any(late < 0):
        raise ValueError("Physical phase epochs must be finite and nonnegative")
    total = early + late
    values, errors, converged, orders = [], [], [], []
    for receiver, source in ((0, 1), (1, 0)):
        first = phase_pair_integral(np.zeros_like(early), early, total, receiver, source, rate, power)
        second = phase_pair_integral(early, late, total, receiver, source, rate, power)
        values.append(first.value + second.value)
        errors.append(first.refinement_error + second.refinement_error)
        converged.append(first.converged & second.converged)
        orders.append(np.maximum(first.order, second.order))
    return Integral(
        np.column_stack(values), np.column_stack(errors), np.column_stack(converged), np.column_stack(orders)
    )


def zero_kinetic_geometry(early: np.ndarray, late: np.ndarray, alpha: float, rate: float, power: float) -> dict:
    """Compute stable path/tied pair influences without task coefficients."""
    if not 0 < alpha < 1:
        raise ValueError("Both physical phases must have positive duration")
    total = early + late
    path = pair_integrals(early, late, rate, power)
    tied = pair_integrals(alpha * total, (1 - alpha) * total, rate, power)
    fractions = np.divide(early, total, out=np.full_like(total, alpha), where=total > 0)
    late_fractions = np.divide(late, total, out=np.full_like(total, 1 - alpha), where=total > 0)
    tolerance = 64 * np.finfo(float).eps
    is_tied = np.all(np.abs(fractions - alpha) <= tolerance * alpha, axis=1) & np.all(
        np.abs(late_fractions - (1 - alpha)) <= tolerance * (1 - alpha), axis=1
    )
    raw = tied.value - path.value
    derivative = raw.copy()
    derivative[is_tied] = 0
    benefit = mariner.benefit(total, rate, power)
    bound = np.repeat(np.prod(benefit, axis=1)[:, None], 2, axis=1)
    return {
        "path": path.value,
        "tied": tied.value,
        "raw_derivative": raw,
        "derivative": derivative,
        "pair_bound": bound,
        "integration_error": path.refinement_error + tied.refinement_error,
        "converged": path.converged & tied.converged,
        "order": np.maximum(path.order, tied.order),
        "is_tied": is_tied,
        "benefit": benefit,
    }


def nuisance_design(exposure: np.ndarray, shape: dict[str, float], scope: str) -> np.ndarray:
    """Return a conditional static tangent span, never an unknown-floor claim."""
    fixed = np.column_stack([np.ones(len(exposure)), mariner.design_matrix(exposure, shape)])
    if scope == "fixed_shape_known_floor":
        return fixed
    if scope != "columnwise_shape_known_floor":
        raise ValueError(f"Unknown nuisance scope: {scope}")
    rate, power, threshold = shape["rate"], shape["power"], shape["threshold"]
    dose = rate * exposure
    hazard = dose**power
    common = hazard * np.exp(-hazard)
    logdose = np.zeros_like(dose)
    np.log(dose, out=logdose, where=dose > 0)
    z = np.log1p(exposure) - threshold
    harm_derivative = -2 * np.logaddexp(z, 0) * expit(z)
    return np.column_stack([fixed, -power * common, -common * logdose, harm_derivative])


def residual_projection(
    design: np.ndarray, directions: np.ndarray, integration_error: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Report unweighted conditional geometry using a scale-normalized SVD."""
    norms = np.linalg.norm(design, axis=0)
    normalized = np.divide(design, norms, out=np.zeros_like(design), where=norms > 0)
    u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    tolerance = float(np.finfo(float).eps * max(normalized.shape) * singular[0])
    rank = int(np.sum(singular > tolerance))
    q = u[:, :rank]
    residual = directions - q @ (q.T @ directions)
    input_norm = np.linalg.norm(directions, axis=0)
    residual_norm = np.linalg.norm(residual, axis=0)
    ratio = [float(r / n) if n > 0 else None for r, n in zip(residual_norm, input_norm, strict=True)]
    return {
        "rows": len(design),
        "nuisance_columns": design.shape[1],
        "nuisance_column_norms": norms.tolist(),
        "nuisance_raw_singular_values": np.linalg.svd(design, compute_uv=False).tolist(),
        "nuisance_normalized_singular_values": singular.tolist(),
        "nuisance_numerical_rank": rank,
        "numerical_rank_tolerance": tolerance,
        "direction_column_norms": input_norm.tolist(),
        "residual_column_norms": residual_norm.tolist(),
        "retained_norm_ratios": ratio,
        "direction_singular_values": np.linalg.svd(directions, compute_uv=False).tolist(),
        "residual_singular_values": np.linalg.svd(residual, compute_uv=False).tolist(),
        "integration_refinement_error_frobenius": float(np.linalg.norm(integration_error)),
        "error_interpretation": "refinement estimate, not a rigorous quadrature or projector-error bound",
        "scope": "unknown benefit amplitudes; no unknown-floor tangent; no statistical threshold or qualification",
    }, residual


def training_subsets(rows: pd.DataFrame, folds: pd.DataFrame) -> dict[str, np.ndarray]:
    """Preserve the original repeat-zero asymmetric training subsets."""
    if rows.row_id.duplicated().any():
        raise ValueError("Duplicate geometry row identifiers")
    table = folds[(folds.panel == PANEL) & (folds.repeat == 0)]
    if set(table.outer_fold) != set(range(5)):
        raise ValueError("Expected all five original outer folds")
    lookup = {row: index for index, row in enumerate(rows.row_id)}
    asymmetric = set(rows.loc[~rows.tied, "row_id"])
    result = {"full_asymmetric": np.flatnonzero(~rows.tied.to_numpy())}
    test_ids = []
    for fold in range(5):
        part = table[table.outer_fold == fold]
        if part.row_id.duplicated().any() or not set(part.role) <= {"train", "test", "tied_test"}:
            raise ValueError("Malformed original fold rows")
        train = set(part.loc[part.role == "train", "row_id"])
        test = set(part.loc[part.role == "test", "row_id"])
        if train & test or train | test != asymmetric:
            raise ValueError("Outer folds must partition all asymmetric policies")
        tied_ids = set(part.loc[part.role == "tied_test", "row_id"])
        if not tied_ids <= set(rows.loc[rows.tied, "row_id"]):
            raise ValueError("Tied diagnostic rows cross asymmetric roles")
        train_groups = set(rows.loc[rows.row_id.isin(train), "group_id"])
        test_groups = set(rows.loc[rows.row_id.isin(test), "group_id"])
        if train_groups & test_groups:
            raise ValueError("Original aggregate groups cross an outer partition")
        result[f"outer_{fold}_train"] = np.array(sorted(lookup[row] for row in train), dtype=int)
        test_ids.extend(test)
    if sorted(test_ids) != sorted(asymmetric):
        raise ValueError("Each asymmetric policy must be tested once")
    return result


def read_geometry(folder: Path) -> Geometry:
    """Read only six policy arrays and explicitly whitelisted metadata columns."""
    metadata_file = json.loads((folder / "metadata.json").read_text())
    metadata = {key: metadata_file[key] for key in METADATA_KEYS}
    if metadata["name"] != PANEL:
        raise ValueError("This runner is restricted to the frozen StarCoder panel")
    if metadata["buckets"] != ["nemotron_full", "starcoder"]:
        raise ValueError("Expected the original ordered web/code buckets")
    with np.load(folder / "policies.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(POLICY_KEYS):
            raise ValueError("Policy archive contains unexpected arrays; outcomes are forbidden")
        arrays = {key: archive[key].copy() for key in POLICY_KEYS}
    rows = pd.read_csv(folder / "rows.csv", usecols=list(ROW_COLUMNS))
    folds = pd.read_csv(folder / "folds.csv", usecols=list(FOLD_COLUMNS))
    if rows.tied.dtype != bool:
        raise ValueError("Tied metadata must be boolean")
    alpha = float(metadata["alpha"])
    c0, c1 = arrays["c0"], arrays["c1"]
    total_scales = c0 + c1
    if c0.shape != (2,) or np.any(c0 <= 0) or np.any(c1 <= 0):
        raise ValueError("Two positive physical exposure scales are required")
    if not np.allclose(c0 / total_scales, alpha, rtol=1e-13, atol=1e-15):
        raise ValueError("Physical exposure fractions disagree with metadata")
    for phase in (0, 1):
        weights = arrays[f"phase{phase}"]
        exposures = arrays[f"phase{phase}_epochs"]
        scale = arrays[f"c{phase}"]
        if weights.shape != (len(rows), 2) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Invalid physical phase weights")
        if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError("Phase weights violate their token simplex")
        if not np.allclose(exposures, weights * scale, rtol=1e-13, atol=1e-14):
            raise ValueError("Stored physical epochs disagree with weights and scales")
    tied = np.all(np.abs(arrays["phase0"] - arrays["phase1"]) <= 1e-12, axis=1)
    if not np.array_equal(tied, rows.tied.to_numpy()):
        raise ValueError("Tied row metadata disagrees with physical policies")
    return Geometry(
        arrays["phase0_epochs"],
        arrays["phase1_epochs"],
        alpha,
        total_scales,
        tuple(metadata["buckets"]),
        rows,
        training_subsets(rows, folds),
    )


def clock_weights(total_scales: np.ndarray) -> dict[str, np.ndarray]:
    """Use fixed singleton groups; common exposure-reference tokens cancel."""
    pools = 1 / total_scales
    return {"equal_group": np.full(2, 0.5), "global_pool": pools / pools.sum()}


def environment() -> dict[str, str]:
    return {
        "python": sys.version,
        **{key: importlib.metadata.version(key) for key in ("numpy", "pandas", "scipy", "threadpoolctl")},
    }


def protocol() -> dict:
    return {
        "candidate_id": "TP40-034",
        "scope": APPROVAL_SCOPE,
        "panel": PANEL,
        "groups": [["nemotron_full"], ["starcoder"]],
        "clocks": list(CLOCKS),
        "rates": list(mariner.SHAPE_RATES),
        "powers": list(mariner.SHAPE_POWERS),
        "thresholds": list(mariner.SHAPE_THRESHOLDS),
        "subsets": ["full_asymmetric", *[f"outer_{i}_train" for i in range(5)]],
        "nuisance_scopes": list(NUISANCE_SCOPES),
        "policy_keys": list(POLICY_KEYS),
        "metadata_keys": list(METADATA_KEYS),
        "row_columns": list(ROW_COLUMNS),
        "fold_columns": list(FOLD_COLUMNS),
        "expected_rows": 346,
        "expected_asymmetric_rows": 326,
        "expected_tied_rows": 20,
        "expected_alpha": 0.7970634504457262,
        "quadrature": {
            "orders": list(QUADRATURE_ORDERS),
            "atol": QUADRATURE_ATOL,
            "rtol": QUADRATURE_RTOL,
            "coordinate": "competing-benefit fraction u=t**(1/p)",
            "error": "successive-order estimate, not a rigorous bound",
        },
        "projection": "unweighted, column-L2-normalized SVD; rank tol eps*max(shape)*largest singular value",
        "forbidden": [
            "outcomes",
            "task coefficients",
            "floor tangent",
            "fitting",
            "optimization",
            "kinetic ladder",
            "TPP40 expansion",
        ],
        "scientific_pass_threshold": None,
    }


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Preserve existing evidence: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_once(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def write_arrays(path: Path, values: dict[str, np.ndarray]) -> None:
    output = io.BytesIO()
    np.savez_compressed(output, **values)
    write_once(path, output.getvalue())


def implementation_hashes() -> dict[str, str]:
    paths = (Path(__file__).resolve(), ROOT / "test_tpp40_acquisition_retention_geometry_20261015.py", NATIVE)
    return {str(path): sha256(path) for path in paths}


def frozen_identity(output: Path) -> dict:
    registration = json.loads((output / "registration.json").read_text())
    if registration["protocol"] != protocol() or registration["implementation_hashes"] != implementation_hashes():
        raise ValueError("Registered procedure or source identity changed")
    inputs = {str(DATA / name): sha256(DATA / name) for name in INPUT_HASHES}
    if (
        inputs != {str(DATA / name): value for name, value in INPUT_HASHES.items()}
        or registration["input_hashes"] != inputs
    ):
        raise ValueError("Frozen policy metadata or covariates changed")
    if registration["environment"] != environment():
        raise ValueError("Registered numerical runtime changed")
    return {
        "registration_sha256": sha256(output / "registration.json"),
        "implementation_hashes": implementation_hashes(),
        "input_hashes": inputs,
        "environment": environment(),
    }


def require_approval(output: Path) -> dict:
    identity = frozen_identity(output)
    if json.loads((output / "execution_identity.json").read_text()) != identity:
        raise ValueError("Execution identity changed")
    approval = json.loads((output / "approval.json").read_text())
    expected = {
        "scope": APPROVAL_SCOPE,
        "registration_sha256": identity["registration_sha256"],
        "execution_identity_sha256": sha256(output / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Exact frozen geometry approval is required")
    return identity


def completed(folder: Path, identity: dict, rate: float, power: float) -> bool:
    receipt = folder / "complete.json"
    if not receipt.exists():
        return False
    saved = json.loads(receipt.read_text())
    if saved["identity"] != identity:
        raise ValueError("Completed geometry cell belongs to another identity")
    if saved["shape"] != {"rate": rate, "power": power}:
        raise ValueError("Completed geometry cell belongs to another shape")
    if any(sha256(folder / name) != digest for name, digest in saved["outputs"].items()):
        raise ValueError("Completed geometry evidence changed")
    return True


def analyze_shape(geometry: Geometry, rate: float, power: float, folder: Path, identity: dict) -> None:
    """Persist one benefit-shape cell; never select a mechanism or shape."""
    if completed(folder, identity, rate, power):
        return
    quantities = zero_kinetic_geometry(geometry.early, geometry.late, geometry.alpha, rate, power)
    if not np.array_equal(quantities["is_tied"], geometry.rows.tied.to_numpy()):
        raise ValueError("Tied identity mask differs from frozen metadata")
    write_arrays(folder / "pair_integrals.npz", quantities)
    weights = clock_weights(geometry.total_scales)
    rows, projections, residuals = [], [], {}
    exposure = geometry.early + geometry.late
    for clock, source_mass in weights.items():
        multiplier = source_mass[::-1]
        direction = quantities["derivative"] * multiplier
        error = quantities["integration_error"] * multiplier
        for index, row_id in enumerate(geometry.rows.row_id):
            for receiver in range(2):
                bound = quantities["pair_bound"][index, receiver] * multiplier[receiver]
                rows.append(
                    {
                        "row_id": row_id,
                        "clock": clock,
                        "receiver": geometry.buckets[receiver],
                        "source": geometry.buckets[1 - receiver],
                        "tied": bool(quantities["is_tied"][index]),
                        "policy_source": str(geometry.rows.iloc[index]["source"]),
                        "region": str(geometry.rows.iloc[index]["region"]),
                        "zero_early_code": bool(geometry.early[index, 1] == 0),
                        "zero_late_code": bool(geometry.late[index, 1] == 0),
                        "early_code_weight": float(
                            geometry.early[index, 1] / (geometry.alpha * geometry.total_scales[1])
                        ),
                        "late_code_weight": float(
                            geometry.late[index, 1] / ((1 - geometry.alpha) * geometry.total_scales[1])
                        ),
                        "early_epochs": float(geometry.early[index, receiver]),
                        "late_epochs": float(geometry.late[index, receiver]),
                        "terminal_clock": float(source_mass[1 - receiver] * quantities["benefit"][index, 1 - receiver]),
                        "source_mass": float(source_mass[1 - receiver]),
                        "derivative": float(direction[index, receiver]),
                        "analytic_bound": float(bound),
                        "bound_normalized_derivative": float(direction[index, receiver] / bound) if bound > 0 else None,
                        "refinement_error": float(error[index, receiver]),
                        "converged": bool(quantities["converged"][index, receiver]),
                        "quadrature_order": int(quantities["order"][index, receiver]),
                    }
                )
        for threshold in mariner.SHAPE_THRESHOLDS:
            shape = {"rate": rate, "power": power, "threshold": threshold}
            for name, indices in geometry.subsets.items():
                for scope in NUISANCE_SCOPES:
                    design = nuisance_design(exposure[indices], shape, scope)
                    metrics, residual = residual_projection(design, direction[indices], error[indices])
                    key = f"{clock}__threshold_{threshold:g}__{name}__{scope}"
                    residuals[key] = residual
                    projections.append(
                        {
                            "clock": clock,
                            "shape": shape,
                            "subset": name,
                            "nuisance_scope": scope,
                            "residual_array_key": key,
                            **metrics,
                        }
                    )
    write_once(folder / "row_diagnostics.csv", pd.DataFrame(rows).to_csv(index=False, float_format="%.17g").encode())
    write_json(folder / "projections.json", projections)
    write_arrays(folder / "residuals.npz", residuals)
    bound = quantities["pair_bound"]
    excess = np.maximum.reduce(
        [quantities["path"] - bound, quantities["tied"] - bound, np.abs(quantities["raw_derivative"]) - bound]
    )
    summary = {
        "rate": rate,
        "power": power,
        "rows": len(geometry.rows),
        "projection_records": len(projections),
        "all_quadrature_converged": bool(np.all(quantities["converged"])),
        "max_integration_refinement_error": float(np.max(quantities["integration_error"])),
        "max_raw_tied_derivative": float(np.max(np.abs(quantities["raw_derivative"][quantities["is_tied"]]), initial=0)),
        "max_bound_excess_signed": float(np.max(excess)),
        "scientific_qualification": "not evaluated",
        "untested_interpretation": (
            "Early code acquisition can cause less interference while web mastery is low; "
            "later own acquisition can repair code state. Net task benefit depends on amplitudes, "
            "power and masses; no interior optimum or empirical gain follows."
        ),
    }
    write_json(folder / "summary.json", summary)
    names = ("pair_integrals.npz", "row_diagnostics.csv", "projections.json", "residuals.npz", "summary.json")
    write_json(
        folder / "complete.json",
        {
            "identity": identity,
            "shape": {"rate": rate, "power": power},
            "outputs": {name: sha256(folder / name) for name in names},
        },
    )


def run(output: Path) -> None:
    identity = require_approval(output)
    geometry = read_geometry(DATA)
    if (
        len(geometry.rows) != 346
        or int(geometry.rows.tied.sum()) != 20
        or geometry.alpha != protocol()["expected_alpha"]
    ):
        raise ValueError("Frozen StarCoder geometry changed")
    write_json(
        output / "geometry_receipt.json",
        {
            "identity": identity,
            "alpha": geometry.alpha,
            "buckets": geometry.buckets,
            "total_epoch_scales": geometry.total_scales.tolist(),
            "clock_source_weights": {key: value.tolist() for key, value in clock_weights(geometry.total_scales).items()},
            "subset_row_ids": {
                key: geometry.rows.iloc[value].row_id.tolist() for key, value in geometry.subsets.items()
            },
            "loaded_policy_keys": POLICY_KEYS,
            "outcomes_read": False,
        },
    )
    cells = []
    with threadpool_limits(limits=1):
        for i, rate in enumerate(mariner.SHAPE_RATES):
            for j, power in enumerate(mariner.SHAPE_POWERS):
                if require_approval(output) != identity:
                    raise ValueError("Identity changed during geometry calculation")
                folder = output / "cells" / f"rate_{i}_power_{j}"
                analyze_shape(geometry, rate, power, folder, identity)
                cells.append(json.loads((folder / "summary.json").read_text()))
    if require_approval(output) != identity:
        raise ValueError("Identity changed before final geometry receipt")
    write_json(
        output / "summary.json",
        {
            "identity": identity,
            "cells": cells,
            "all_quadrature_converged": all(cell["all_quadrature_converged"] for cell in cells),
            "scientific_qualification": "not evaluated; no coefficient, floor or noise identification claim",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "freeze":
        write_json(args.output / "execution_identity.json", frozen_identity(args.output))
    else:
        run(args.output)


if __name__ == "__main__":
    main()
