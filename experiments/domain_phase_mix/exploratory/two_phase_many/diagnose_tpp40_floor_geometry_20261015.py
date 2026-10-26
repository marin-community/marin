# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Local response sensitivity of the fixed synthetic acquisition/retention laws.

No parameter is fitted. The floor direction is compared with amplitude, shape,
and kinetic tangents on asymmetric training subsets. Noise-scaled inverse norms
are local unconstrained diagnostics, not confidence intervals or global IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_oracle_20261015 as oracle,
)

benchmark = oracle.benchmark
models = oracle.models
OUTPUT = oracle.PARENT / "floor_geometry"
CONTRACT: dict[str, Any] = {
    "scope": "synthetic_response_jacobian_no_fits",
    "worlds": [
        {"name": "native_equal", "clock": "equal_group", "mu": 0.0},
        {"name": "native_pool", "clock": "global_pool", "mu": 0.0},
        {"name": "positive_equal", "clock": "equal_group", "mu": 3.0},
        {"name": "positive_pool", "clock": "global_pool", "mu": 3.0},
    ],
    "head": oracle.HEAD,
    "shape": oracle.SHAPE,
    "steps": [0.001, 0.0005],
    "derivative_absolute_tolerance": 1e-6,
    "derivative_relative_tolerance": 1e-4,
    "residual_step_relative_tolerance": 0.05,
    "projection_rcond": 1e-10,
    "noise_sd": 0.003,
    "quadrature": oracle.CONTRACT["quadrature"],
    "outer_folds": [0, 1, 2, 3, 4],
    "subsets": ["full", "inner_train_0", "inner_train_1", "inner_train_2"],
}


def source_paths() -> set[Path]:
    return oracle.source_paths() | {Path(__file__), Path(__file__).with_name("test_tpp40_floor_geometry_20261015.py")}


def identity() -> dict[str, Any]:
    spec = json.loads((OUTPUT / "registration.json").read_text())
    if spec.get("status") != "registered" or spec.get("contract") != CONTRACT:
        raise ValueError("Register the fixed no-fit sensitivity scope before execution")
    if set(spec["source_hashes"]) != {str(p.resolve()) for p in source_paths()}:
        raise ValueError("Source pin set changed")
    benchmark.verify_hashes(spec["source_hashes"])
    benchmark.verify_hashes(spec["input_hashes"])
    if set(spec["input_hashes"]) != {str(p.resolve()) for p in oracle.input_paths()}:
        raise ValueError("Only the original covariate files are admitted")
    if spec["environment"] != benchmark.environment_identity():
        raise ValueError("Registered runtime changed")
    prior = oracle.OUTPUT / "result_manifest.json"
    if benchmark.sha256(prior) != spec["stage1_seal_sha256"]:
        raise ValueError("The sealed prerequisite changed")
    seal = json.loads(prior.read_text())
    benchmark.verify_hashes({name: row["sha256"] for name, row in seal["files"].items()}, oracle.OUTPUT)
    audit = json.loads((oracle.OUTPUT / "audit/independent_replay_report_v1.json").read_text())
    if not audit["passed"] or not audit["oracle_gates_passed"]:
        raise ValueError("Numerical and conditional oracle prerequisites must pass")
    return {"registration_sha256": benchmark.sha256(OUTPUT / "registration.json"), **spec}


def residualized(column: np.ndarray, nuisance: np.ndarray) -> dict[str, Any]:
    """Project after unit-norm scaling; preserve small residuals without flooring."""
    scales = np.linalg.norm(nuisance, axis=0)
    selected = scales > 0
    scaled = nuisance[:, selected] / scales[selected]
    left, singular, _ = svd(scaled, full_matrices=False, lapack_driver="gesvd")
    left, singular = np.asarray(left), np.asarray(singular)
    rank = int(np.sum(singular > CONTRACT["projection_rcond"] * singular[0])) if len(singular) else 0
    remainder = column - left[:, :rank] @ (left[:, :rank].T @ column)
    raw_norm, residual_norm = float(np.linalg.norm(column)), float(np.linalg.norm(remainder))
    return {
        "nuisance_rank": rank,
        "scaled_singular_values": singular.tolist(),
        "column_norm": raw_norm,
        "residual_norm": residual_norm,
        "retained_norm_ratio": residual_norm / raw_norm if raw_norm > 0 else None,
        "local_noise_scaled_inverse_norm": CONTRACT["noise_sd"] / residual_norm if residual_norm > 0 else None,
    }


def response(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    config: models.FeatureConfig,
    shape: dict[str, float],
    strength: float,
    cache: models.BenefitCache,
) -> tuple[np.ndarray, np.ndarray]:
    design, offset = models.design_and_offset(early, late, alpha, shape, models.MODEL, {"mu": strength}, config, cache)
    linear = oracle.HEAD["intercept"] + design @ np.asarray(oracle.HEAD["coefficients"]) + offset
    if not np.isfinite(linear).all() or np.any(abs(linear) >= models.mariner.LOG_CLIP):
        raise FloatingPointError("Jacobian response would clip")
    return oracle.HEAD["floor"] + np.exp(linear), design


def tangents(data: oracle.geometry.Geometry, world: dict[str, Any], cache: models.BenefitCache) -> dict[str, np.ndarray]:
    config = oracle.feature_config(data.total_scales, world["clock"])
    base, design = response(data.early, data.late, data.alpha, config, oracle.SHAPE, world["mu"], cache)
    q = base - oracle.HEAD["floor"]
    result = {"response": base, "amplitudes": q[:, None] * np.column_stack((np.ones(len(q)), design))}
    for step in CONTRACT["steps"]:
        derivatives = []
        for key in ("rate", "power", "threshold", "mu"):
            if key == "mu" and world["mu"] == 0:
                first = response(data.early, data.late, data.alpha, config, oracle.SHAPE, step, cache)[0]
                second = response(data.early, data.late, data.alpha, config, oracle.SHAPE, 2 * step, cache)[0]
                derivative = (-3 * base + 4 * first - second) / (2 * step)
            else:
                values = []
                for sign in (-1, 1):
                    shape, strength = dict(oracle.SHAPE), world["mu"]
                    if key == "mu":
                        strength += sign * step
                    else:
                        shape[key] *= np.exp(sign * step)
                    values.append(response(data.early, data.late, data.alpha, config, shape, strength, cache)[0])
                derivative = (values[1] - values[0]) / (2 * step)
            derivatives.append(derivative)
        result[f"derivatives_{step}"] = np.column_stack(derivatives)
    return result


def summarize_case(values: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
    coarse, fine = (values[f"derivatives_{step}"][indices] for step in CONTRACT["steps"])
    maximum = np.max(abs(fine), axis=0)
    errors = np.max(abs(fine - coarse), axis=0)
    limits = CONTRACT["derivative_absolute_tolerance"] + CONTRACT["derivative_relative_tolerance"] * maximum
    amplitudes, shape, phase = values["amplitudes"][indices], fine[:, :3], fine[:, 3]
    floor = np.ones(len(indices))
    known = np.column_stack((amplitudes, shape))
    fine_phase = residualized(phase, np.column_stack((known, floor)))
    fine_floor = residualized(floor, np.column_stack((known, phase)))
    coarse_phase = residualized(coarse[:, 3], np.column_stack((amplitudes, coarse[:, :3], floor)))
    coarse_floor = residualized(floor, np.column_stack((amplitudes, coarse)))
    stability = {}
    for name, first, second in (("phase", fine_phase, coarse_phase), ("floor", fine_floor, coarse_floor)):
        scale = max(first["residual_norm"], second["residual_norm"])
        change = abs(first["residual_norm"] - second["residual_norm"]) / scale if scale > 0 else None
        stability[name] = {
            "relative_residual_norm_change": change,
            "rank_agrees": first["nuisance_rank"] == second["nuisance_rank"],
            "step_resolved": (
                change is not None
                and change <= CONTRACT["residual_step_relative_tolerance"]
                and first["nuisance_rank"] == second["nuisance_rank"]
            ),
        }
    return {
        "count": len(indices),
        "indices": indices.tolist(),
        "finite_difference_errors": errors.tolist(),
        "finite_difference_limits": limits.tolist(),
        "finite_difference_resolved": bool(np.all(errors <= limits)),
        "phase_given_amplitudes": residualized(phase, amplitudes),
        "phase_given_shape_and_amplitudes": residualized(phase, known),
        "phase_given_floor_shape_and_amplitudes": fine_phase,
        "floor_given_shape_phase_and_amplitudes": fine_floor,
        "coarse_phase_given_floor_shape_and_amplitudes": coarse_phase,
        "coarse_floor_given_shape_phase_and_amplitudes": coarse_floor,
        "projected_residual_step_stability": stability,
    }


def run() -> None:
    frozen = identity()
    spec = {"stage": "response_jacobian", "contract": CONTRACT}
    if oracle.completed(OUTPUT, spec, frozen):
        return
    data, folds = oracle.inputs()
    cache = models.BenefitCache(OUTPUT / "features.sqlite")
    records = []
    for world in CONTRACT["worlds"]:
        folder = OUTPUT / world["name"]
        cell = {"world": world, "contract": CONTRACT}
        if not oracle.completed(folder, cell, frozen):
            values = tangents(data, world, cache)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "tangents.npz", **values)
            oracle.finish(folder, cell, frozen, ("tangents.npz",))
        with np.load(folder / "tangents.npz") as saved:
            values = dict(saved)
        for outer in CONTRACT["outer_folds"]:
            split = oracle.split_indices(data.rows, folds, outer)
            subsets = {"full": split["train"]} | {
                f"inner_train_{label}": split["train"][split["inner"] != label] for label in range(3)
            }
            for name, indices in subsets.items():
                records.append({"world": world, "outer_fold": outer, "subset": name, **summarize_case(values, indices)})
    benchmark.write_json(
        OUTPUT / "summary.json",
        {
            "records": records,
            "finite_difference_resolved": all(row["finite_difference_resolved"] for row in records),
            "scope": (
                "Local unconstrained response conditioning only; no fit, statistical interval or model qualification"
            ),
            "learned_floor_domain_failure_preserved": True,
            "boundary_note": "At mu0 the kinetic tangent is one-sided; projection uses an unconstrained linear span",
        },
    )
    oracle.finish(
        OUTPUT,
        spec,
        frozen,
        ("summary.json",),
        tuple(OUTPUT / world["name"] / "complete.json" for world in CONTRACT["worlds"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
