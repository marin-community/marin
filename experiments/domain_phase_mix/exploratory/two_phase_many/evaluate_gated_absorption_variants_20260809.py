# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Evaluate preregistered dimensional-consistency variants of SUR-102.

Variants isolate three questions: whether the pure-early column earns its
complexity, whether the absorption gate uses phase-correct epoch units, and
whether repetition harm should be computed from realized materialized epochs
rather than a freely weighted phase dose. See
``.agents/projects/gated_absorption_independent_registry_20260809.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, nnls

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import multitarget_ile_wsd80_20260806 as wsd

RPL_INTERIOR_RMSE = 0.007575
RPL_REGRET_LIMIT = 0.004842
OPTIMUM_DISTANCE_LIMIT = 0.05
GAIN_ERROR_LIMIT = harness.WSD_GAIN_ERROR_LIMIT
N_FOLDS = 3
N_INNER_FOLDS = 5
SURFACE_GRID = 801
OPTIMIZER_SEED = 20260807
COLUMN_RELATIVE_TOLERANCE = 1e-12
BROAD, CODE = 0, 1

PANEL, TARGETS = wsd.load_targets()
GEOMETRY = wsd.geometry()
TOTAL_EPOCHS = GEOMETRY.c0 + GEOMETRY.c1
INTERIOR = wsd.interior_mask(PANEL)
PRIMARY = TARGETS.values[:, TARGETS.names.index(harness.PRIMARY_TARGET)]

PARAMETER_BOUNDS = {
    "near": (0.0, 1.0),
    "gamma_broad": (0.005, 2.0),
    "gamma_code": (0.005, 1.5),
    "log_offset": (-5.0, -0.3),
    "tau": (0.2, 10.0),
    "damage_horizon": (0.0, 1.0),
    "kappa_code": (0.02, 20.0),
    "beta": (0.3, 40.0),
    "nu": (0.01, 2.0),
    "log_eps": (-6.0, 1.4),
    "kappa_broad": (0.02, 5.0),
    "log_groundwork_scale": (-2.0, 1.5),
    "groundwork_sharpness": (0.3, 20.0),
}
COMMON_PARAMETERS = ("near", "gamma_broad", "gamma_code", "log_offset", "tau")
GATE_PARAMETERS = ("kappa_code", "beta", "kappa_broad")
EARLY_PARAMETERS = ("nu", "log_eps")
GROUNDWORK_PARAMETERS = ("log_groundwork_scale", "groundwork_sharpness")


@dataclass(frozen=True)
class Variant:
    id: str
    damage: str
    corrected_gate_units: bool
    pure_early: bool
    signed_conflict: bool = True
    absorption: bool = True
    late_benefit: bool = True
    wide_exponents: bool = False
    groundwork_deficit: bool = False
    groundwork_survival: bool = False

    @property
    def parameter_names(self) -> tuple[str, ...]:
        damage = ("damage_horizon",) if self.damage == "horizon" else ()
        gate = GATE_PARAMETERS if self.absorption else ()
        early = EARLY_PARAMETERS if self.pure_early else ()
        groundwork = GROUNDWORK_PARAMETERS if self.groundwork_deficit else ()
        survival = ("log_groundwork_scale",) if self.groundwork_survival else ()
        return COMMON_PARAMETERS + damage + gate + early + groundwork + survival

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        bounds = dict(PARAMETER_BOUNDS)
        if self.wide_exponents:
            bounds["gamma_broad"] = (0.01, 10.0)
            bounds["nu"] = (0.01, 10.0)
        return tuple(bounds[name] for name in self.parameter_names)


VARIANTS = {
    "GA-000": Variant("GA-000", damage="horizon", corrected_gate_units=False, pure_early=True),
    "GA-001": Variant("GA-001", damage="horizon", corrected_gate_units=False, pure_early=False),
    "GA-002": Variant("GA-002", damage="horizon", corrected_gate_units=True, pure_early=True),
    "GA-003": Variant("GA-003", damage="total", corrected_gate_units=False, pure_early=True),
    "GA-004": Variant("GA-004", damage="split", corrected_gate_units=False, pure_early=True),
    "GA-005": Variant("GA-005", damage="total", corrected_gate_units=True, pure_early=False),
    "GA-009": Variant("GA-009", damage="horizon", corrected_gate_units=True, pure_early=True, signed_conflict=False),
    "GA-010": Variant("GA-010", damage="horizon", corrected_gate_units=True, pure_early=True, absorption=False),
    "GA-011": Variant("GA-011", damage="horizon", corrected_gate_units=True, pure_early=True, late_benefit=False),
    "GA-012": Variant("GA-012", damage="split", corrected_gate_units=True, pure_early=True),
    "GA-013": Variant(
        "GA-013",
        damage="horizon",
        corrected_gate_units=True,
        pure_early=True,
        absorption=False,
        wide_exponents=True,
    ),
    "GA-014": Variant("GA-014", damage="horizon", corrected_gate_units=True, pure_early=False, absorption=False),
    "GA-015": Variant(
        "GA-015",
        damage="horizon",
        corrected_gate_units=True,
        pure_early=False,
        absorption=False,
        groundwork_deficit=True,
    ),
    "GA-016": Variant(
        "GA-016",
        damage="horizon",
        corrected_gate_units=True,
        pure_early=False,
        absorption=False,
        groundwork_survival=True,
    ),
    "GA-017": Variant(
        "GA-017",
        damage="horizon",
        corrected_gate_units=True,
        pure_early=False,
        absorption=False,
        wide_exponents=True,
        groundwork_survival=True,
    ),
    "GA-018": Variant(
        "GA-018",
        damage="horizon",
        corrected_gate_units=True,
        pure_early=False,
        absorption=True,
        groundwork_survival=True,
    ),
    "GA-021": Variant(
        "GA-021",
        damage="horizon",
        corrected_gate_units=True,
        pure_early=False,
        absorption=False,
        wide_exponents=True,
    ),
    "GA-022": Variant(
        "GA-022",
        damage="horizon",
        corrected_gate_units=True,
        pure_early=False,
        signed_conflict=False,
        absorption=False,
        wide_exponents=True,
        groundwork_survival=True,
    ),
    "GA-023": Variant(
        "GA-023",
        damage="horizon",
        corrected_gate_units=True,
        pure_early=False,
        absorption=False,
        late_benefit=False,
        wide_exponents=True,
        groundwork_survival=True,
    ),
}


def parameters(variant: Variant, shape: np.ndarray) -> dict[str, float]:
    """Name a variant's nonlinear parameter vector."""
    return dict(zip(variant.parameter_names, map(float, shape), strict=True))


def phase_weighted_dose(weights: np.ndarray, domain: int, horizon: float) -> np.ndarray:
    """Return a dimensionless phase-utility-weighted dose in epoch units."""
    return TOTAL_EPOCHS[domain] * ((1.0 - horizon) * weights[:, 0, domain] + horizon * weights[:, 1, domain])


def absorbed_dose(weights: np.ndarray, domain: int, scale: float, sharpness: float, corrected: bool) -> np.ndarray:
    """Return late dose gated by early groundwork, under either unit convention."""
    if corrected:
        early = GEOMETRY.c0[domain] * weights[:, 0, domain]
        late = GEOMETRY.c1[domain] * weights[:, 1, domain]
    else:
        early = TOTAL_EPOCHS[domain] * weights[:, 0, domain]
        late = TOTAL_EPOCHS[domain] * weights[:, 1, domain]
    gate = early**sharpness / (early**sharpness + scale**sharpness)
    return early + late * gate


def repeat_columns(weights: np.ndarray, variant: Variant, values: dict[str, float]) -> list[np.ndarray]:
    """Build repetition-harm columns under the selected physical law."""
    tau = values["tau"]
    if variant.damage == "horizon":
        excess = np.maximum(phase_weighted_dose(weights, CODE, values["damage_horizon"]) - 1.0, 0.0)
        return [excess**tau]

    early = GEOMETRY.c0[CODE] * weights[:, 0, CODE]
    late = GEOMETRY.c1[CODE] * weights[:, 1, CODE]
    total_excess = np.maximum(early + late - 1.0, 0.0)
    if variant.damage == "total":
        return [total_excess**tau]
    if variant.damage == "split":
        early_excess = np.maximum(early - 1.0, 0.0)
        late_excess = total_excess - early_excess
        return [early_excess**tau, late_excess**tau]
    raise ValueError(f"Unknown damage law: {variant.damage}")


def design(weights: np.ndarray, variant: Variant, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build free-sign and nonnegative design blocks."""
    values = parameters(variant, shape)
    offset = 10.0 ** values["log_offset"]
    free_columns = [np.ones(len(weights))]
    if variant.signed_conflict:
        free_columns.append(weights[:, 1, BROAD])
    free = np.column_stack(free_columns)
    constrained = [
        (phase_weighted_dose(weights, CODE, values["near"]) + offset) ** -values["gamma_code"],
        (phase_weighted_dose(weights, BROAD, values["near"]) + offset) ** -values["gamma_broad"],
    ]
    if variant.late_benefit:
        constrained.extend(
            [
                (phase_weighted_dose(weights, CODE, 1.0) + offset) ** -values["gamma_code"],
                (phase_weighted_dose(weights, BROAD, 1.0) + offset) ** -values["gamma_broad"],
            ]
        )
    if variant.absorption:
        constrained.extend(
            [
                (
                    absorbed_dose(
                        weights,
                        CODE,
                        values["kappa_code"],
                        values["beta"],
                        variant.corrected_gate_units,
                    )
                    + offset
                )
                ** -values["gamma_code"],
                (
                    absorbed_dose(
                        weights,
                        BROAD,
                        values["kappa_broad"],
                        values["beta"],
                        variant.corrected_gate_units,
                    )
                    + offset
                )
                ** -values["gamma_broad"],
            ]
        )
    if variant.pure_early:
        eps = 10.0 ** values["log_eps"]
        constrained.append((phase_weighted_dose(weights, CODE, 0.0) + eps) ** -values["nu"])
    if variant.groundwork_deficit:
        early = GEOMETRY.c0[CODE] * weights[:, 0, CODE]
        scale = 10.0 ** values["log_groundwork_scale"]
        constrained.append(1.0 / (1.0 + (early / scale) ** values["groundwork_sharpness"]))
    if variant.groundwork_survival:
        early = GEOMETRY.c0[CODE] * weights[:, 0, CODE]
        scale = 10.0 ** values["log_groundwork_scale"]
        constrained.append(np.exp(-early / scale))
    constrained.extend(repeat_columns(weights, variant, values))
    return free, np.column_stack(constrained)


def fit_head(free: np.ndarray, constrained: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the partitioned free-sign and nonnegative head after scaling."""
    basis, _ = np.linalg.qr(free)
    columns = constrained - basis @ (basis.T @ constrained)
    target = response - basis @ (basis.T @ response)
    scale = np.linalg.norm(columns, axis=0)
    active = scale > COLUMN_RELATIVE_TOLERANCE * np.max(scale, initial=0.0)
    amplitudes = np.zeros(constrained.shape[1])
    if np.any(active):
        fitted, _ = nnls(columns[:, active] / scale[active], target, maxiter=20000)
        amplitudes[active] = fitted / scale[active]
    free_amplitudes = np.linalg.lstsq(free, response - constrained @ amplitudes, rcond=None)[0]
    return free_amplitudes, amplitudes


def select(
    response: np.ndarray,
    rows: np.ndarray,
    seed: int,
    variant: Variant,
    optimizer_seed: int = OPTIMIZER_SEED,
    fold_mode: str = "random",
) -> np.ndarray:
    """Select nonlinear parameters using inner folds of the supplied rows."""
    folds = harness.wsd80_folds(fold_mode, PANEL.weights[rows], np.arange(len(rows)), N_INNER_FOLDS, seed)
    subset = response[rows]
    subset_interior = INTERIOR[rows]

    def objective(shape: np.ndarray) -> float:
        free, constrained = design(PANEL.weights[rows], variant, shape)
        if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
            return 1e3
        total = 0.0
        for train, test in folds:
            b, a = fit_head(free[train], constrained[train], subset[train])
            residual = free[test] @ b + constrained[test] @ a - subset[test]
            scored = residual[subset_interior[test]]
            if len(scored):
                total += float(scored @ scored)
        return total

    return differential_evolution(
        objective,
        variant.bounds,
        rng=np.random.default_rng(optimizer_seed),
        popsize=14,
        maxiter=120,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def evaluate(
    variant: Variant,
    seed: int,
    response: np.ndarray = PRIMARY,
    optimizer_seed: int = OPTIMIZER_SEED,
    fold_mode: str = "random",
) -> dict[str, object]:
    """Evaluate one variant under the frozen nested WSD80 protocol."""
    outer = harness.wsd80_folds(fold_mode, PANEL.weights, np.arange(len(response)), N_FOLDS, seed)
    predictions = np.empty_like(response)
    for train, test in outer:
        shape = select(response, train, seed, variant, optimizer_seed, fold_mode)
        free, constrained = design(PANEL.weights, variant, shape)
        b, a = fit_head(free[train], constrained[train], response[train])
        predictions[test] = free[test] @ b + constrained[test] @ a

    interior_rows = np.flatnonzero(INTERIOR)
    observed_best = int(interior_rows[np.argmin(response[interior_rows])])
    ranked = interior_rows[np.argsort(predictions[interior_rows])]

    shape = select(response, np.arange(len(response)), seed, variant, optimizer_seed, fold_mode)
    free, constrained = design(PANEL.weights, variant, shape)
    b, a = fit_head(free, constrained, response)
    axis = np.linspace(0.0, 1.0, SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = grid_0.ravel(), grid_1.ravel()
    grid_free, grid_constrained = design(wsd.grid_weights(flat_0, flat_1), variant, shape)
    surface = grid_free @ b + grid_constrained @ a
    tied_axis = np.linspace(0.0, 1.0, SURFACE_GRID * SURFACE_GRID // 4)
    tied_free, tied_constrained = design(wsd.grid_weights(tied_axis, tied_axis), variant, shape)
    tied = tied_free @ b + tied_constrained @ a
    best = int(np.argmin(surface))
    optimum = (float(flat_0[best]), float(flat_1[best]))
    observed_optimum = (float(PANEL.phase_0[observed_best, 1]), float(PANEL.phase_1[observed_best, 1]))
    repeat_width = 2 if variant.damage == "split" else 1
    repeat_amplitudes = a[-repeat_width:]
    repeat_contributions = repeat_amplitudes * np.std(constrained[:, -repeat_width:], axis=0)
    return {
        "variant": variant.id,
        "seed": seed,
        "rmse": float(np.sqrt(np.mean((predictions - response)[INTERIOR] ** 2))),
        "regret_1": float(response[ranked[0]] - response[observed_best]),
        "regret_5": float(response[ranked[:5]].min() - response[observed_best]),
        "optimum": optimum,
        "distance": float(np.hypot(optimum[0] - observed_optimum[0], optimum[1] - observed_optimum[1])),
        "gain": float(tied.min() - surface.min()),
        "shape": parameters(variant, shape),
        "repeat_amplitudes": repeat_amplitudes.tolist(),
        "repeat_standardized_contributions": repeat_contributions.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=tuple(VARIANTS))
    parser.add_argument("seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--target", choices=TARGETS.names, default=harness.PRIMARY_TARGET)
    parser.add_argument("--optimizer-seed", type=int, default=OPTIMIZER_SEED)
    parser.add_argument("--fold-mode", choices=("random", "blocked"), default="random")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    response = TARGETS.values[:, TARGETS.names.index(args.target)]
    primary = args.target == harness.PRIMARY_TARGET
    print(
        f"{variant.id}: target={args.target}, damage={variant.damage}, "
        f"corrected_gate_units={variant.corrected_gate_units}, pure_early={variant.pure_early}, "
        f"signed_conflict={variant.signed_conflict}, absorption={variant.absorption}, "
        f"late_benefit={variant.late_benefit}, wide_exponents={variant.wide_exponents}, "
        f"groundwork_deficit={variant.groundwork_deficit}, groundwork_survival={variant.groundwork_survival}, "
        f"optimizer_seed={args.optimizer_seed}, fold_mode={args.fold_mode}"
    )
    for seed in args.seeds:
        row = evaluate(variant, seed, response, args.optimizer_seed, args.fold_mode)
        shape_summary = ", ".join(f"{name}={value:.4g}" for name, value in row["shape"].items())
        if not primary:
            print(
                f"seed {seed}: RMSE {row['rmse']:.6f}; Regret@1 {row['regret_1']:.6f}; "
                f"optimum ({row['optimum'][0]:.3f},{row['optimum'][1]:.3f}); gain {row['gain']:+.6f}; "
                f"repeat standardized contributions {row['repeat_standardized_contributions']}"
            )
            print(f"         full shape: {shape_summary}")
            continue
        gain_error = abs(float(row["gain"]) - harness.OBSERVED_WSD_GAIN)
        checks = (
            float(row["rmse"]) <= RPL_INTERIOR_RMSE * 1.05,
            float(row["regret_1"]) <= RPL_REGRET_LIMIT,
            float(row["distance"]) <= OPTIMUM_DISTANCE_LIMIT,
            gain_error <= GAIN_ERROR_LIMIT,
        )
        print(
            f"seed {seed}: RMSE {row['rmse']:.6f}{'P' if checks[0] else 'F'}; "
            f"Regret@1 {row['regret_1']:.6f}{'P' if checks[1] else 'F'}; "
            f"distance {row['distance']:.6f}{'P' if checks[2] else 'F'}; "
            f"gain error {gain_error:.6f}{'P' if checks[3] else 'F'}; "
            f"optimum ({row['optimum'][0]:.3f},{row['optimum'][1]:.3f}); "
            f"gain {row['gain']:+.6f}; repeat amplitudes {row['repeat_amplitudes']}; "
            f"repeat standardized contributions {row['repeat_standardized_contributions']}; total {sum(checks)}/4"
        )
        print(f"         full shape: {shape_summary}")


if __name__ == "__main__":
    main()
