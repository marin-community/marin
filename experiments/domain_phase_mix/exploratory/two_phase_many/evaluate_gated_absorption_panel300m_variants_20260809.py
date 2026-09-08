# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Evaluate physical repetition-state variants on the 300M 39-bucket panel.

This is the grouped-outer-OOF counterpart to the WSD80 ablations in
``evaluate_gated_absorption_variants_20260809.py``.  It keeps the current
family-pooled benefit and corrected absorption blocks fixed while changing
only the state used for repetition harm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_gated_absorption_panel300m_20260808 as parent,
)

N_FOLDS = 3
N_INNER_FOLDS = 3
OPTIMIZER_SEED = 20260808

PARAMETER_BOUNDS = {
    "near": (0.0, 1.0),
    "gamma_0": (0.005, 1.5),
    "gamma_1": (0.005, 1.5),
    "gamma_2": (0.005, 1.5),
    "log_offset": (-5.0, -0.3),
    "tau": (0.2, 10.0),
    "damage_horizon": (0.0, 1.0),
    "log_knee": (-1.0, 3.0),
    "log_kappa": (-2.0, 2.5),
    "beta": (0.3, 20.0),
    "log_ridge": (-6.0, 1.0),
    "log_groundwork_scale": (-2.0, 1.5),
}
COMMON_PARAMETERS = (
    "near",
    "gamma_0",
    "gamma_1",
    "gamma_2",
    "log_offset",
    "tau",
)
GATE_PARAMETERS = ("log_kappa", "beta")


@dataclass(frozen=True)
class Variant:
    id: str
    damage: str
    absorption: bool = True
    groundwork_survival: bool = False

    @property
    def parameter_names(self) -> tuple[str, ...]:
        if self.damage == "horizon":
            damage_parameters = ("damage_horizon",)
        elif self.damage == "total_knee":
            damage_parameters = ("log_knee",)
        else:
            damage_parameters = ()
        gate_parameters = GATE_PARAMETERS if self.absorption else ()
        groundwork_parameters = ("log_groundwork_scale",) if self.groundwork_survival else ()
        return COMMON_PARAMETERS + damage_parameters + gate_parameters + groundwork_parameters + ("log_ridge",)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(PARAMETER_BOUNDS[name] for name in self.parameter_names)


VARIANTS = {
    "GA-300M-000": Variant("GA-300M-000", damage="horizon"),
    "GA-006": Variant("GA-006", damage="total"),
    "GA-007": Variant("GA-007", damage="split"),
    "GA-008": Variant("GA-008", damage="total_knee"),
    "GA-300M-009": Variant("GA-300M-009", damage="horizon", absorption=False),
    "GA-300M-010": Variant("GA-300M-010", damage="horizon", groundwork_survival=True),
}


def parameters(variant: Variant, shape: np.ndarray) -> dict[str, float]:
    """Name a variant's nonlinear parameters."""
    return dict(zip(variant.parameter_names, map(float, shape), strict=True))


def repeat_features(weights: np.ndarray, data, variant: Variant, values: dict[str, float]) -> list[np.ndarray]:
    """Return bucket-level repetition states in realized or latent epoch units."""
    tau = values["tau"]
    early = data.c0 * weights[:, 0, :]
    late = data.c1 * weights[:, 1, :]
    if variant.damage == "horizon":
        horizon = values["damage_horizon"]
        total_epochs = data.c0 + data.c1
        dose = total_epochs * ((1.0 - horizon) * weights[:, 0, :] + horizon * weights[:, 1, :])
        return [np.maximum(dose - 1.0, 0.0) ** tau]

    total_excess = np.maximum(early + late - 1.0, 0.0)
    if variant.damage == "total":
        return [total_excess**tau]
    if variant.damage == "total_knee":
        knee = 10.0 ** values["log_knee"]
        return [total_excess**tau / (1.0 + (total_excess / knee) ** tau)]
    if variant.damage == "split":
        early_excess = np.maximum(early - 1.0, 0.0)
        late_excess = total_excess - early_excess
        return [early_excess**tau, late_excess**tau]
    raise ValueError(f"Unknown damage law: {variant.damage}")


def design(weights: np.ndarray, data, variant: Variant, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the fixed benefit/gate design and the selected repetition state."""
    values = parameters(variant, shape)
    offset = 10.0 ** values["log_offset"]
    exponents = np.array([values["gamma_0"], values["gamma_1"], values["gamma_2"]])[data.family_index]
    total_epochs = data.c0 + data.c1

    def phase_weighted_dose(horizon: float) -> np.ndarray:
        return total_epochs * ((1.0 - horizon) * weights[:, 0, :] + horizon * weights[:, 1, :])

    early = data.c0 * weights[:, 0, :]
    late = data.c1 * weights[:, 1, :]
    near_benefit = (phase_weighted_dose(values["near"]) + offset) ** -exponents

    pooled_blocks = [
        parent.family_sums(near_benefit, data.family_index),
        parent.family_sums((phase_weighted_dose(1.0) + offset) ** -exponents, data.family_index),
    ]
    if variant.absorption:
        kappa = 10.0 ** values["log_kappa"]
        gate = early ** values["beta"] / (early ** values["beta"] + kappa ** values["beta"])
        absorbed = early + late * gate
        pooled_blocks.append(parent.family_sums((absorbed + offset) ** -exponents, data.family_index))
    if variant.groundwork_survival:
        scale = 10.0 ** values["log_groundwork_scale"]
        pooled_blocks.append(parent.family_sums(np.exp(-early / scale), data.family_index))
    for feature in repeat_features(weights, data, variant, values):
        pooled_blocks.append(parent.family_sums(feature, data.family_index))
    pooled = sum(block.shape[1] for block in pooled_blocks)
    constrained = np.column_stack([*pooled_blocks, near_benefit])
    return np.ones((len(weights), 1)), constrained, pooled


def select(data, rows: np.ndarray, seed: int, variant: Variant, optimizer_seed: int) -> np.ndarray:
    """Select nonlinear parameters using grouped inner folds of the supplied rows."""
    inner = parent.panel.grouped_folds(data.frame.iloc[rows].reset_index(drop=True), seed, N_INNER_FOLDS)
    response = data.y[rows]
    weights = data.weights[rows]

    def objective(shape: np.ndarray) -> float:
        free, constrained, pooled = design(weights, data, variant, shape)
        if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
            return 1e6
        ridge = 10.0 ** parameters(variant, shape)["log_ridge"]
        total = 0.0
        for train, test in inner:
            b, a = parent.fit_head(free[train], constrained[train], response[train], ridge, pooled)
            residual = free[test] @ b + constrained[test] @ a - response[test]
            total += float(residual @ residual)
        return total

    return differential_evolution(
        objective,
        variant.bounds,
        rng=np.random.default_rng(optimizer_seed),
        popsize=12,
        maxiter=80,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def paired_effects(data, predictions: np.ndarray) -> tuple[float, float, float, float, int]:
    """Compute aggregate-matched tied-minus-two-phase effects."""
    tied = np.all(np.isclose(data.weights[:, 0, :], data.weights[:, 1, :]), axis=1)
    keys = data.frame["phase_correspondence_key"].astype(str).to_numpy()
    predicted: list[float] = []
    observed: list[float] = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        moved = rows[~tied[rows]]
        held = rows[tied[rows]]
        if len(moved) and len(held):
            predicted.append(float(predictions[held].min() - predictions[moved].min()))
            observed.append(float(data.y[held].min() - data.y[moved].min()))
    return (
        float(np.mean(predicted)),
        float(np.mean(observed)),
        float(np.max(predicted)),
        float(np.max(observed)),
        len(predicted),
    )


def evaluate(data, seed: int, variant: Variant, optimizer_seed: int) -> dict[str, object]:
    """Return grouped outer-OOF fit, selection, pair effects, and full-fit diagnostics."""
    predictions = np.empty_like(data.y)
    shapes: list[dict[str, float]] = []
    for train, test in parent.panel.grouped_folds(data.frame, seed, N_FOLDS):
        shape = select(data, train, seed, variant, optimizer_seed)
        free, constrained, pooled = design(data.weights, data, variant, shape)
        values = parameters(variant, shape)
        b, a = parent.fit_head(free[train], constrained[train], data.y[train], 10.0 ** values["log_ridge"], pooled)
        predictions[test] = free[test] @ b + constrained[test] @ a
        shapes.append(values)

    full_shape = select(data, np.arange(data.n), seed, variant, optimizer_seed)
    free, constrained, pooled = design(data.weights, data, variant, full_shape)
    full_values = parameters(variant, full_shape)
    b, a = parent.fit_head(free, constrained, data.y, 10.0 ** full_values["log_ridge"], pooled)
    repeat_width = 2 if variant.damage == "split" else 1
    repeat_amplitudes = a[pooled - 3 * repeat_width : pooled]
    repeat_columns = constrained[:, pooled - 3 * repeat_width : pooled]
    repeat_contributions = repeat_amplitudes * np.std(repeat_columns, axis=0)

    selected = int(np.argmin(predictions))
    observed_best = int(np.argmin(data.y))
    predicted_mean, observed_mean, predicted_max, observed_max, n_pairs = paired_effects(data, predictions)
    return {
        "rmse": float(np.sqrt(np.mean((predictions - data.y) ** 2))),
        "regret_at_1": float(data.y[selected] - data.y[observed_best]),
        "predicted_pair_gain": predicted_mean,
        "observed_pair_gain": observed_mean,
        "predicted_max_pair_gain": predicted_max,
        "observed_max_pair_gain": observed_max,
        "n_pairs": n_pairs,
        "fold_shapes": shapes,
        "full_shape": full_values,
        "repeat_standardized_contributions": repeat_contributions.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=tuple(VARIANTS))
    parser.add_argument("target", choices=("uncheatable", "table9"))
    parser.add_argument("seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--optimizer-seed", type=int, default=OPTIMIZER_SEED)
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    data = parent.panel.load_300m(args.target)
    reference = parent.HPR[args.target]
    print(
        f"{variant.id}: target={args.target}, damage={variant.damage}, absorption={variant.absorption}, "
        f"groundwork_survival={variant.groundwork_survival}, "
        f"optimizer_seed={args.optimizer_seed}"
    )
    for seed in args.seeds:
        row = evaluate(data, seed, variant, args.optimizer_seed)
        rmse_ok = row["rmse"] <= reference["all_rmse"] * parent.RMSE_SLACK
        regret_ok = row["regret_at_1"] <= reference["regret_at_1"] + parent.REGRET_SLACK
        shape_summary = ", ".join(f"{name}={value:.4g}" for name, value in row["full_shape"].items())
        print(
            f"seed {seed}: OOF RMSE {row['rmse']:.6f}{'P' if rmse_ok else 'F'}; "
            f"Regret@1 {row['regret_at_1']:.6f}{'P' if regret_ok else 'F'}; "
            f"pair gain {row['predicted_pair_gain']:+.6f} observed {row['observed_pair_gain']:+.6f}; "
            f"max {row['predicted_max_pair_gain']:+.6f} observed {row['observed_max_pair_gain']:+.6f}"
        )
        print(f"         full shape: {shape_summary}")
        print(f"         repeat standardized contributions: {row['repeat_standardized_contributions']}")


if __name__ == "__main__":
    main()
