# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""WSD80-SUR-095 on the 80/20 StarCoder panel, with shape selection inside inner folds.

Selection is nested. Every nonlinear parameter is chosen on inner splits of an outer fold's training
rows and never sees that fold's outcomes, because the previous candidate's headline was withdrawn for
exactly this: it selected 17,280 shapes on the same three partitions it then reported as out-of-fold,
and a matched nested rerun moved interior RMSE from 0.006746 to 0.008040.

Four gates are reported against frozen references, plus three things the review asked for and the
previous artifact did not supply: the raw unclipped optimum alongside the interior one, the two-phase
gain labelled as the in-sample development diagnostic it is, and the forgetting-rate ablation that
returns the model to the refuted single-index null.

Run with no arguments for the primary fit. `python ... controls` adds the broad-text negative controls
and the transfer metrics.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    acquisition_forgetting_state_20260807 as afs,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_wsd80_20260806 as wsd,
)

# Frozen references, both measured under random 3-fold seed 0 on this panel's interior rows.
RPL_INTERIOR_RMSE = 0.007575
RPL_REGRET_LIMIT = 0.004842
OPTIMUM_DISTANCE_LIMIT = 0.05
CONTROL_GAIN_LIMIT = 0.005

# Zero is the single-index null and is deliberately inside the grid: the impossibility argument says it
# cannot beat the tied class, so the selector is free to return the refuted family if it fits better.
FORGETTING_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 35.0)
RHO_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0)
# Readout exponents are separated by domain because the cross-scale sweep measured that they genuinely
# differ, code shallow and broad text steep, and gave the bands these grids cover.
CODE_EXPONENT_GRID = (0.05, 0.1, 0.2, 0.3, 0.5)
BROAD_EXPONENT_GRID = (0.15, 0.3, 0.5, 1.0)
OFFSET_GRID = (0.01, 0.05, 0.2)
DAMAGE_EXPONENT_GRID = (1.0, 2.0, 3.0, 4.0)
N_FOLDS = 3
SEED = 0
SURFACE_GRID = 401

SHAPES = tuple(
    afs.Shape(rho=rho, forgetting=(forgetting,), exponent=(broad, code), offset=offset, damage_exponent=exponent)
    for rho in RHO_GRID
    for forgetting in FORGETTING_GRID
    for broad in BROAD_EXPONENT_GRID
    for code in CODE_EXPONENT_GRID
    for offset in OFFSET_GRID
    for exponent in DAMAGE_EXPONENT_GRID
)


def fit_head(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Least squares with a free intercept and non-negative amplitudes.

    Sweeping out the intercept first turns the sign-constrained problem into a plain non-negative least
    squares on the centred columns, which is both exact and fast enough to sit inside a nested search.
    """
    centre_design = design[:, 1:].mean(axis=0)
    centre_response = float(response.mean())
    amplitudes, _ = nnls(design[:, 1:] - centre_design, response - centre_response)
    return np.concatenate([[centre_response - centre_design @ amplitudes], amplitudes])


def fold_error(design: np.ndarray, response: np.ndarray, folds) -> float:
    total = 0.0
    for train, test in folds:
        residual = design[test] @ fit_head(design[train], response[train]) - response[test]
        total += float(residual @ residual)
    return total


def select_shape(weights: np.ndarray, response: np.ndarray, rows: np.ndarray, protocol: str) -> afs.Shape:
    """Pick the shape on inner splits of `rows` only. The caller's held-out rows are absent by construction."""
    inner = harness.wsd80_folds(protocol, weights[rows], np.arange(len(rows)), N_FOLDS, SEED)
    scored = [
        (fold_error(afs.design_matrix(weights[rows], GEOMETRY, shape), response[rows], inner), index)
        for index, shape in enumerate(SHAPES)
    ]
    return SHAPES[min(scored)[1]]


def nested_predictions(weights: np.ndarray, response: np.ndarray, protocol: str):
    """Out-of-fold predictions where the nonlinear shape is re-selected for every outer fold."""
    outer = harness.wsd80_folds(protocol, weights, np.arange(len(response)), N_FOLDS, SEED)
    predictions = np.empty_like(response)
    selected = []
    for train, test in outer:
        shape = select_shape(weights, response, train, protocol)
        design = afs.design_matrix(weights, GEOMETRY, shape)
        predictions[test] = design[test] @ fit_head(design[train], response[train])
        selected.append(shape)
    return predictions, selected


def surface(shape: afs.Shape, coefficients: np.ndarray):
    """Predicted loss over the two-phase square and over the tied diagonal, at matched resolution."""
    axis = np.linspace(0.0, 1.0, SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = grid_0.ravel(), grid_1.ravel()
    predicted = afs.design_matrix(wsd.grid_weights(flat_0, flat_1), GEOMETRY, shape) @ coefficients
    tied_axis = np.linspace(0.0, 1.0, SURFACE_GRID * SURFACE_GRID)
    tied = afs.design_matrix(wsd.grid_weights(tied_axis, tied_axis), GEOMETRY, shape) @ coefficients
    return flat_0, flat_1, predicted, tied


def report(protocol: str, target_name: str, response: np.ndarray, verbose: bool = True) -> dict:
    predictions, selected = nested_predictions(PANEL.weights, response, protocol)
    interior_rows = np.flatnonzero(INTERIOR)
    rmse = float(np.sqrt(np.mean((predictions - response)[INTERIOR] ** 2)))

    ranked = interior_rows[np.argsort(predictions[interior_rows])]
    observed_best = int(interior_rows[np.argmin(response[interior_rows])])
    regret = float(response[ranked[0]] - response[observed_best])

    # The full-panel fit is a development surface, not a validated prediction: its coefficients see
    # every outcome the observed gain is computed from. Reported, and labelled, on that basis.
    shape = select_shape(PANEL.weights, response, np.arange(len(response)), protocol)
    design = afs.design_matrix(PANEL.weights, GEOMETRY, shape)
    coefficients = fit_head(design, response)
    flat_0, flat_1, predicted, tied = surface(shape, coefficients)

    raw = int(np.argmin(predicted))
    margin = harness.BOUNDARY_MARGIN
    inside = (flat_0 > margin) & (flat_1 > margin) & (flat_0 < 1 - margin) & (flat_1 < 1 - margin)
    interior_grid = np.flatnonzero(inside)
    clipped = int(interior_grid[np.argmin(predicted[interior_grid])])
    target_0, target_1 = PANEL.phase_0[observed_best, 1], PANEL.phase_1[observed_best, 1]
    distance = lambda row: float(np.hypot(flat_0[row] - target_0, flat_1[row] - target_1))  # noqa: E731

    result = {
        "protocol": protocol,
        "target": target_name,
        "rmse": rmse,
        "regret": regret,
        "raw_gain": float(tied.min() - predicted[raw]),
        "interior_gain": float(tied.min() - predicted[clipped]),
        "raw_optimum": (float(flat_0[raw]), float(flat_1[raw])),
        "interior_optimum": (float(flat_0[clipped]), float(flat_1[clipped])),
        "raw_distance": distance(raw),
        "interior_distance": distance(clipped),
        "shape": shape,
        "selected": selected,
        "coefficients": coefficients,
    }
    if not verbose:
        return result

    print(f"\n=== {target_name}   protocol {protocol} ===")
    print(
        f"  selected on full panel: rho {shape.rho}  forgetting {shape.forgetting}  exponent {shape.exponent}"
        f"  offset {shape.offset}  tau {shape.damage_exponent}"
    )
    print(f"  per-outer-fold forgetting: {[s.forgetting for s in selected]}")
    print(f"  amplitudes: intercept {coefficients[0]:.6f}  benefit {coefficients[1:-1]}  damage {coefficients[-1]:.6f}")
    ok = rmse <= RPL_INTERIOR_RMSE * 1.05
    print(
        f"  nested interior OOF RMSE {rmse:.6f}  (RPL {RPL_INTERIOR_RMSE}, +5% = {RPL_INTERIOR_RMSE*1.05:.6f})"
        f"  {'PASS' if ok else 'FAIL'}"
    )
    print(f"  Regret@1 {regret:.6f}  limit {RPL_REGRET_LIMIT}  {'PASS' if regret <= RPL_REGRET_LIMIT else 'FAIL'}")
    print(
        f"  raw optimum      ({result['raw_optimum'][0]:.3f},{result['raw_optimum'][1]:.3f})  "
        f"distance {result['raw_distance']:.4f}  gain {result['raw_gain']:+.6f}"
    )
    print(
        f"  interior optimum ({result['interior_optimum'][0]:.3f},{result['interior_optimum'][1]:.3f})  "
        f"distance {result['interior_distance']:.4f}  gain {result['interior_gain']:+.6f}"
    )
    passes = result["raw_distance"] <= OPTIMUM_DISTANCE_LIMIT
    print(f"  optimum distance gate <= {OPTIMUM_DISTANCE_LIMIT} on the RAW optimum: {'PASS' if passes else 'FAIL'}")
    error = abs(result["raw_gain"] - harness.OBSERVED_WSD_GAIN)
    print(f"  [development only] gain error {error:.6f}  vs limit {harness.WSD_GAIN_ERROR_LIMIT}")
    return result


def ablation(protocol: str, response: np.ndarray) -> None:
    """Refit with the forgetting rate pinned at zero: the exact single-index null the panel refutes."""
    shapes = tuple(s for s in SHAPES if s.forgetting == (0.0,))
    outer = harness.wsd80_folds(protocol, PANEL.weights, np.arange(len(response)), N_FOLDS, SEED)
    predictions = np.empty_like(response)
    for train, test in outer:
        inner = harness.wsd80_folds(protocol, PANEL.weights[train], np.arange(len(train)), N_FOLDS, SEED)
        scored = [
            (fold_error(afs.design_matrix(PANEL.weights[train], GEOMETRY, s), response[train], inner), i)
            for i, s in enumerate(shapes)
        ]
        design = afs.design_matrix(PANEL.weights, GEOMETRY, shapes[min(scored)[1]])
        predictions[test] = design[test] @ fit_head(design[train], response[train])
    rmse = float(np.sqrt(np.mean((predictions - response)[INTERIOR] ** 2)))

    best = min(
        (fold_error(afs.design_matrix(PANEL.weights, GEOMETRY, s), response, [(np.arange(len(response)),) * 2]), i)
        for i, s in enumerate(shapes)
    )
    shape = shapes[best[1]]
    design = afs.design_matrix(PANEL.weights, GEOMETRY, shape)
    _, _, predicted, tied = surface(shape, fit_head(design, response))
    print(f"\n--- forgetting = 0 ablation ({protocol}) : the refuted single-index null ---")
    print(f"  nested interior OOF RMSE {rmse:.6f}   predicted two-phase gain {tied.min() - predicted.min():+.9f}")


PANEL, TARGETS = wsd.load_targets()
_c0, _c1 = wsd.geometry().c0, wsd.geometry().c1
GEOMETRY = afs.Geometry(epochs_per_unit_weight=_c0 + _c1, phase_1_fraction=wsd.wsd80.REALIZED_PHASE_1_FRACTION)
INTERIOR = wsd.interior_mask(PANEL)
PRIMARY = TARGETS.values[:, TARGETS.names.index(harness.PRIMARY_TARGET)]


def main() -> None:
    print(f"panel {len(PANEL.y)} rows, {int(INTERIOR.sum())} interior, {len(SHAPES)} shapes, nested {N_FOLDS}x{N_FOLDS}")
    for _protocol in ("random", "blocked"):
        report(_protocol, harness.PRIMARY_TARGET, PRIMARY)
        ablation(_protocol, PRIMARY)

    if len(sys.argv) > 1 and sys.argv[1] == "controls":
        print("\n=== transfer and negative controls (random protocol) ===")
        for _name in TARGETS.names:
            if _name == harness.PRIMARY_TARGET:
                continue
            _row = report("random", _name, TARGETS.values[:, TARGETS.names.index(_name)], verbose=False)
            _flag = "" if _row["raw_gain"] <= CONTROL_GAIN_LIMIT else "  <-- above control limit"
            print(
                f"  {_name:70s} rmse {_row['rmse']:.6f}  gain {_row['raw_gain']:+.6f}"
                f"  forgetting {_row['shape'].forgetting:5.2f}{_flag}"
            )


if __name__ == "__main__":
    main()
