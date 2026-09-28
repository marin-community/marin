# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""WSD80-SUR-096: the two-exposure form under nested selection, with every flagged grid extended.

This is the SUR-094 response surface evaluated the way the frozen ladder requires and the way the
previous artifact did not. Three things change, all of them because independent review found them:

Selection is nested. Nonlinear parameters are chosen on inner splits of each outer fold's training rows
and never see that fold's outcomes. Reporting selection-set error as out-of-fold is what inflated the
previous headline by 19.2 percent.

Every grid that the previous fit selected at a bound is extended past it. `slow` stopped at 0.45 and
selected 0.45; it now runs to 0.65. `gamma_broad` stopped at 0.2 and selected 0.2; it now runs down to
0.02. `tau` stopped at 3.0; it now runs to 6.0. A parameter that still lands on a bound is reported as
unidentified rather than quietly accepted.

The optimum is reported raw as well as interior-clipped, and the gain is labelled for what it is: a
development diagnostic computed from coefficients that have seen every outcome it is compared against.

On interpretation, `slow` is a per-token consolidation ratio, not a free knob. A horizon of `phi`
against a realized phase-1 token share of `p` means late tokens count `(phi / p) / ((1 - phi) / (1 - p))`
times an early token for that channel. The fitted value corresponds to about a factor of three, which is
the direction and rough size the decay phase of a WSD schedule is expected to produce. That reading is
offered as interpretation only. It is not a mechanism this panel can test, because any horizon in
`[0.113, 1]` is reachable from some forgetting rate under this schedule, so the derivation adds no
constraint here. Testing it needs a panel with a different schedule.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import itertools  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_wsd80_20260806 as wsd,
)

RPL_INTERIOR_RMSE = 0.007575
RPL_REGRET_LIMIT = 0.004842
OPTIMUM_DISTANCE_LIMIT = 0.05
CONTROL_GAIN_LIMIT = 0.005
N_FOLDS = 3
SEED = 0
SURFACE_GRID = 401

# Extended past every bound the previous fit selected at. The fast horizon stays at 1.0 because that is
# the physical endpoint "only the decay phase counts", not an arbitrary cap.
SLOW = (0.05, 0.10, 0.15, 0.203, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65)
FAST = 1.0
DAMAGE_HORIZON = (0.05, 0.1, 0.203, 0.3, 0.45, 0.6, 0.8, 1.0)
# 0.02 is INCLUDED. It was briefly excluded on the argument that the column tends to a constant as the
# exponent tends to zero and is therefore collinear with the intercept. That argument was checked and is
# false at this exponent: the benefit block's condition number at 0.02 has median 427 and maximum 1239
# over the whole grid, which no standard threshold separates from the values that were kept, and the
# criterion is computed from the policy weights alone so it cannot be tuned. Excluding 0.02 improves
# Regret@1 from 0.005936 to 0.002842, which is exactly why it must stay: removing it was post-outcome
# grid tuning with a wrong justification attached, and the headline has to be the pre-registered grid.
BROAD_EXPONENT = (0.02, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0)
CODE_EXPONENT = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5)
OFFSET = (0.005, 0.01, 0.05, 0.2)
DAMAGE_EXPONENT = (1.0, 2.0, 3.0, 4.0, 6.0)

PANEL, TARGETS = wsd.load_targets()
GEOMETRY = wsd.geometry()
EPOCHS_PER_UNIT = GEOMETRY.c0 + GEOMETRY.c1
PHASE_1_FRACTION = wsd.wsd80.REALIZED_PHASE_1_FRACTION
INTERIOR = wsd.interior_mask(PANEL)
CODE, BROAD = 1, 0

SHAPES = tuple(itertools.product(SLOW, DAMAGE_HORIZON, BROAD_EXPONENT, CODE_EXPONENT, OFFSET, DAMAGE_EXPONENT))
BOUNDS = {
    "slow": (min(SLOW), max(SLOW)),
    "damage_horizon": (min(DAMAGE_HORIZON), max(DAMAGE_HORIZON)),
    "gamma_broad": (min(BROAD_EXPONENT), max(BROAD_EXPONENT)),
    "gamma_code": (min(CODE_EXPONENT), max(CODE_EXPONENT)),
    "offset": (min(OFFSET), max(OFFSET)),
    "tau": (min(DAMAGE_EXPONENT), max(DAMAGE_EXPONENT)),
}


def exposure(weights: np.ndarray, horizon: float) -> np.ndarray:
    """Epochs of each bucket as seen by a channel that weights the decay phase by `horizon`."""
    return EPOCHS_PER_UNIT * ((1.0 - horizon) * weights[:, 0, :] + horizon * weights[:, 1, :])


def per_token_ratio(horizon: float) -> float:
    """How much a decay-phase token counts against a stable-phase one for a channel at `horizon`."""
    return (horizon / PHASE_1_FRACTION) / ((1.0 - horizon) / (1.0 - PHASE_1_FRACTION))


def design_matrix(weights: np.ndarray, shape) -> np.ndarray:
    slow, damage_horizon, gamma_broad, gamma_code, offset, tau = shape
    near, late = exposure(weights, slow), exposure(weights, FAST)
    repetition = exposure(weights, damage_horizon)
    return np.column_stack(
        [
            np.ones(len(weights)),
            (near[:, CODE] + offset) ** -gamma_code,
            (near[:, BROAD] + offset) ** -gamma_broad,
            (late[:, CODE] + offset) ** -gamma_code,
            (late[:, BROAD] + offset) ** -gamma_broad,
            np.maximum(repetition[:, CODE] - 1.0, 0.0) ** tau,
        ]
    )


def fit_head(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Free intercept, non-negative amplitudes, by sweeping the intercept out before the sign solve."""
    centre_design = design[:, 1:].mean(axis=0)
    centre_response = float(response.mean())
    amplitudes, _ = nnls(design[:, 1:] - centre_design, response - centre_response)
    return np.concatenate([[centre_response - centre_design @ amplitudes], amplitudes])


def build_columns(weights: np.ndarray) -> dict:
    """Every column any shape can need, built once. Assembling a design is then a column stack."""
    cache = {}
    for horizon in (*SLOW, FAST):
        values = exposure(weights, horizon)
        for offset in OFFSET:
            for gamma in CODE_EXPONENT:
                cache["c", horizon, offset, gamma] = (values[:, CODE] + offset) ** -gamma
            for gamma in BROAD_EXPONENT:
                cache["b", horizon, offset, gamma] = (values[:, BROAD] + offset) ** -gamma
    for horizon in DAMAGE_HORIZON:
        excess = np.maximum(exposure(weights, horizon)[:, CODE] - 1.0, 0.0)
        for tau in DAMAGE_EXPONENT:
            cache["d", horizon, tau] = excess**tau
    return cache


def cached_design(cache: dict, shape) -> np.ndarray:
    slow, damage_horizon, gamma_broad, gamma_code, offset, tau = shape
    return np.column_stack(
        [
            ONES,
            cache["c", slow, offset, gamma_code],
            cache["b", slow, offset, gamma_broad],
            cache["c", FAST, offset, gamma_code],
            cache["b", FAST, offset, gamma_broad],
            cache["d", damage_horizon, tau],
        ]
    )


def select_shape(response: np.ndarray, rows: np.ndarray, protocol: str):
    """Choose the nonlinear shape using only `rows`, split into inner folds."""
    cache = build_columns(PANEL.weights[rows])
    inner = harness.wsd80_folds(protocol, PANEL.weights[rows], np.arange(len(rows)), N_FOLDS, SEED)
    subset = response[rows]
    best = (np.inf, None)
    for shape in SHAPES:
        design = np.column_stack(
            [
                np.ones(len(rows)),
                cache["c", shape[0], shape[4], shape[3]],
                cache["b", shape[0], shape[4], shape[2]],
                cache["c", FAST, shape[4], shape[3]],
                cache["b", FAST, shape[4], shape[2]],
                cache["d", shape[1], shape[5]],
            ]
        )
        total = 0.0
        for train, test in inner:
            residual = design[test] @ fit_head(design[train], subset[train]) - subset[test]
            total += float(residual @ residual)
        if total < best[0]:
            best = (total, shape)
    return best[1]


def evaluate(protocol: str, name: str, response: np.ndarray, verbose: bool = True) -> dict:
    outer = harness.wsd80_folds(protocol, PANEL.weights, np.arange(len(response)), N_FOLDS, SEED)
    predictions = np.empty_like(response)
    per_fold = []
    for train, test in outer:
        shape = select_shape(response, train, protocol)
        design = design_matrix(PANEL.weights, shape)
        predictions[test] = design[test] @ fit_head(design[train], response[train])
        per_fold.append(shape)

    rmse = float(np.sqrt(np.mean((predictions - response)[INTERIOR] ** 2)))
    interior_rows = np.flatnonzero(INTERIOR)
    ranked = interior_rows[np.argsort(predictions[interior_rows])]
    observed_best = int(interior_rows[np.argmin(response[interior_rows])])
    regret = float(response[ranked[0]] - response[observed_best])

    shape = select_shape(response, np.arange(len(response)), protocol)
    design = design_matrix(PANEL.weights, shape)
    coefficients = fit_head(design, response)

    axis = np.linspace(0.0, 1.0, SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = grid_0.ravel(), grid_1.ravel()
    predicted = design_matrix(wsd.grid_weights(flat_0, flat_1), shape) @ coefficients
    tied_axis = np.linspace(0.0, 1.0, SURFACE_GRID * SURFACE_GRID)
    tied = design_matrix(wsd.grid_weights(tied_axis, tied_axis), shape) @ coefficients

    raw = int(np.argmin(predicted))
    margin = harness.BOUNDARY_MARGIN
    inside = np.flatnonzero((flat_0 > margin) & (flat_1 > margin) & (flat_0 < 1 - margin) & (flat_1 < 1 - margin))
    clipped = int(inside[np.argmin(predicted[inside])])
    target_0, target_1 = PANEL.phase_0[observed_best, 1], PANEL.phase_1[observed_best, 1]

    result = {
        "target": name,
        "protocol": protocol,
        "rmse": rmse,
        "regret": regret,
        "shape": shape,
        "per_fold": per_fold,
        "raw_optimum": (float(flat_0[raw]), float(flat_1[raw])),
        "raw_distance": float(np.hypot(flat_0[raw] - target_0, flat_1[raw] - target_1)),
        "raw_gain": float(tied.min() - predicted[raw]),
        "interior_optimum": (float(flat_0[clipped]), float(flat_1[clipped])),
        "interior_distance": float(np.hypot(flat_0[clipped] - target_0, flat_1[clipped] - target_1)),
        "interior_gain": float(tied.min() - predicted[clipped]),
    }
    if verbose:
        announce(result, coefficients)
    return result


def announce(result: dict, coefficients: np.ndarray) -> None:
    slow, damage_horizon, gamma_broad, gamma_code, offset, tau = result["shape"]
    names = ("slow", "damage_horizon", "gamma_broad", "gamma_code", "offset", "tau")
    at_bound = [n for n, v in zip(names, result["shape"], strict=True) if v in BOUNDS[n]]
    print(f"\n=== {result['target']}   protocol {result['protocol']} ===")
    print(
        f"  slow {slow}  damage_horizon {damage_horizon}  gamma_broad {gamma_broad}  gamma_code {gamma_code}"
        f"  offset {offset}  tau {tau}"
    )
    print(f"  per-token late:early ratio implied by slow = {per_token_ratio(slow):.2f}x")
    print(f"  parameters still on a grid bound: {at_bound if at_bound else 'none'}")
    print(f"  per-outer-fold slow horizons: {[s[0] for s in result['per_fold']]}")
    print(f"  amplitudes {np.array2string(coefficients, precision=5)}")
    ok = result["rmse"] <= RPL_INTERIOR_RMSE * 1.05
    print(
        f"  NESTED interior OOF RMSE {result['rmse']:.6f}   RPL {RPL_INTERIOR_RMSE}, +5% gate"
        f" {RPL_INTERIOR_RMSE * 1.05:.6f}   {'PASS' if ok else 'FAIL'}"
    )
    print(
        f"  Regret@1 {result['regret']:.6f}   gate <= {RPL_REGRET_LIMIT}"
        f"   {'PASS' if result['regret'] <= RPL_REGRET_LIMIT else 'FAIL'}"
    )
    for kind in ("raw", "interior"):
        print(
            f"  {kind:8s} optimum ({result[kind + '_optimum'][0]:.3f},{result[kind + '_optimum'][1]:.3f})"
            f"  distance {result[kind + '_distance']:.4f}  gain {result[kind + '_gain']:+.6f}"
        )
    passes = result["raw_distance"] <= OPTIMUM_DISTANCE_LIMIT
    print(f"  optimum-distance gate on the RAW optimum <= {OPTIMUM_DISTANCE_LIMIT}: {'PASS' if passes else 'FAIL'}")
    error = abs(result["raw_gain"] - harness.OBSERVED_WSD_GAIN)
    print(f"  [development diagnostic, not a validated gate] gain error {error:.6f} vs {harness.WSD_GAIN_ERROR_LIMIT}")


ONES = np.ones(len(PANEL.y))


def main() -> None:
    print(f"panel {len(PANEL.y)} rows, {int(INTERIOR.sum())} interior, {len(SHAPES)} shapes, nested {N_FOLDS}x{N_FOLDS}")
    primary = TARGETS.values[:, TARGETS.names.index(harness.PRIMARY_TARGET)]
    for protocol in ("random", "blocked"):
        evaluate(protocol, harness.PRIMARY_TARGET, primary)
    if len(sys.argv) > 1 and sys.argv[1] == "controls":
        print("\n=== transfer and negative controls (random protocol) ===")
        for name in TARGETS.names:
            if name == harness.PRIMARY_TARGET:
                continue
            row = evaluate("random", name, TARGETS.values[:, TARGETS.names.index(name)], verbose=False)
            flag = "" if row["raw_gain"] <= CONTROL_GAIN_LIMIT else "  <-- above control limit"
            print(f"  {name:68s} rmse {row['rmse']:.6f}  gain {row['raw_gain']:+.6f}{flag}")


if __name__ == "__main__":
    main()
