# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Audits that must pass before WSD80-SUR-095 is fitted to anything.

Four questions, in the order that matters. Does the closed form solve the differential equation it
claims to solve? Does the forgetting rate collapse to the refuted single-index null exactly, and is that
null actually unable to beat the tied class? Can a positive forgetting rate produce a two-phase gain at
all? And can a grid search recover a known shape from noisy data, or is the family unidentified before
any real panel is involved?

Every check is against an independent computation: numerical integration for the state, brute-force
search over the tied class for the impossibility claim, simulation for recovery.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    acquisition_forgetting_state_20260807 as afs,
)

GEOMETRY = afs.Geometry(epochs_per_unit_weight=np.array([1.0, 26.457868]), phase_1_fraction=0.203)
RNG = np.random.default_rng(20260807)


def two_phase(share_0: np.ndarray, share_1: np.ndarray) -> np.ndarray:
    """Lift second-bucket shares into the panel's (row, phase, bucket) weight array."""
    return np.stack(
        [np.column_stack([1.0 - share_0, share_0]), np.column_stack([1.0 - share_1, share_1])],
        axis=1,
    )


def integrated_state(share_0: float, share_1: float, shape: afs.Shape) -> np.ndarray:
    """The same state by numerical integration of the differential equation, for cross-checking."""
    boundary = GEOMETRY.phase_0_fraction
    rates = afs.delivery_rates(two_phase(np.array([share_0]), np.array([share_1])), GEOMETRY)
    rho = shape.rho

    def derivative(time, state):
        rate = rates[0][0] if time < boundary else rates[1][0]
        return rho * rate * (1.0 - state) - shape.forgetting[0] * state

    solved = solve_ivp(
        derivative,
        (0.0, 1.0),
        np.zeros(GEOMETRY.n_domains),
        t_eval=[1.0],
        rtol=1e-11,
        atol=1e-13,
        max_step=1e-3,
    )
    return solved.y[:, -1]


def integrated_overexposure(share_0: float, share_1: float, forgetting: float) -> np.ndarray:
    """Discounted repetition by direct quadrature on a fine grid, for cross-checking the crossing time."""
    boundary = GEOMETRY.phase_0_fraction
    grid = np.linspace(0.0, 1.0, 4_000_001)
    rate_0, rate_1 = afs.delivery_rates(two_phase(np.array([share_0]), np.array([share_1])), GEOMETRY)
    rate = np.where(grid[:, None] < boundary, rate_0[0], rate_1[0])
    cumulative = np.cumsum(np.vstack([np.zeros(GEOMETRY.n_domains), rate[:-1] * np.diff(grid)[:, None]]), axis=0)
    weight = np.exp(-forgetting * (1.0 - grid))[:, None] * (cumulative > 1.0)
    return np.trapezoid(rate * weight, grid, axis=0)


def audit_closed_form() -> None:
    print("1. closed form against numerical integration")
    worst_state, worst_damage = 0.0, 0.0
    for forgetting in (0.0, 0.5, 3.0, 12.0):
        for share_0, share_1 in ((0.1, 0.5), (0.6, 0.05), (0.35, 0.35), (0.9, 0.9), (0.02, 0.98)):
            shape = afs.Shape(rho=0.7, forgetting=(forgetting,), exponent=(0.5, 0.2), offset=0.05, damage_exponent=1.0)
            weights = two_phase(np.array([share_0]), np.array([share_1]))
            closed = afs.held_fraction(weights, GEOMETRY, shape.rho, shape.forgetting[0])[0]
            worst_state = max(worst_state, float(np.max(np.abs(closed - integrated_state(share_0, share_1, shape)))))
            closed_damage = afs.discounted_overexposure(weights, GEOMETRY, forgetting)[0]
            reference = integrated_overexposure(share_0, share_1, forgetting)
            worst_damage = max(worst_damage, float(np.max(np.abs(closed_damage - reference))))
    print(f"   max state error   {worst_state:.3e}   {'PASS' if worst_state < 1e-7 else 'FAIL'}")
    print(f"   max damage error  {worst_damage:.3e}   {'PASS' if worst_damage < 1e-4 else 'FAIL'}")


def audit_nested_null() -> None:
    print("2. forgetting rate zero is the single-index null, exactly")
    share_0 = RNG.uniform(0.0, 1.0, 400)
    share_1 = RNG.uniform(0.0, 1.0, 400)
    weights = two_phase(share_0, share_1)
    shape = afs.Shape(rho=0.7, forgetting=(0.0,), exponent=(0.5, 0.2), offset=0.05, damage_exponent=1.0)

    rate_0, rate_1 = afs.delivery_rates(weights, GEOMETRY)
    epochs = rate_0 * GEOMETRY.phase_0_fraction + rate_1 * GEOMETRY.phase_1_fraction
    state_error = np.max(np.abs(afs.held_fraction(weights, GEOMETRY, shape.rho, 0.0) - -np.expm1(-shape.rho * epochs)))
    damage_error = np.max(np.abs(afs.discounted_overexposure(weights, GEOMETRY, 0.0) - np.maximum(epochs - 1.0, 0.0)))
    retained_error = np.max(np.abs(afs.retained_epochs(weights, GEOMETRY, shape.rho, shape.forgetting[0]) - epochs))
    print(f"   state vs 1-exp(-rho * total epochs)  {state_error:.3e}   {'PASS' if state_error < 1e-12 else 'FAIL'}")
    print(
        f"   retained epochs vs total epochs      {retained_error:.3e}   {'PASS' if retained_error < 1e-12 else 'FAIL'}"
    )
    print(f"   damage vs max(0, epochs - 1)         {damage_error:.3e}   {'PASS' if damage_error < 1e-12 else 'FAIL'}")

    # The state depends smoothly on the rate, so the drift away from the null must fall linearly with
    # it. A fixed tolerance would only be testing which rate happened to be substituted.
    base = afs.held_fraction(weights, GEOMETRY, shape.rho, shape.forgetting[0])
    slopes = [
        np.max(np.abs(afs.held_fraction(weights, GEOMETRY, 0.7, rate) - base)) / rate for rate in (1e-8, 1e-6, 1e-4)
    ]
    # Residual spread across four decades of rate is the second-order term, so the bar is that the
    # leading behaviour is linear, not that the curvature is absent.
    spread = float(np.ptp(slopes) / np.mean(slopes))
    print(f"   drift per unit rate {np.mean(slopes):.4f}, spread {spread:.2e}   {'PASS' if spread < 1e-3 else 'FAIL'}")


def audit_impossibility() -> None:
    print("3. what the null can and cannot do")
    axis = np.linspace(0.0, 1.0, 601)
    coefficients = np.array([1.2, 0.35, 0.90, 0.25])
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")

    # The theorem itself, checked as an identity rather than through a grid. Under the null every
    # policy acts through the phase-averaged share alone, so a two-phase policy and the tied policy at
    # its index must produce the SAME design row, hence the same predicted loss whatever the head is.
    share_0, share_1 = RNG.uniform(0.0, 1.0, 500), RNG.uniform(0.0, 1.0, 500)
    index = share_0 * GEOMETRY.phase_0_fraction + share_1 * GEOMETRY.phase_1_fraction
    null = afs.Shape(rho=0.8, forgetting=(0.0,), exponent=(0.5, 0.2), offset=0.05, damage_exponent=2.0)
    identity = np.max(
        np.abs(
            afs.design_matrix(two_phase(share_0, share_1), GEOMETRY, null)
            - afs.design_matrix(two_phase(index, index), GEOMETRY, null)
        )
    )
    print(
        f"   null: two-phase row equals tied row at its index  {identity:.3e}  {'PASS' if identity < 1e-10 else 'FAIL'}"
    )

    # The tied class must be searched at least as finely as the achievable index values, or the
    # reported gain is grid resolution rather than mechanism.
    tied_axis = np.linspace(0.0, 1.0, 601 * 601)
    for forgetting in (0.0, 1.0, 4.0, 10.0):
        shape = afs.Shape(rho=0.8, forgetting=(forgetting,), exponent=(0.5, 0.2), offset=0.05, damage_exponent=2.0)
        surface = afs.design_matrix(two_phase(grid_0.ravel(), grid_1.ravel()), GEOMETRY, shape) @ coefficients
        tied = afs.design_matrix(two_phase(tied_axis, tied_axis), GEOMETRY, shape) @ coefficients
        gain = float(tied.min() - surface.min())
        best = np.unravel_index(int(np.argmin(surface)), grid_0.shape)
        print(f"   forgetting {forgetting:5.1f}  gain {gain:+.6f}  optimum ({axis[best[0]]:.3f},{axis[best[1]]:.3f})")
        if forgetting == 0.0:
            assert gain < 1e-9, f"single-index null produced a two-phase gain of {gain}"


def audit_recovery() -> None:
    print("4. shape recovery from simulated data at panel scale and noise")
    truth = afs.Shape(rho=0.6, forgetting=(4.0,), exponent=(0.5, 0.2), offset=0.05, damage_exponent=2.0)
    coefficients = np.array([1.15, 0.40, 0.85, 0.20])
    share_0 = RNG.uniform(0.02, 0.98, 346)
    share_1 = RNG.uniform(0.02, 0.98, 346)
    weights = two_phase(share_0, share_1)
    clean = afs.design_matrix(weights, GEOMETRY, truth) @ coefficients

    rho_grid = (0.05, 0.1, 0.2, 0.35, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0)
    code_grid = (0.05, 0.1, 0.2, 0.3, 0.5)
    forgetting_grid = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0)
    for noise in (0.0, 0.002, 0.006):
        observed = clean + RNG.normal(0.0, noise, len(clean))
        scored = []
        for broad in rho_grid:
            for code in code_grid:
                for forgetting in forgetting_grid:
                    for exponent in (1.0, 2.0, 3.0):
                        shape = afs.Shape(
                            rho=broad,
                            forgetting=(forgetting,),
                            exponent=(0.5, code),
                            offset=0.05,
                            damage_exponent=exponent,
                        )
                        design = afs.design_matrix(weights, GEOMETRY, shape)
                        residual = design @ afs.solve(design, observed) - observed
                        scored.append((float(residual @ residual), (broad, code, forgetting, exponent)))
        found = min(scored)[1]
        exact = found == (truth.rho, truth.exponent[1], truth.forgetting[0], truth.damage_exponent)
        print(
            f"   noise {noise:.3f} BPB -> rho {found[0]:.2f}/{found[1]:.2f} forgetting {found[2]:.2f} "
            f"tau {found[3]:.0f}   {'exact' if exact else 'moved'}"
        )


audit_closed_form()
audit_nested_null()
audit_impossibility()
audit_recovery()
