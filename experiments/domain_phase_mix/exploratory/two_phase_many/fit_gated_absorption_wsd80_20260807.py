# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""WSD80-SUR-102: gated absorption, from the two mechanisms the experimenters proposed.

The panel measures a two-phase gain that no single-index surrogate can produce, and separately measures
a penalty concentrated at EXACTLY zero early code share: runs with no code in the stable phase are 0.002
to 0.007 BPB worse than runs with 2.5 percent of it, at matched late share. Every additive surrogate in
this project puts its predicted optimum at that point, because a sum of independent exposure channels
has no way to say "this only matters when the domain is absent early".

Three channels carry the mechanisms, each nesting out at coefficient zero.

Absorption is gated. Late exposure counts only through a Hill gate on early exposure of the same domain,
``early^beta / (early^beta + kappa^beta)``, so late tokens are worth something only to the extent
groundwork exists to absorb them. This is multiplicative and is what no additive model can express. Both
domains get a gate: code needs code groundwork, and the general ability that code capability rides on
has to be built before the decay phase too.

Conflict is signed. A free-sign channel in the decay-phase off-domain share lets heavy late broad text
RAISE code BPB. Every previous form forbade this structurally, because non-negative amplitudes on
``(exposure + offset)^-gamma`` make more broad text unconditionally helpful. NOTE that this coefficient's
sign is NOT robustly identified across column sets, only across folds at a fixed column set, so the
channel is carried for the fit it buys and not as evidence for the conflict hypothesis.

Repetition harm keeps its own memory horizon. A phase-split version was tried, charging stable-phase and
decay-phase excess separately, and the stable-phase amplitude fitted to exactly zero in every run, which
is the experimenters' prediction that early repeats cost little. It is not used here because it also cost
0.002 BPB of fit; the finding is recorded, the structure is not adopted.

Selection is nested and continuous. Nonlinear parameters are chosen by differential evolution on inner
splits of each outer fold's training rows and never see that fold's outcomes. Grid selection was used
earlier in this project and cost about 30 percent of the achievable error, besides producing one false
representability ceiling and one headline that moved when a bound was widened.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, nnls  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_wsd80_20260806 as wsd,
)

RPL_INTERIOR_RMSE = 0.007575
RPL_REGRET_LIMIT = 0.004842
OPTIMUM_DISTANCE_LIMIT = 0.05
GAIN_ERROR_LIMIT = harness.WSD_GAIN_ERROR_LIMIT
CONTROL_GAIN_LIMIT = 0.005
N_FOLDS = 3
# Inner folds are finer than outer ones because selection variance, not fit, is what moved the optimum
# across seeds at 3x3. This is a protocol choice about the estimator, made independently of any outcome.
N_INNER_FOLDS = 5
SURFACE_GRID = 801
BROAD, CODE = 0, 1

# phi_near, gamma_broad, gamma_code, log_offset, tau, phi_damage, kappa_code, beta, nu, log_eps, kappa_broad
BOUNDS = (
    (0.0, 1.0),
    (0.005, 2.0),
    (0.005, 1.5),
    (-5.0, -0.3),
    (0.2, 10.0),
    (0.0, 1.0),
    (0.02, 20.0),
    (0.3, 40.0),
    (0.01, 2.0),
    (-6.0, 1.4),
    (0.02, 5.0),
)

PANEL, TARGETS = wsd.load_targets()
GEOMETRY = wsd.geometry()
EPOCHS = GEOMETRY.c0 + GEOMETRY.c1
INTERIOR = wsd.interior_mask(PANEL)


def exposure(weights: np.ndarray, domain: int, horizon: float) -> np.ndarray:
    """Epochs of `domain` as seen by a channel weighting the decay phase by `horizon`."""
    return EPOCHS[domain] * ((1.0 - horizon) * weights[:, 0, domain] + horizon * weights[:, 1, domain])


def absorbed(weights: np.ndarray, domain: int, gate_scale: float, sharpness: float) -> np.ndarray:
    """Exposure a run can actually use: late tokens discounted by how little groundwork preceded them."""
    early = EPOCHS[domain] * weights[:, 0, domain]
    gate = early**sharpness / (early**sharpness + gate_scale**sharpness)
    return EPOCHS[domain] * (weights[:, 0, domain] + weights[:, 1, domain] * gate)


def design(weights: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Free columns (intercept and signed conflict) and sign-constrained columns."""
    near, gamma_broad, gamma_code, log_offset, tau, damage_horizon, kappa, beta, nu, log_eps, kappa_b = shape
    offset, eps = 10.0**log_offset, 10.0**log_eps
    free = np.column_stack([np.ones(len(weights)), weights[:, 1, BROAD]])
    constrained = np.column_stack(
        [
            (exposure(weights, CODE, near) + offset) ** -gamma_code,
            (exposure(weights, BROAD, near) + offset) ** -gamma_broad,
            (exposure(weights, CODE, 1.0) + offset) ** -gamma_code,
            (exposure(weights, BROAD, 1.0) + offset) ** -gamma_broad,
            (absorbed(weights, CODE, kappa, beta) + offset) ** -gamma_code,
            (absorbed(weights, BROAD, kappa_b, beta) + offset) ** -gamma_broad,
            (exposure(weights, CODE, 0.0) + eps) ** -nu,
            np.maximum(exposure(weights, CODE, damage_horizon) - 1.0, 0.0) ** tau,
        ]
    )
    return free, constrained


def fit_head(free: np.ndarray, constrained: np.ndarray, response: np.ndarray):
    """Exact partitioned solve: free columns unconstrained, the rest non-negative.

    Columns are normalised before the sign-constrained solve and the amplitudes unscaled after. This does
    not change the model -- non-negativity is invariant to positive rescaling -- but it is not optional.
    The repetition column is an excess raised to a fitted exponent, and over this panel's exposure range
    with the exponent free to 10 it spans fourteen orders of magnitude. Solving unscaled made the
    sign-constrained solver miss its iteration cap outright on some folds and pushed the saturation scale
    to absurd values on others.
    """
    basis, _ = np.linalg.qr(free)
    columns = constrained - basis @ (basis.T @ constrained)
    scale = np.maximum(np.linalg.norm(columns, axis=0), 1e-300)
    amplitudes, _ = nnls(columns / scale, response - basis @ (basis.T @ response), maxiter=20000)
    amplitudes = amplitudes / scale
    return np.linalg.lstsq(free, response - constrained @ amplitudes, rcond=None)[0], amplitudes


def select(response: np.ndarray, rows: np.ndarray, seed: int):
    """Choose nonlinear parameters on inner splits of `rows`; held-out rows are absent by construction."""
    folds = harness.wsd80_folds("random", PANEL.weights[rows], np.arange(len(rows)), N_INNER_FOLDS, seed)
    subset, subset_interior = response[rows], INTERIOR[rows]

    def inner_error(shape) -> float:
        free, constrained = design(PANEL.weights[rows], shape)
        if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
            return 1e3
        total = 0.0
        for train, test in folds:
            b, a = fit_head(free[train], constrained[train], subset[train])
            residual = free[test] @ b + constrained[test] @ a - subset[test]
            # Score selection on the SAME rows the result is graded on. Fitting uses every row, including
            # the boundary rows that carry the cliff evidence, but selecting against an all-row objective
            # while reporting an interior metric is a mismatch review flagged on an earlier candidate.
            scored = residual[subset_interior[test]]
            if len(scored):
                total += float(scored @ scored)
        return total

    return differential_evolution(
        inner_error,
        BOUNDS,
        rng=np.random.default_rng(20260807),
        popsize=14,
        maxiter=120,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def evaluate(response: np.ndarray, seed: int) -> dict:
    outer = harness.wsd80_folds("random", PANEL.weights, np.arange(len(response)), N_FOLDS, seed)
    predictions = np.empty_like(response)
    for train, test in outer:
        shape = select(response, train, seed)
        free, constrained = design(PANEL.weights, shape)
        b, a = fit_head(free[train], constrained[train], response[train])
        predictions[test] = free[test] @ b + constrained[test] @ a

    interior_rows = np.flatnonzero(INTERIOR)
    observed_best = int(interior_rows[np.argmin(response[interior_rows])])
    ranked = interior_rows[np.argsort(predictions[interior_rows])]

    shape = select(response, np.arange(len(response)), seed)
    free, constrained = design(PANEL.weights, shape)
    b, a = fit_head(free, constrained, response)
    axis = np.linspace(0.0, 1.0, SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = grid_0.ravel(), grid_1.ravel()
    fg, cg = design(wsd.grid_weights(flat_0, flat_1), shape)
    surface = fg @ b + cg @ a
    tied_axis = np.linspace(0.0, 1.0, SURFACE_GRID * SURFACE_GRID // 4)
    ft, ct = design(wsd.grid_weights(tied_axis, tied_axis), shape)
    tied = ft @ b + ct @ a
    best = int(np.argmin(surface))

    return {
        "seed": seed,
        "rmse": float(np.sqrt(np.mean((predictions - response)[INTERIOR] ** 2))),
        "regret_1": float(response[ranked[0]] - response[observed_best]),
        "regret_5": float(response[ranked[:5]].min() - response[observed_best]),
        "optimum": (float(flat_0[best]), float(flat_1[best])),
        "distance": float(
            np.hypot(flat_0[best] - PANEL.phase_0[observed_best, 1], flat_1[best] - PANEL.phase_1[observed_best, 1])
        ),
        "gain": float(tied.min() - surface.min()),
        "shape": shape,
    }


def announce(row: dict) -> int:
    checks = {
        "RMSE": (row["rmse"], row["rmse"] <= RPL_INTERIOR_RMSE * 1.05),
        "Regret@1": (row["regret_1"], row["regret_1"] <= RPL_REGRET_LIMIT),
        "distance": (row["distance"], row["distance"] <= OPTIMUM_DISTANCE_LIMIT),
        "gain err": (
            abs(row["gain"] - harness.OBSERVED_WSD_GAIN),
            abs(row["gain"] - harness.OBSERVED_WSD_GAIN) <= GAIN_ERROR_LIMIT,
        ),
    }
    passed = sum(ok for _, ok in checks.values())
    body = "  ".join(f"{name} {value:.6f}{'P' if ok else 'F'}" for name, (value, ok) in checks.items())
    print(
        f"  seed {row['seed']}: {body}  [{passed}/4]  optimum ({row['optimum'][0]:.3f},{row['optimum'][1]:.3f})"
        f"  Regret@5 {row['regret_5']:.6f}"
    )
    return passed


def controls(seed: int) -> None:
    """Every other metric on the panel, fitted the same way, with the phase gain each one implies.

    The sharpest test this project has. Repaired RPL invents about 0.029 BPB of phase gain on C4 English
    and Falcon RefinedWeb, where the sampled optimum is tied and the observed gain is exactly zero. A
    surrogate that separates the families should predict near zero there while keeping real gain on code.
    """
    print(f"\ntransfer and negative controls, seed {seed}, gain limit {CONTROL_GAIN_LIMIT} on broad text")
    for name in TARGETS.names:
        row = evaluate(TARGETS.values[:, TARGETS.names.index(name)], seed)
        # The limit applies to broad text only. Code targets are POSITIVE controls and are expected to
        # carry real phase gain, so flagging them as violations would invert the test's meaning.
        code_target = any(k in name for k in ("programing_languages", "github_"))
        flag = "" if code_target or row["gain"] <= CONTROL_GAIN_LIMIT else "   <-- above control limit"
        marker = "+" if code_target else " "
        print(f" {marker}{name:66s} rmse {row['rmse']:.6f}  gain {row['gain']:+.6f}{flag}")


def main() -> None:
    if sys.argv[1:2] == ["controls"]:
        controls(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        return
    primary = TARGETS.values[:, TARGETS.names.index(harness.PRIMARY_TARGET)]
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4, 5]
    print(
        f"WSD80-SUR-102, {len(PANEL.y)} rows, {int(INTERIOR.sum())} interior,"
        f" nested {N_FOLDS} outer x {N_INNER_FOLDS} inner, continuous"
    )
    print(
        f"gates: RMSE<={RPL_INTERIOR_RMSE * 1.05:.6f}  Regret@1<={RPL_REGRET_LIMIT}"
        f"  distance<={OPTIMUM_DISTANCE_LIMIT}  |gain err|<={GAIN_ERROR_LIMIT}"
    )
    rows = [evaluate(primary, seed) for seed in seeds]
    total = sum(announce(row) for row in rows)
    print(f"\n  {total}/{4 * len(rows)} gate-passes over {len(rows)} seeds")
    optima = np.array([row["optimum"] for row in rows])
    print(
        f"  optimum across seeds: early {optima[:, 0].min():.3f}-{optima[:, 0].max():.3f}"
        f"  late {optima[:, 1].min():.3f}-{optima[:, 1].max():.3f}"
    )
    for name, limit in (("Regret@1", RPL_REGRET_LIMIT), ("distance", OPTIMUM_DISTANCE_LIMIT)):
        key = "regret_1" if name == "Regret@1" else "distance"
        values = np.array([row[key] for row in rows])
        print(
            f"  {name}: {int((values <= limit).sum())}/{len(rows)} seeds pass,"
            f" range {values.min():.6f}-{values.max():.6f}"
        )


if __name__ == "__main__":
    main()
