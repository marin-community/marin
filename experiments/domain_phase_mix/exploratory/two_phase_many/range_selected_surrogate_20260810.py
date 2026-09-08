# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""The range-selected surrogate and its validation (GEN-027).

Two changes to the committed bucket-general model, both forced by the extrapolation defect that made
random parameters from its own search box predict up to 23x worse than a constant, and made one ordinary
300M component fit return an RMSE of 562 BPB.

SATURATING READOUT. ``1 / (1 + (E / E0) ** gamma)`` replaces ``(E + offset) ** -gamma``. The power law is
unbounded as exposure goes to zero and is regularised only by the offset, which caps it at
``offset ** -gamma``; that reaches 1e10 on buckets with exactly zero weight, of which the 300M panel has
41 and WSD80 has 2. The saturating form lies in [0,1] by construction and keeps the same monotone
decreasing shape over the sampled range. It belongs to the Hill family already used for damage, so the
model gains no new machinery.

RANGE-PENALISED SELECTION, which is the part that matters. Shrinking the per-bucket departures cannot fix
this: the ridge is fitted, so the inner CV lowers it to cancel any penalty multiplier -- measured
effective penalties are indistinguishable at multipliers 1 and 100. The criterion selects the same
effective shrinkage however the penalty is written, and it will never select one that bounds predictions,
because bounding predictions does not reduce fold error. So the CRITERION changes: selection pays a
penalty when predictions run outside the training target range, evaluated over EVERY row's design.
Scoring unlabelled rows' designs is not leakage -- a policy's mixture is known without consulting its
outcome -- and it is exactly what lets selection see extrapolation that fold error structurally cannot.

Deleting the departures block also fixes the extrapolation but costs 66 percent of 300M Uncheatable RMSE
against a 5 percent gate, so it is rejected; this keeps the block.

Usage: ``uv run python ... [wsd80|controls|components] [seeds]``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_general_surrogate_wsd80_20260810 as driver,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_wsd80_20260806 as wsd,
)

# Selected at 1 rather than 10: at 10 the 300M fit and arc_challenge are marginally better but WSD80
# falls from 11/12 to 10/12, so 10 buys fit with the optimum.
RANGE_WEIGHT = 1.0
CONTROL_GAIN_LIMIT = harness.WSD_NEGATIVE_GAIN_LIMIT
CODE_MARKERS = ("programing_languages", "github_")


def design(panel: model.Panel, shape: model.Shape) -> tuple[np.ndarray, np.ndarray, int]:
    """Family-pooled blocks plus shrunk per-bucket departures, with a bounded readout."""
    exponent = np.asarray(shape.readout_exponent, dtype=float)[panel.family_index]
    scale = np.asarray(shape.boundary_scale, dtype=float)[panel.exposure_stratum()]

    def readout(exposure: np.ndarray) -> np.ndarray:
        powered = (np.maximum(exposure, 0.0) / shape.offset) ** exponent
        return 1.0 / (1.0 + powered)

    near = readout(panel.exposure(shape.near_horizon))
    excess = np.maximum(panel.exposure(shape.damage_horizon) - 1.0, 0.0) / model.DAMAGE_KNEE
    powered = excess**shape.damage_exponent

    free = np.column_stack([np.ones(len(panel.weights)), model.family_sums(panel.weights[:, 1, :], panel.family_index)])
    constrained = np.column_stack(
        [
            model.family_sums(near, panel.family_index),
            model.family_sums(readout(panel.exposure(1.0)), panel.family_index),
            model.family_sums(np.exp(-panel.early_epochs() / scale), panel.family_index),
            model.family_sums(powered / (1.0 + powered), panel.family_index),
            near,
        ]
    )
    return free, constrained, 4 * panel.n_families


def bounds_for(panel: model.Panel) -> tuple:
    """`offset` is a SCALE in epochs for the saturating readout, not a floor, so it needs a wider box."""
    base = list(model.bounds(panel.n_families, panel.n_exposure_strata()))
    base[2] = (-2.0, 1.5)
    return tuple(base)


def range_excess(predictions: np.ndarray, observed: np.ndarray) -> float:
    """How far predictions run past the training targets, in units of the training spread."""
    low, high = float(np.min(observed)), float(np.max(observed))
    span = max(high - low, 1e-12)
    return (max(0.0, float(np.max(predictions)) - high) + max(0.0, low - float(np.min(predictions)))) / span


def inner_error(vector, panel, full, response, folds) -> float:
    """Fold error plus a penalty for leaving the target range, scored on EVERY row's design.

    Residuals are NOT restricted to interior rows, unlike the committed WSD80 driver. That restriction
    was tried here and is badly harmful in this single-target setting: seed 0 falls from 4/4 to 1/4, with
    the optimum collapsing from (0.060,0.485) to (0.000,0.525). Boundary rows evidently carry information
    selection needs, even though the gates themselves are scored on interior rows only.
    """
    shape, ridge = model.unpack(vector, panel.n_families, panel.n_exposure_strata())
    free, constrained, pooled = design(panel, shape)
    if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
        return 1e6
    free_all, constrained_all, _ = design(full, shape)
    total = 0.0
    for train, test in folds:
        head, amplitudes = model.fit_head(free[train], constrained[train], response[train], ridge, pooled)
        residual = free[test] @ head + constrained[test] @ amplitudes - response[test]
        error = float(residual @ residual)
        total += error
        if RANGE_WEIGHT > 0:
            excess = range_excess(free_all @ head + constrained_all @ amplitudes, response[train])
            total += RANGE_WEIGHT * excess**2 * (error + 1e-12)
    return total


def select(panel, full, response, folds, seed: int) -> np.ndarray:
    return differential_evolution(
        inner_error,
        bounds_for(panel),
        args=(panel, full, response, folds),
        rng=np.random.default_rng(20260812 + seed),
        popsize=10,
        maxiter=50,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def wsd80_surface(shape, head, amplitudes):
    axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = grid_0.ravel(), grid_1.ravel()
    free, constrained, _ = design(driver.grid_panel(wsd.grid_weights(flat_0, flat_1)), shape)
    tied_axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID * driver.SURFACE_GRID // 4)
    tied_free, tied_constrained, _ = design(driver.grid_panel(wsd.grid_weights(tied_axis, tied_axis)), shape)
    surface = free @ head + constrained @ amplitudes
    return flat_0, flat_1, surface, float((tied_free @ head + tied_constrained @ amplitudes).min())


def wsd80_row(response: np.ndarray, seed: int) -> dict:
    swarm = driver.SWARM
    rows = np.arange(len(response))
    predictions = np.empty_like(response)
    for train, test in harness.wsd80_folds("random", swarm.weights, rows, driver.reference.N_FOLDS, seed):
        sub = model.Panel(swarm.weights[train], swarm.epochs_early, swarm.epochs_late, swarm.family_index)
        inner = harness.wsd80_folds("random", sub.weights, np.arange(len(train)), driver.reference.N_INNER_FOLDS, seed)
        shape, ridge = model.unpack(
            select(sub, swarm, response[train], inner, seed), swarm.n_families, swarm.n_exposure_strata()
        )
        free, constrained, pooled = design(swarm, shape)
        head, amplitudes = model.fit_head(free[train], constrained[train], response[train], ridge, pooled)
        predictions[test] = free[test] @ head + constrained[test] @ amplitudes

    interior_rows = np.flatnonzero(driver.INTERIOR)
    observed_best = int(interior_rows[np.argmin(response[interior_rows])])
    ranked = interior_rows[np.argsort(predictions[interior_rows])]

    inner_full = harness.wsd80_folds("random", swarm.weights, rows, driver.reference.N_INNER_FOLDS, seed)
    shape, ridge = model.unpack(
        select(swarm, swarm, response, inner_full, seed), swarm.n_families, swarm.n_exposure_strata()
    )
    free, constrained, pooled = design(swarm, shape)
    head, amplitudes = model.fit_head(free, constrained, response, ridge, pooled)
    flat_0, flat_1, surface, tied_best = wsd80_surface(shape, head, amplitudes)
    cell = int(np.argmin(surface))

    return {
        "rmse": float(np.sqrt(np.mean((predictions - response)[driver.INTERIOR] ** 2))),
        "regret_1": float(response[ranked[0]] - response[observed_best]),
        "distance": float(
            np.hypot(
                flat_0[cell] - driver.PANEL.phase_0[observed_best, 1],
                flat_1[cell] - driver.PANEL.phase_1[observed_best, 1],
            )
        ),
        "gain_error": abs(tied_best - surface.min() - harness.OBSERVED_WSD_GAIN),
        "gain": tied_best - surface.min(),
        "optimum": (float(flat_0[cell]), float(flat_1[cell])),
    }


def run_wsd80(seeds: list[int]) -> None:
    primary = driver.TARGETS[:, driver.reference.TARGETS.names.index(harness.PRIMARY_TARGET)]
    print(f"GEN-027 WSD80, range weight {RANGE_WEIGHT}, {len(seeds)} seeds")
    print(f"gates RMSE<={driver.RMSE_LIMIT} R@1<={driver.REGRET_LIMIT} dist<={driver.DISTANCE_LIMIT}")
    print(f"      |gain err|<={driver.GAIN_ERROR_LIMIT};  committed model scores 40/44 over 11 seeds\n")
    tally = {"rmse": 0, "regret_1": 0, "distance": 0, "gain_error": 0}
    for seed in seeds:
        row = wsd80_row(primary, seed)
        checks = {
            "rmse": row["rmse"] <= driver.RMSE_LIMIT,
            "regret_1": row["regret_1"] <= driver.REGRET_LIMIT,
            "distance": row["distance"] <= driver.DISTANCE_LIMIT,
            "gain_error": row["gain_error"] <= driver.GAIN_ERROR_LIMIT,
        }
        for key, ok in checks.items():
            tally[key] += ok
        body = "  ".join(f"{k} {row[k]:.6f}{'P' if v else 'F'}" for k, v in checks.items())
        print(
            f"  seed {seed}: {body}  [{sum(checks.values())}/4]"
            f"  optimum ({row['optimum'][0]:.3f},{row['optimum'][1]:.3f})",
            flush=True,
        )
    print(f"\n{sum(tally.values())}/{4 * len(seeds)}: " + "  ".join(f"{k} {v}/{len(seeds)}" for k, v in tally.items()))


def run_controls(seeds: list[int]) -> None:
    names = driver.reference.TARGETS.names
    print(f"GEN-027 WSD80 controls, range weight {RANGE_WEIGHT}, seeds {seeds}")
    print(f"gate: predicted phase gain <= {CONTROL_GAIN_LIMIT} on every broad-text target")
    print("committed model holds 23/26 on all five seeds\n")
    held = 0
    checked = 0
    for name in names:
        response = driver.TARGETS[:, names.index(name)]
        gains = np.array([wsd80_row(response, seed)["gain"] for seed in seeds])
        is_code = any(marker in name for marker in CODE_MARKERS)
        if not is_code:
            checked += 1
            failed = int((gains > CONTROL_GAIN_LIMIT).sum())
            held += failed == 0
            flag = f"   <-- {failed}/{len(seeds)} above limit" if failed else ""
        else:
            flag = ""
        print(
            f" {'+' if is_code else ' '}{name:62s} worst {gains.max():+.6f}  "
            f"[{' '.join(f'{g:+.6f}' for g in gains)}]{flag}",
            flush=True,
        )
    print(f"\n{held}/{checked} negative controls hold on ALL {len(seeds)} seeds")


def run_components(seed: int) -> None:
    data = packet.load_300m("uncheatable")
    panel = model.Panel(data.weights, data.c0, data.c1, data.family_index)
    uncheatable = sorted(
        c
        for c in data.frame.columns
        if c.startswith("eval_uncheatable_eval_") and c.endswith("_bpb") and c != "eval_uncheatable_eval_bpb"
    )
    table9 = sorted(c for c in data.frame.columns if c.startswith("olmo_base") and data.frame[c].notna().sum() >= 280)
    print(f"GEN-027 300M components, range weight {RANGE_WEIGHT}, seed {seed}")
    print(f"  {len(uncheatable)} Uncheatable, {len(table9)} Table-9; ratio >= 1 means worse than the mean")
    print("  committed model: 4/1/5 regressions on seeds 0/1/2, with arc_challenge at 6586 on seed 2\n")

    regressions = []
    for label, columns, strip in (
        ("UNCHEATABLE", uncheatable, "eval_uncheatable_eval_"),
        ("TABLE-9", table9, "olmo_base_"),
    ):
        print(label)
        for column in columns:
            rows = np.flatnonzero(data.frame[column].notna().to_numpy())
            if len(rows) < 100:
                continue
            y = data.frame[column].to_numpy(dtype=float)[rows]
            here = model.Panel(panel.weights[rows], panel.epochs_early, panel.epochs_late, panel.family_index)
            frame = data.frame.iloc[rows].reset_index(drop=True)
            predictions = np.empty(len(rows))
            baseline = np.empty(len(rows))
            for train, test in packet.grouped_folds(frame, seed, 3):
                inner = packet.grouped_folds(frame.iloc[train].reset_index(drop=True), seed, 3)
                sub = model.Panel(here.weights[train], here.epochs_early, here.epochs_late, here.family_index)
                shape, ridge = model.unpack(
                    select(sub, here, y[train], inner, seed), here.n_families, here.n_exposure_strata()
                )
                free, constrained, pooled = design(here, shape)
                head, amplitudes = model.fit_head(free[train], constrained[train], y[train], ridge, pooled)
                predictions[test] = free[test] @ head + constrained[test] @ amplitudes
                baseline[test] = y[train].mean()
            rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))
            base = float(np.sqrt(np.mean((baseline - y) ** 2)))
            ratio = rmse / base
            flag = ""
            if ratio >= 1.0:
                regressions.append((column, ratio))
                flag = "   <-- WORSE THAN THE MEAN"
            print(
                f"   {column.replace(strip, '')[:36]:36s} n={len(rows):3d}  RMSE {rmse:.6f}  "
                f"baseline {base:.6f}  ratio {ratio:.3f}{flag}",
                flush=True,
            )
        print()
    print(f"components worse than an out-of-fold intercept: {len(regressions)}")
    for column, ratio in sorted(regressions, key=lambda item: -item[1]):
        print(f"   {column}  ratio {ratio:.3f}")


def main() -> None:
    argv = sys.argv[1:]
    mode = argv[0] if argv and argv[0] in ("wsd80", "controls", "components") else "wsd80"
    seeds = [int(s) for s in argv[1:] if s.isdigit()]
    if mode == "components":
        run_components(seeds[0] if seeds else 0)
    elif mode == "controls":
        run_controls(seeds or [0, 1, 2, 3, 4])
    else:
        run_wsd80(seeds or list(range(11)))


if __name__ == "__main__":
    main()
