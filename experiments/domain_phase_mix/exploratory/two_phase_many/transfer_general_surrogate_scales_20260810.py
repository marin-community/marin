# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""GEN-002 across model scales: does the shared nonlinear form transfer? (GEN-011)

The round charter asks for other swarms as transfer checks, and the strongest one available is a change
of MODEL SCALE at fixed swarm design. The 60M, 300M and delphi_3e18 panels use the same thirty-nine
buckets, the same three families, and the same epoch geometry (4.80 to 1723.89 epochs per unit weight);
only the trained model differs. Every ordered pair of scales is run, so each scale is both a source and a
target of transferred parameters. A surrogate whose nonlinear parameters describe training dynamics rather
than one panel's idiosyncrasies should carry those parameters across that change with little loss.

Three fits per scale and target, all on identical grouped folds:

  own      theta selected in-fold at this scale        -- the form works here at all
  frozen   theta selected at the OTHER scale, heads refit here, NOTHING else refit
  floor    intercept only                              -- the variance any fit must beat

`frozen` is the real test and it is deliberately harsh: the transferred theta never sees a single row of
the target scale, not even through fold selection. If `frozen` lands near `own`, the nonlinear parameters
are a property of the training process. If it lands near `floor`, they were panel-fitting.

Heldout splits are loaded through ``swarm39_harness.load_scale``, which asserts the sealed series is
absent, so the sealed panel stays sealed.

Usage: ``uv run python ... [seeds]``, default 0-2.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    swarm39_harness_20260725 as swarm,
)

SCALES = ("60m", "300m", "delphi_3e18")
TARGETS = ("uncheatable_bpb", "table9_macro_bpb")
N_FOLDS = 3
N_INNER_FOLDS = 3


def to_model_panel(panel) -> model.Panel:
    return model.Panel(
        weights=np.stack([panel.phase0, panel.phase1], axis=1),
        epochs_early=panel.c0,
        epochs_late=panel.c1,
        family_index=panel.family_index,
    )


def subset(panel: model.Panel, rows: np.ndarray) -> model.Panel:
    return model.Panel(panel.weights[rows], panel.epochs_early, panel.epochs_late, panel.family_index)


def inner_error(vector: np.ndarray, panel: model.Panel, response: np.ndarray, folds, n_families, n_strata) -> float:
    shape, ridge = model.unpack(vector, n_families, n_strata)
    free, constrained = model.design(panel, shape)
    if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
        return 1e6
    total = 0.0
    for train, test in folds:
        b, a = model.fit_head(free[train], constrained[train], response[train], ridge, model.pooled_width(panel))
        residual = free[test] @ b + constrained[test] @ a - response[test]
        total += float(residual @ residual)
    return total


def select(panel: model.Panel, response: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    n_families, n_strata = panel.n_families, panel.n_exposure_strata()
    folds = grouped_folds(groups, N_INNER_FOLDS, seed + 1000)
    return differential_evolution(
        inner_error,
        model.bounds(n_families, n_strata),
        args=(panel, response, folds, n_families, n_strata),
        rng=np.random.default_rng(20260810 + seed),
        popsize=12,
        maxiter=60,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every row sharing a group id lands in one fold; correlated labels cannot straddle the split."""
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(unique))
    assignment = {unique[i]: k % n_folds for k, i in enumerate(order)}
    fold_of = np.array([assignment[g] for g in groups])
    return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(n_folds)]


def policy_keys(raw) -> np.ndarray:
    """Exact policy identity: the concatenated phase-0 and phase-1 weight vector, rounded."""
    joined = np.round(np.concatenate([raw.phase0, raw.phase1], axis=1), 9)
    return np.array([hash(tuple(row)) for row in joined])


def oof_predictions(panel, response, vector, folds, n_families, n_strata) -> np.ndarray:
    shape, ridge = model.unpack(vector, n_families, n_strata)
    free, constrained = model.design(panel, shape)
    predictions = np.empty_like(response)
    for train, test in folds:
        b, a = model.fit_head(free[train], constrained[train], response[train], ridge, model.pooled_width(panel))
        predictions[test] = free[test] @ b + constrained[test] @ a
    return predictions


def oof_rmse(panel: model.Panel, response: np.ndarray, vector: np.ndarray, folds, n_families, n_strata) -> float:
    """Score a FIXED theta out of fold; only the head is refit per fold."""
    shape, ridge = model.unpack(vector, n_families, n_strata)
    free, constrained = model.design(panel, shape)
    predictions = np.empty_like(response)
    for train, test in folds:
        b, a = model.fit_head(free[train], constrained[train], response[train], ridge, model.pooled_width(panel))
        predictions[test] = free[test] @ b + constrained[test] @ a
    return float(np.sqrt(np.mean((predictions - response) ** 2)))


def nested_own_rmse(panel, response, groups, folds, n_families, n_strata, seed: int) -> float:
    """Own-scale arm with theta selected INSIDE each outer fold.

    The first version of this driver selected own-scale theta once on ALL target rows and then scored it
    across the outer folds. That leaks the test rows into the own arm while the frozen arm stays blind,
    so the two arms were not comparable -- and the leak flattered `own`, which made transfer look WORSE
    than it is. Selection now happens on training rows only.
    """
    predictions = np.empty_like(response)
    for train, test in folds:
        sub = subset(panel, train)
        vector = select(sub, response[train], groups[train], seed)
        shape, ridge = model.unpack(vector, n_families, n_strata)
        free, constrained = model.design(panel, shape)
        b, a = model.fit_head(free[train], constrained[train], response[train], ridge, model.pooled_width(panel))
        predictions[test] = free[test] @ b + constrained[test] @ a
    return float(np.sqrt(np.mean((predictions - response) ** 2)))


def floor_rmse(response: np.ndarray, folds) -> float:
    predictions = np.empty_like(response)
    for train, test in folds:
        predictions[test] = response[train].mean()
    return float(np.sqrt(np.mean((predictions - response) ** 2)))


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    loaded = {}
    for scale in SCALES:
        fit_panel, _ = swarm.load_scale(scale)
        loaded[scale] = (fit_panel, to_model_panel(fit_panel))

    print("GEN-011: cross-scale transfer of the GEN-002 nonlinear form")
    print(f"scales {SCALES}, targets {TARGETS}, {N_FOLDS} grouped outer folds, seeds {seeds}")
    for scale in SCALES:
        raw, mp = loaded[scale]
        print(f"  {scale}: {len(raw.phase0)} rows, {mp.n_families} families, {mp.n_exposure_strata()} strata")
    print()

    for target in TARGETS:
        print(f"=== {target} ===")
        for scale in SCALES:
            raw, panel = loaded[scale]
            response = raw.targets[target]
            keep = np.flatnonzero(np.isfinite(response))
            here = subset(panel, keep)
            y = response[keep]
            groups = raw.group[keep]
            n_families, n_strata = here.n_families, here.n_exposure_strata()

            for seed in seeds:
                folds = grouped_folds(groups, N_FOLDS, seed)
                own_rmse = nested_own_rmse(here, y, groups, folds, n_families, n_strata, seed)
                base = floor_rmse(y, folds)
                for other in SCALES:
                    if other == scale:
                        continue
                    other_raw, other_panel = loaded[other]
                    other_response = other_raw.targets[target]
                    other_keep = np.flatnonzero(np.isfinite(other_response))
                    frozen = select(
                        subset(other_panel, other_keep),
                        other_response[other_keep],
                        other_raw.group[other_keep],
                        seed,
                    )
                    frozen_predictions = oof_predictions(here, y, frozen, folds, n_families, n_strata)
                    frozen_rmse = float(np.sqrt(np.mean((frozen_predictions - y) ** 2)))
                    # Does theta transfer, or does the shared panel design do the work? The scales reuse
                    # most policy coordinates exactly, so score the target rows whose exact policy is
                    # ABSENT from the source scale separately. Those are the only rows where the source
                    # could not have been fitted at the same coordinates.
                    source_keys = set(policy_keys(other_raw).tolist())
                    disjoint = np.array([k not in source_keys for k in policy_keys(raw)[keep]])
                    disjoint_rmse = (
                        float(np.sqrt(np.mean((frozen_predictions - y)[disjoint] ** 2)))
                        if disjoint.sum() >= 10
                        else float("nan")
                    )
                    # Guard the ratio: if the model barely beats the intercept, "variance recovered" is a
                    # ratio of two small numbers and says nothing. Reported as nan rather than a big number.
                    # Share of explainable MSE recovered. The earlier version differenced RMSE, which is
                    # not a variance share; MSE is the additive quantity.
                    headroom = base**2 - own_rmse**2
                    recovered = (base**2 - frozen_rmse**2) / headroom if headroom > 0.1 * base**2 else float("nan")
                    print(
                        f" {scale:11s} seed {seed}: own {own_rmse:.6f}   frozen<-{other:11s} {frozen_rmse:.6f}"
                        f"   floor {base:.6f}   penalty {frozen_rmse / own_rmse:.3f}x"
                        f"   MSE recovered {recovered:.3f}"
                        f"   frozen on {int(disjoint.sum()):3d} source-absent rows {disjoint_rmse:.6f}"
                    )
        print()


if __name__ == "__main__":
    main()
