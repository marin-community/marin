# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""The 300M phase-sensitive gate for GEN-002, with a clustered bootstrap (GEN-012).

The frozen 300M gate asks for one phase-sensitive diagnostic improved beyond clustered-bootstrap
uncertainty without regressing the other target. Previous rounds reported pooled RMSE and pair gains as
point estimates, which cannot answer that: a difference of the size at stake here is well inside what
280 correlated correspondence groups can produce by chance.

Diagnostic: PAIRED GAP ERROR. Every asymmetric policy in the panel has an exact aggregate-matched tied
counterpart inside its correspondence group, so the difference between them removes the aggregate level
and leaves only the phase contrast. Root-mean-square error of the predicted gap against the observed gap
is therefore phase-sensitive in a way pooled RMSE is not -- a model can score well on pooled RMSE purely
by placing the aggregate level correctly and never getting a single phase contrast right.

Comparison: shared theta across both targets against theta selected independently per target, which is
the charter's MULTI-TARGET DIRECTION test. Both arms use identical grouped folds and identical heads, so
the ONLY difference is whether the nonlinear parameters are forced to serve both targets at once.

Uncertainty: bootstrap over correspondence keys, not rows. Resampling rows would break the pairing and
understate the spread, since the two members of a pair share a checkpoint and are strongly correlated.

Usage: ``uv run python ... [n_bootstrap] [seeds]``, default 2000 and 0-2.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as panel,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as model,
)

TARGET_NAMES = ("uncheatable", "table9")
N_FOLDS = 3
N_INNER_FOLDS = 3
BOOTSTRAP_SEED = 20260810


def to_model_panel(data) -> model.Panel:
    return model.Panel(data.weights, data.c0, data.c1, data.family_index)


def inner_error(vector, mp, responses, folds, n_families, n_strata) -> float:
    shape, ridge = model.unpack(vector, n_families, n_strata)
    free, constrained = model.design(mp, shape)
    if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
        return 1e6
    total = 0.0
    for train, test in folds:
        scale = np.maximum(responses[train].std(axis=0), 1e-9)
        for column in range(responses.shape[1]):
            b, a = model.fit_head(
                free[train], constrained[train], responses[train, column], ridge, model.pooled_width(mp)
            )
            residual = free[test] @ b + constrained[test] @ a - responses[test, column]
            total += float(residual @ residual) / (scale[column] ** 2)
    return total


def select(mp, responses, folds, n_families, n_strata, seed: int) -> np.ndarray:
    return differential_evolution(
        inner_error,
        model.bounds(n_families, n_strata),
        args=(mp, responses, folds, n_families, n_strata),
        rng=np.random.default_rng(BOOTSTRAP_SEED + seed),
        popsize=12,
        maxiter=60,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def out_of_fold(data, mp, responses, seed: int, shared: bool) -> np.ndarray:
    """Predictions for every target column, selection done inside each outer fold only."""
    outer = panel.grouped_folds(data.frame, seed, N_FOLDS)
    predictions = np.empty_like(responses)
    n_families, n_strata = mp.n_families, mp.n_exposure_strata()
    for train, test in outer:
        inner = panel.grouped_folds(data.frame.iloc[train].reset_index(drop=True), seed + 1000, N_INNER_FOLDS)
        sub = model.Panel(mp.weights[train], mp.epochs_early, mp.epochs_late, mp.family_index)
        free, constrained = None, None
        if shared:
            vector = select(sub, responses[train], inner, n_families, n_strata, seed)
            shape, ridge = model.unpack(vector, n_families, n_strata)
            free, constrained = model.design(mp, shape)
        for column in range(responses.shape[1]):
            if not shared:
                vector = select(sub, responses[train][:, [column]], inner, n_families, n_strata, seed)
                shape, ridge = model.unpack(vector, n_families, n_strata)
                free, constrained = model.design(mp, shape)
            b, a = model.fit_head(
                free[train], constrained[train], responses[train, column], ridge, model.pooled_width(mp)
            )
            predictions[test, column] = free[test] @ b + constrained[test] @ a
    return predictions


def pair_index(data) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Rows of the best tied and best asymmetric policy inside each correspondence group."""
    tied = np.all(np.isclose(data.weights[:, 0, :], data.weights[:, 1, :]), axis=1)
    keys = data.frame["phase_correspondence_key"].astype(str).to_numpy()
    held_rows, moved_rows, group_keys = [], [], []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        moved, held = rows[~tied[rows]], rows[tied[rows]]
        if len(moved) and len(held):
            held_rows.append(held)
            moved_rows.append(moved)
            group_keys.append(key)
    # Plain lists, NOT object arrays: when every group has the same number of rows numpy collapses a
    # list of equal-length index arrays into a 2-D object array, and indexing with a row of that array
    # then fails because its dtype is object rather than int.
    return held_rows, moved_rows, np.array(group_keys)


def gap_errors(observed: np.ndarray, predicted: np.ndarray, held: list, moved: list) -> np.ndarray:
    """Per-group error of the predicted phase contrast; the aggregate level cancels."""
    errors = np.empty(len(held))
    for i, (h, m) in enumerate(zip(held, moved, strict=True)):
        observed_gap = observed[h].min() - observed[m].min()
        predicted_gap = predicted[h].min() - predicted[m].min()
        errors[i] = predicted_gap - observed_gap
    return errors


def main() -> None:
    argv = sys.argv[1:]
    n_boot = int(argv[0]) if argv else 2000
    seeds = [int(s) for s in argv[1:]] or [0, 1, 2]

    datasets = {name: panel.load_300m(name) for name in TARGET_NAMES}
    base = datasets[TARGET_NAMES[0]]
    responses = np.column_stack([datasets[name].y for name in TARGET_NAMES])
    mp = to_model_panel(base)
    held, moved, keys = pair_index(base)

    print("GEN-012: 300M phase-sensitive gate for GEN-002, clustered bootstrap")
    print(f"{base.n} rows, {len(keys)} correspondence groups, targets {TARGET_NAMES}")
    print("diagnostic: RMSE of predicted-minus-observed PHASE GAP per group (aggregate level cancels)")
    print(f"bootstrap: {n_boot} resamples over correspondence GROUPS, seeds {seeds}\n")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))

    for seed in seeds:
        shared_predictions = out_of_fold(base, mp, responses, seed, shared=True)
        independent_predictions = out_of_fold(base, mp, responses, seed, shared=False)
        print(f"seed {seed}:")
        for column, name in enumerate(TARGET_NAMES):
            shared_errors = gap_errors(responses[:, column], shared_predictions[:, column], held, moved)
            independent_errors = gap_errors(responses[:, column], independent_predictions[:, column], held, moved)
            shared_rmse = float(np.sqrt(np.mean(shared_errors**2)))
            independent_rmse = float(np.sqrt(np.mean(independent_errors**2)))

            differences = np.sqrt((shared_errors[draws] ** 2).mean(axis=1)) - np.sqrt(
                (independent_errors[draws] ** 2).mean(axis=1)
            )
            low, high = np.percentile(differences, [2.5, 97.5])
            verdict = "IMPROVES" if high < 0 else ("REGRESSES" if low > 0 else "indistinguishable")
            print(
                f"  {name:12s} shared {shared_rmse:.6f}  independent {independent_rmse:.6f}  "
                f"delta {shared_rmse - independent_rmse:+.6f}  95% CI [{low:+.6f},{high:+.6f}]  {verdict}"
            )
        print()


if __name__ == "__main__":
    main()
