# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Per-component 300M regressions for GEN-002 (GEN-015), as a checked-in driver.

The frozen 300M gate requires component regressions to be reported, because a pooled macro can improve
while individual components get actively worse. This closes two defects an independent review raised
against the earlier attempt.

FIRST, reproducibility. The previous component sweep lived in an untracked scratch file, so nothing in
the repository fitted GEN-002 to the 300M panel and none of its numbers could be regenerated. This
driver is checked in and imports the model directly.

SECOND, a mismatched baseline. The previous sweep compared the model's OUT-OF-FOLD error against the
full-data standard deviation of the target. Those are different estimators: the in-sample mean is a
better predictor than a mean estimated without the test rows, so the baseline was optimistic and every
ratio was harder to beat than a matched comparison. Here the baseline is an intercept fitted on the same
training folds and evaluated on the same held-out rows, so "worse than the mean" means what it says.

Table-9 components live under the ``olmo_base`` prefix, not a ``table9`` name; the only table9-named
column is the macro. Uncheatable components are observed on the 280 two-phase rows while the macro spans
all 520, so each component is fitted on its own observed rows rather than averaged across unset ones.

Usage: ``uv run python ... [seed]``, default 0.
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

N_FOLDS = 3
N_INNER_FOLDS = 3
MIN_ROWS = 100
# Lighter than the WSD80 driver's 12/70 because this sweeps 55 components; stated so the difference is
# visible rather than assumed.
POPSIZE = 8
MAXITER = 35


def component_columns(frame) -> tuple[list[str], list[str]]:
    uncheatable = [
        c
        for c in frame.columns
        if c.startswith("eval_uncheatable_eval_") and c.endswith("_bpb") and c != "eval_uncheatable_eval_bpb"
    ]
    table9 = [c for c in frame.columns if c.startswith("olmo_base") and frame[c].notna().sum() >= 280]
    return sorted(uncheatable), sorted(table9)


def evaluate_component(data, full_panel, column: str, seed: int) -> tuple[float, float, int] | None:
    rows = np.flatnonzero(data.frame[column].notna().to_numpy())
    if len(rows) < MIN_ROWS:
        return None
    y = data.frame[column].to_numpy(dtype=float)[rows]
    here = model.Panel(
        full_panel.weights[rows], full_panel.epochs_early, full_panel.epochs_late, full_panel.family_index
    )
    frame = data.frame.iloc[rows].reset_index(drop=True)
    n_families, n_strata = here.n_families, here.n_exposure_strata()
    bounds = model.bounds(n_families, n_strata)

    predictions = np.empty(len(rows))
    baseline = np.empty(len(rows))
    for train, test in panel.grouped_folds(frame, seed, N_FOLDS):
        inner = panel.grouped_folds(frame.iloc[train].reset_index(drop=True), seed, N_INNER_FOLDS)
        sub = model.Panel(here.weights[train], here.epochs_early, here.epochs_late, here.family_index)
        target = y[train]

        def inner_error(vector, sub=sub, target=target, inner=inner, n_families=n_families, n_strata=n_strata):
            shape, ridge = model.unpack(vector, n_families, n_strata)
            free, constrained = model.design(sub, shape)
            if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
                return 1e6
            total = 0.0
            for a, b in inner:
                head, amplitudes = model.fit_head(free[a], constrained[a], target[a], ridge, model.pooled_width(sub))
                predictions = free[b] @ head + constrained[b] @ amplitudes
                if model.predictions_escape_range(predictions, target[a]):
                    return 1e6
                residual = predictions - target[b]
                total += float(residual @ residual)
            return total

        vector = differential_evolution(
            inner_error,
            bounds,
            rng=np.random.default_rng(20260810 + seed),
            popsize=POPSIZE,
            maxiter=MAXITER,
            tol=1e-10,
            polish=True,
            init="sobol",
        ).x
        shape, ridge = model.unpack(vector, n_families, n_strata)
        free, constrained = model.design(here, shape)
        head, amplitudes = model.fit_head(free[train], constrained[train], y[train], ridge, model.pooled_width(here))
        predictions[test] = free[test] @ head + constrained[test] @ amplitudes
        # Matched baseline: an intercept fitted on the SAME training rows, scored on the SAME test rows.
        baseline[test] = y[train].mean()

    return (
        float(np.sqrt(np.mean((predictions - y) ** 2))),
        float(np.sqrt(np.mean((baseline - y) ** 2))),
        len(rows),
    )


def main() -> None:
    seed = int(sys.argv[1]) if sys.argv[1:] else 0
    data = panel.load_300m("uncheatable")
    full_panel = model.Panel(data.weights, data.c0, data.c1, data.family_index)
    uncheatable, table9 = component_columns(data.frame)

    print("GEN-015: 300M per-component regressions for GEN-002 (corrected rank-truncating solve)")
    print(f"  {len(uncheatable)} Uncheatable components, {len(table9)} Table-9 components, seed {seed}")
    print("  baseline: intercept on the SAME training folds; ratio >= 1 means worse than predicting the mean\n")

    regressions = []
    for label, columns, strip in (
        ("UNCHEATABLE", uncheatable, "eval_uncheatable_eval_"),
        ("TABLE-9", table9, "olmo_base_"),
    ):
        print(f"{label} COMPONENTS")
        for column in columns:
            result = evaluate_component(data, full_panel, column, seed)
            if result is None:
                continue
            rmse, base, n = result
            ratio = rmse / base
            flag = ""
            if ratio >= 1.0:
                regressions.append((column, ratio))
                flag = "   <-- WORSE THAN THE MEAN"
            print(
                f"   {column.replace(strip, '')[:36]:36s} n={n:3d}  RMSE {rmse:.6f}  "
                f"baseline {base:.6f}  ratio {ratio:.3f}{flag}"
            )
        print()

    print(f"components worse than an out-of-fold intercept: {len(regressions)}")
    for column, ratio in sorted(regressions, key=lambda item: -item[1]):
        print(f"   {column}  ratio {ratio:.3f}")


if __name__ == "__main__":
    main()
