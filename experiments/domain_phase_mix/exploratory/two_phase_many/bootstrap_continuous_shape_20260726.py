# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "scipy",
# ]
# ///
"""Paired bootstrap of continuous shape optimization against the grid.

The two arms differ only in how the shape is chosen, so their metrics are strongly
correlated across resamples and a marginal interval on each arm says nothing about the
sign of the difference. Every draw here resamples once and scores both arms on that same
resample, so the reported quantity is the per-draw difference and its interval.

Resampling is stratified by ``panel.series``: the 300M fit panel is 241 ``qsplit_signal``
rows and 39 ``domain_deletion`` rows, and an unstratified resample would sometimes drop
the deletion series almost entirely and move the comparison for a reason unrelated to
the shape.

The censored set is fixed. It is defined once from the full panel, and each draw
resamples only the training rows, so every draw scores extrapolation against the same
policies. Resampling the censored set too would confound the metric's target with its
estimator.

Selection reruns inside every draw. The claim under test is about a *procedure*, so the
procedure including its shape search has to be exposed to the resample; reusing the
point-estimate shape would understate the variance the search itself contributes.

One consequence of preserving group identity is worth stating. The unresampled 300M
panel has 280 groups for 280 rows, so the harness splitter falls back to shuffled
K-fold. A resample has repeated rows, so grouped K-fold applies and keeps the copies of
a policy in the same fold. That is the correct choice, since letting copies straddle
folds would inflate out-of-fold fit for whichever arm has more capacity, but it means
the bootstrap distribution centres near, not exactly on, the point estimate. The
comparison is unaffected because both arms share the splits within a draw.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from continuous_shape_20260726 import continuous_selector, grid_selector  # noqa: E402
from dual_objective_harness_20260726 import (  # noqa: E402
    build_benchmark,
    fit_metrics,
    fit_on,
    out_of_fold_predictions,
)
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model, Panel  # noqa: E402
from swarm39_models_20260725 import (  # noqa: E402
    _state_shapes,
    build_compact_retained_state,
    build_hierarchical_phase_replay,
)

CENSOR_FRACTION = 0.10
DRAWS = 120
SEED = 20260726
INTERVAL = 95.0

SELECTORS = {"grid": grid_selector, "continuous": continuous_selector}
BUILDERS = {
    "hierarchical_phase_replay": (build_hierarchical_phase_replay, True),
    "compact_retained_state": (build_compact_retained_state, False),
}
L2_GRIDS = {"no_ridge": (0.0,), "cv_ridge": (0.0, 0.01, 0.1, 1.0)}


def _ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values))


def stratified_indices(strata: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample ``pool`` row indices with replacement within each stratum."""
    out = []
    for value in np.unique(strata[pool]):
        members = pool[strata[pool] == value]
        out.append(rng.choice(members, size=len(members), replace=True))
    return np.sort(np.concatenate(out))


def censored_mask(panel: Panel, target: str, fraction: float) -> np.ndarray:
    observed = panel.targets[target]
    available = np.isfinite(observed)
    n_censored = max(1, int(fraction * available.sum()))
    ordering = np.argsort(np.where(available, observed, np.inf))
    mask = np.zeros(len(observed), dtype=bool)
    mask[ordering[:n_censored]] = True
    return mask


@dataclass(frozen=True)
class Draw:
    """One shared resample: 300M full rows, 300M uncensored rows, and 60M rows."""

    fit_300m: np.ndarray
    train_300m: np.ndarray
    fit_60m: np.ndarray


def make_draws(bench, target: str, draws: int, seed: int) -> tuple[list[Draw], np.ndarray]:
    rng = np.random.default_rng(seed)
    censored = censored_mask(bench.fit_300m, target, CENSOR_FRACTION)
    finite_300m = np.flatnonzero(np.isfinite(bench.fit_300m.targets[target]))
    train_pool = np.flatnonzero(np.isfinite(bench.fit_300m.targets[target]) & ~censored)
    finite_60m = np.flatnonzero(np.isfinite(bench.fit_60m.targets[target]))
    series_300m, series_60m = bench.fit_300m.series, bench.fit_60m.series
    out = [
        Draw(
            stratified_indices(series_300m, finite_300m, rng),
            stratified_indices(series_300m, train_pool, rng),
            stratified_indices(series_60m, finite_60m, rng),
        )
        for _ in range(draws)
    ]
    return out, censored


def score_draw(bench, model: Model, target: str, criterion: str, selector, draw: Draw, censored: np.ndarray) -> dict:
    """Fit-quality, fixed-censored-set, and phase metrics for one arm on one resample."""
    oof_panel = bench.fit_300m.subset(draw.fit_300m)
    shape, l2, _ = selector(oof_panel, model, target, criterion, None)
    oof = out_of_fold_predictions(oof_panel, model, target, shape, l2)
    fit = fit_metrics(oof_panel.targets[target], oof)

    # The censored arm trains on a resample of the uncensored rows and always scores the
    # same held-back policies, taken from the unresampled panel.
    train_panel = bench.fit_300m.subset(draw.train_300m)
    shape_c, l2_c, _ = selector(train_panel, model, target, criterion, None)
    fitted = fit_on(train_panel, model, target, shape_c, l2_c)
    truth = bench.fit_300m.targets[target][censored]
    predicted = fitted.predict(bench.fit_300m.subset(censored))
    residual = predicted - truth

    panel_60m = bench.fit_60m.subset(draw.fit_60m)
    shape_p, l2_p, _ = selector(panel_60m, model, target, criterion, None)
    fitted_60m = fit_on(panel_60m, model, target, shape_p, l2_p)
    delta = fitted_60m.predict(bench.paired_300m.two_phase_panel) - fitted_60m.predict(bench.paired_300m.tied_panel)
    phase = phase_decision_skill(delta, bench.paired_300m.observed_delta[target])

    return {
        "oof_rmse": fit["rmse"],
        "oof_spearman": fit["spearman"],
        "oof_low_tail_rmse": fit["low_tail_rmse"],
        "cens_rmse": float(np.sqrt(np.mean(residual**2))),
        "cens_bias": float(np.mean(residual)),
        "cens_abs_bias": float(abs(np.mean(residual))),
        "cens_spearman": float(np.corrcoef(_ranks(predicted), _ranks(truth))[0, 1]),
        "phase_skill": phase["phase_skill_score"],
        "phase_accuracy": phase["decision_accuracy"],
    }


_STATE: dict[str, Any] = {}


def _initializer(model_name: str, l2_name: str, target: str, criterion: str) -> None:
    build, include_threshold = BUILDERS[model_name]
    bench = build_benchmark()
    _STATE["bench"] = bench
    _STATE["model"] = Model(model_name, build, (lambda i=include_threshold: _state_shapes(i)), l2_grid=L2_GRIDS[l2_name])
    _STATE["target"] = target
    _STATE["criterion"] = criterion
    _STATE["censored"] = censored_mask(bench.fit_300m, target, CENSOR_FRACTION)


def _worker(payload: tuple[int, Draw]) -> tuple[int, dict, dict]:
    index, draw = payload
    args = (_STATE["bench"], _STATE["model"], _STATE["target"], _STATE["criterion"])
    grid = score_draw(*args, SELECTORS["grid"], draw, _STATE["censored"])
    continuous = score_draw(*args, SELECTORS["continuous"], draw, _STATE["censored"])
    return index, grid, continuous


def paired_summary(grid: list[dict], continuous: list[dict]) -> dict[str, dict[str, float]]:
    """Per-draw differences (continuous minus grid) with a percentile interval.

    A signed bias has no better direction, so ``fraction_continuous_better`` is omitted
    for it: a bias moving from +0.004 to -0.004 is neither an improvement nor a
    regression. Read the paired ``*_abs_bias`` entry for that judgement.
    """
    low, high = (100.0 - INTERVAL) / 2.0, 100.0 - (100.0 - INTERVAL) / 2.0
    out = {}
    for key in grid[0]:
        a = np.array([row[key] for row in continuous], dtype=float)
        b = np.array([row[key] for row in grid], dtype=float)
        usable = np.isfinite(a) & np.isfinite(b)
        difference = a[usable] - b[usable]
        signed = key.endswith("bias") and "abs" not in key
        lower_is_better = "rmse" in key or "abs_bias" in key
        row = {
            "n_draws": int(usable.sum()),
            "grid_mean": float(b[usable].mean()),
            "continuous_mean": float(a[usable].mean()),
            "diff_mean": float(difference.mean()),
            "diff_lo": float(np.percentile(difference, low)),
            "diff_hi": float(np.percentile(difference, high)),
            "direction": "signed" if signed else ("lower_is_better" if lower_is_better else "higher_is_better"),
        }
        if not signed:
            better = difference < 0 if lower_is_better else difference > 0
            row["fraction_continuous_better"] = float(better.mean())
        out[key] = row
    return out


def run(model_name: str, l2_name: str, target: str, criterion: str, draws: int, processes: int) -> dict:
    bench = build_benchmark()
    draw_list, _ = make_draws(bench, target, draws, SEED)
    grid_rows: list[dict | None] = [None] * draws
    continuous_rows: list[dict | None] = [None] * draws
    with Pool(processes, initializer=_initializer, initargs=(model_name, l2_name, target, criterion)) as pool:
        for index, grid, continuous in pool.imap_unordered(_worker, list(enumerate(draw_list)), chunksize=1):
            grid_rows[index] = grid
            continuous_rows[index] = continuous
    assert all(row is not None for row in grid_rows), "a bootstrap draw did not return"
    return {
        "model": model_name,
        "l2_grid": l2_name,
        "target": target,
        "criterion": criterion,
        "draws": draws,
        "censor_fraction": CENSOR_FRACTION,
        "interval": INTERVAL,
        "paired": paired_summary(grid_rows, continuous_rows),  # type: ignore[arg-type]
    }


def main() -> None:
    combinations = [
        ("hierarchical_phase_replay", "no_ridge", UNCHEATABLE),
        ("hierarchical_phase_replay", "cv_ridge", UNCHEATABLE),
        ("hierarchical_phase_replay", "no_ridge", TABLE9),
        ("hierarchical_phase_replay", "cv_ridge", TABLE9),
        ("compact_retained_state", "no_ridge", UNCHEATABLE),
        ("compact_retained_state", "no_ridge", TABLE9),
    ]
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "bootstrap_continuous_shape.json"
    results = []
    for model_name, l2_name, target in combinations:
        result = run(model_name, l2_name, target, "rmse", DRAWS, 14)
        results.append(result)
        print(f"=== {model_name} {l2_name} {target} ({result['draws']} draws)", flush=True)
        for key, row in result["paired"].items():
            arrow = "better" if row["fraction_continuous_better"] > 0.5 else "worse"
            print(
                f"  {key:18s} grid={row['grid_mean']:+.5f} cont={row['continuous_mean']:+.5f} "
                f"diff={row['diff_mean']:+.5f} [{row['diff_lo']:+.5f},{row['diff_hi']:+.5f}] "
                f"frac_better={row['fraction_continuous_better']:.3f} {arrow}",
                flush=True,
            )
        destination.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
