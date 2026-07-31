# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Fast development-only model selection for 39-bucket surrogates.

This is not a replacement for the frozen nested audit. It is the first rung of the modeling ladder:

1. use one deterministic grouped split of the fit swarm;
2. select nonlinear shape and ridge over three folds;
3. parallelize independent shapes;
4. never inspect the append-only heldout archive.

The full audit repeats model selection inside five outer folds, which multiplies one selection by six
and is appropriate only after a model form is frozen. Running that audit after every mechanism tweak
turns a 280-row fitting problem into tens of thousands of robust constrained regressions.

The selection objective and robust head are otherwise identical to the authoritative harness. A model
that does not improve here is rejected cheaply. A model that does improve is only eligible for, not
confirmed by, the nested and heldout audits.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmark_retained_power_law_swarm39_20260728 as retained_benchmark  # noqa: E402
import swarm39_harness_20260725 as harness  # noqa: E402

DEFAULT_OUTPUT = harness.REFERENCE_OUTPUTS / "fast_surrogate_iteration_20260729.csv"


def _shape_score(
    panel: harness.Panel,
    model: harness.Model,
    target: str,
    observed: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    shape: dict,
) -> tuple[float, dict, float]:
    """Best ridge score for one nonlinear shape."""
    design = model.build(panel, shape).matrix
    multipliers = None if model.penalty_scale is None else model.penalty_scale(panel, shape)
    floor = harness.link_floor(model, shape, observed)
    response = harness.to_link(model, observed, floor)
    solve = model.head or harness.fit_head
    best: tuple[float, dict, float] | None = None
    for ridge in model.l2_grid:
        errors = []
        for train, test in splits:
            intercept, coefficients = solve(design[train], response[train], ridge, multipliers)
            predicted = harness.from_link(model, intercept + design[test] @ coefficients, floor)
            errors.append(predicted - observed[test])
        score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        if best is None or score < best[0]:
            best = (score, shape, ridge)
    assert best is not None
    return best


def _shape_batch_scores(
    panel: harness.Panel,
    model: harness.Model,
    target: str,
    observed: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    shapes: list[dict],
) -> list[tuple[float, dict, float]]:
    """Evaluate a batch in one worker to amortize process-transfer overhead."""
    return [_shape_score(panel, model, target, observed, splits, shape) for shape in shapes]


def fit_model(
    panel: harness.Panel,
    model: harness.Model,
    target: str,
    n_splits: int,
    seed: int,
    workers: int,
) -> harness.Fit:
    """Select a shape in parallel on one deterministic grouped-CV partition."""
    observed = panel.targets[target]
    panel = panel.subset(np.isfinite(observed))
    observed = panel.targets[target]
    splits = harness.grouped_splits(panel, n_splits, seed)
    shapes = list(model.shapes())
    completed = 0
    results: list[tuple[float, dict, float]] = []
    worker_count = min(workers, len(shapes))
    if worker_count == 1:
        results = [_shape_score(panel, model, target, observed, splits, shape) for shape in shapes]
    else:
        # The robust IRLS head contains enough Python work that threads use fewer than two cores.
        # Four batches per worker balance stragglers without serializing the panel for every shape.
        batch_count = min(len(shapes), worker_count * 4)
        batch_size = (len(shapes) + batch_count - 1) // batch_count
        batches = [shapes[start : start + batch_size] for start in range(0, len(shapes), batch_size)]
        executor = ProcessPoolExecutor(max_workers=worker_count)
        futures = []
        try:
            futures = [
                executor.submit(_shape_batch_scores, panel, model, target, observed, splits, batch) for batch in batches
            ]
            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)
                completed += len(batch_results)
                print(f"    evaluated {completed}/{len(shapes)} shapes", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    score, shape, ridge = min(results, key=lambda result: result[0])
    design = model.build(panel, shape)
    floor = harness.link_floor(model, shape, observed)
    solve = model.head or harness.fit_head
    intercept, coefficients = solve(
        design.matrix,
        harness.to_link(model, observed, floor),
        ridge,
        None if model.penalty_scale is None else model.penalty_scale(panel, shape),
    )
    return harness.Fit(
        floor=floor,
        model=model.name,
        shape=shape,
        l2=ridge,
        intercept=intercept,
        coefficients=coefficients,
        names=design.names,
        oof_rmse=score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", choices=("60m", "delphi_3e18"), default=["delphi_3e18"])
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=(harness.UNCHEATABLE, harness.TABLE9),
        default=[harness.UNCHEATABLE],
    )
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.splits < 2:
        raise ValueError("--splits must be at least two")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    rows = []
    for scale in args.scales:
        panel, _heldout = harness.load_scale(scale)
        for target in args.targets:
            model = retained_benchmark.retained_model()
            start = perf_counter()
            fit = fit_model(panel, model, target, args.splits, args.seed, args.workers)
            elapsed = perf_counter() - start
            rows.append(
                {
                    "scale": scale,
                    "target": target,
                    "model": model.name,
                    "fit_rows": len(panel),
                    "splits": args.splits,
                    "seed": args.seed,
                    "workers": args.workers,
                    "selection_oof_rmse": fit.oof_rmse,
                    "ridge": fit.l2,
                    "shape": json.dumps(fit.shape, sort_keys=True),
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                f"{scale} / {target}: OOF {fit.oof_rmse:.6f}, ridge {fit.l2:g}, " f"{elapsed:.1f} seconds",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
